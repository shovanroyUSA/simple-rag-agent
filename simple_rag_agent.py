"""
Simple RAG Agent for live UAV GPS telemetry.

What it does:
1) Builds/loads a small Chroma knowledge base (threat profiles + rules)
2) Subscribes to MQTT topic: iobt/uav/gps
3) For each GPS message:
   - retrieves top-3 relevant threat chunks from Chroma
   - sends telemetry + retrieved context to local GGUF model
   - prints a simple detection result

Run (example):
    python3 simple_rag_agent.py

Required packages:
    pip install paho-mqtt chromadb sentence-transformers llama-cpp-python
"""

import os
import json
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt
import chromadb
from chromadb.utils import embedding_functions
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# =========================
# 1) Basic configuration
# =========================

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "iobt/uav/gps"
MODEL_PATH = Path("/home/user/models") / "qwen2-1_5b-instruct-q4_k_m.gguf"
MODEL_CANDIDATES = [
    Path("/home/user/models") / "qwen2-1_5b-instruct-q4_k_m.gguf",
    Path("/home/user/models") / "Phi-3-mini-128k-instruct.Q4_K_M.gguf",
    #Path("/home/user/models") / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    Path("/home/user/models") / "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
]
# Path to your quantized model (GGUF)
# MODEL_PATH = os.getenv("MODEL_PATH", "./models/qwen2-1_5b-instruct-q4_k_m.gguf")

# Chroma persistence folder
CHROMA_PATH = "./chroma_simple_rag_db"
COLLECTION_NAME = "gps_threat_knowledge"

# Keep generation short and deterministic
LLM_MAX_TOKENS = 220
LLM_TEMPERATURE = 0.1

# Track last altitude per device to detect jumps
LAST_ALTITUDE = {}


# =========================
# 2) Very small knowledge base
# =========================
# Add/replace threat/rule text later.
SEED_DOCS = [
    "Threat: GPS spoofing Sybil attack. Symptom: multiple conflicting location signals from same device.",
    "Threat: GPS spoofing Custom attack. Symptom: Exact gps latitude 64.1 and gps longitude -97.6 is a known decoy location.",
    "Threat: GPS spoofing teleport jump. Symptom: sudden large location jump in very short time.",
    "Threat: Impossible speed signature. Symptom: speed exceeds realistic UAV limits.",
    "Threat: Altitude spoofing. Symptom: altitude not consistent with flight profile.",
    "Rule: If multiple location signals from same device < 5 seconds, flag possible Sybil attack.",
    "Rule: If GPS latitude = 64.1 and longitude = -97.6, always flag Confirm Custom attack regardless of other values.",
    "Rule: If distance jump > 500 meters in < 2 seconds, flag possible spoofing.",
    "Rule: If implied speed > 70 m/s for this UAV profile, mark as suspicious.",
    "Rule: If altitude changes > 80 meters between consecutive samples, check spoofing risk.",
]


# =========================
# 3) Initialize Chroma
# =========================

def init_chroma():
    """Create/load Chroma collection and seed data once."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Local embedding model
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn
    )

    # Seed only if empty
    existing = collection.count()
    if existing == 0:
        ids = [f"doc_{i}" for i in range(len(SEED_DOCS))]
        metas = [{"source": "seed"} for _ in SEED_DOCS]
        collection.add(ids=ids, documents=SEED_DOCS, metadatas=metas)
        print(f"[RAG] Seeded Chroma with {len(SEED_DOCS)} docs.")
    else:
        print(f"[RAG] Loaded existing Chroma collection with {existing} docs.")

    return collection


# =========================
# 4) Initialize local GGUF model
# =========================

def init_llm():
    """Load quantized local model using llama.cpp binding."""
    env_model_path = os.getenv("MODEL_PATH")
    candidate_paths = [Path(env_model_path)] if env_model_path else []
    candidate_paths.extend(MODEL_CANDIDATES)

    existing_candidates = []
    seen = set()
    for path in candidate_paths:
        path_str = str(path)
        if path_str not in seen:
            seen.add(path_str)
            if path.exists():
                existing_candidates.append(path)

    if not existing_candidates:
        raise FileNotFoundError(
            "No compatible GGUF model file found. Checked: "
            + ", ".join(str(p) for p in candidate_paths)
        )

    last_error = None
    for model_path in existing_candidates:
        try:
            llm = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=max(1, os.cpu_count() // 2),
                n_gpu_layers=0,
                verbose=False
            )
            print(f"[LLM] Loaded model: {model_path}")
            return llm
        except Exception as exc:
            last_error = exc
            print(f"[LLM] Failed to load {model_path}: {exc}")

    raise RuntimeError(f"Could not load any local GGUF model: {last_error}")


# =========================
# 5) Helpers
# =========================

def telemetry_to_text(gps: dict) -> str:
    """Convert GPS JSON message into plain text for retrieval + prompt."""
    device = gps.get("device", "unknown")
    lat = gps.get("latitude", "na")
    lon = gps.get("longitude", "na")
    alt = gps.get("altitude", "na")
    acc = gps.get("accuracy", "na")
    ts = gps.get("timestamp", datetime.utcnow().isoformat())

    alt_delta = "na"
    altitude_jump = "unknown"
    try:
        alt_value = float(alt)
        prev_alt = LAST_ALTITUDE.get(device)
        if prev_alt is not None:
            alt_delta = abs(alt_value - prev_alt)
            altitude_jump = "YES" if alt_delta > 80 else "NO"
        LAST_ALTITUDE[device] = alt_value
    except (TypeError, ValueError):
        pass

    return (
        f"UAV telemetry | device={device} | lat={lat} | lon={lon} | "
        f"alt={alt}m | accuracy={acc}m | timestamp={ts} | "
        f"alt_delta={alt_delta}m | altitude_jump={altitude_jump}"
    )


def retrieve_top3(collection, query_text: str):
    """Retrieve top-3 relevant knowledge chunks from Chroma with scores."""
    res = collection.query(
        query_texts=[query_text],
        n_results=3,
        include=["documents", "distances"]
    )
    docs = res.get("documents", [[]])[0]
    distances = res.get("distances", [[]])[0]
    similarities = [max(0.0, 1.0 - d) for d in distances]
    return list(zip(docs, similarities))


def ask_llm(llm, telemetry_text: str, top_docs: list[str]) -> str:
    """Ask the model to detect possible spoofing using retrieved context."""
    context = "\n".join([f"- {d}" for d in top_docs]) if top_docs else "- No context retrieved."

    prompt = f"""
You are a UAV GPS security assistant.
Use only the telemetry and retrieved context below.

Telemetry:
{telemetry_text}

Retrieved threat context:
{context}

Return STRICT JSON with keys:
attack_suspected (true/false),
attack_type (string),
confidence (0 to 1),
reason (short string).
"""

    # Simple completion call
    out = llm(
        prompt,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        stop=["\n\n\n"]
    )
    text = out["choices"][0]["text"].strip()
    return text


# =========================
# 6) MQTT callbacks
# =========================

def build_mqtt_client(collection, llm):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="simple-rag-agent")

    def on_connect(c, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] Connected. Subscribing to:", MQTT_TOPIC)
            c.subscribe(MQTT_TOPIC, qos=1)
        else:
            print("[MQTT] Connection failed with code:", rc)

    def on_message(c, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            gps = json.loads(payload)

            telemetry_text = telemetry_to_text(gps)
            top_docs = retrieve_top3(collection, telemetry_text)
            result = ask_llm(llm, telemetry_text, [d for d, _ in top_docs])

            print("\n" + "=" * 60)
            print("[Telemetry]")
            print(telemetry_text)
            print("\n[Top-3 Retrieved Threat Docs]")
            for i, (d, similarity) in enumerate(top_docs, start=1):
                print(f"{i}. {d} (similarity={similarity:.4f})")
            print("\n[LLM Detection Result]")
            print(result)
            print("=" * 60 + "\n")

        except Exception as e:
            print("[ERROR] Failed processing MQTT message:", e)

    client.on_connect = on_connect
    client.on_message = on_message
    return client


# =========================
# 7) Main
# =========================

def main():
    print("[START] Simple RAG Agent")
    collection = init_chroma()
    llm = init_llm()

    client = build_mqtt_client(collection, llm)
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except OSError as exc:
        print(
            f"[MQTT] Could not connect to broker at {MQTT_BROKER}:{MQTT_PORT} - {exc}."
        )
        print("[MQTT] Start a local broker and rerun the agent.")
        return

    print("[RUNNING] Waiting for live GPS telemetry...")
    client.loop_forever()


# How to test the RAG agent: 
# Open a new terminal
# Generate data and publish into a topic: iobt/uav/gps to trigger the agent, example payload (Copy):
# mosquitto_pub -t iobt/uav/gps -m '{"device":"UAV-GPS","latitude":65.1,"longitude":-97.6,"altitude":100.0,"accuracy":1.0,"timestamp":"2026-03-12T19:05:00Z"}' && sleep 1 && mosquitto_pub -t iobt/uav/gps -m '{"device":"UAV-GPS","latitude":55.0000,"longitude":55.0000,"altitude":100.0,"accuracy":1.0,"timestamp":"2026-03-12T19:05:01Z"}' && sleep 1 && mosquitto_pub -t iobt/uav/gps -m '{"device":"UAV-SPEED","latitude":64.1,"longitude":-97.6,"altitude":120.0,"accuracy":1.0,"timestamp":"2026-03-12T19:05:10Z"}' && sleep 1 && mosquitto_pub -t iobt/uav/gps -m '{"device":"UAV-SPEED","latitude":64.2000,"longitude":-97.7000,"altitude":120.0,"accuracy":1.5,"timestamp":"2026-3-12T19:5:11Z"}'

if __name__ == "__main__":
    main()