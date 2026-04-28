# Simple RAG Agent

A small Retrieval-Augmented Generation (RAG) agent that listens to live UAV GPS telemetry (MQTT), retrieves relevant threat/rule context from a local Chroma vector DB, augments a local quantized LLM prompt, and emits a structured JSON detection result.

## What it does (brief)
- Subscribes to MQTT topic `iobt/uav/gps`
- Retrieves top-3 relevant threat documents from Chroma
- Augments the LLM prompt with retrieved context
- Uses a local quantized GGUF model to generate a JSON detection report:
  `{ attack_suspected, attack_type, confidence, reason }`

## Quick Start

### Prerequisites
- Python 3.8+
- MQTT broker (Mosquitto recommended)
- ~8 GB RAM (4+GB for the model), ~30 GB disk for models
- Create and activate a virtualenv (recommended)

### Install Python deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Or individually:
# pip install paho-mqtt chromadb sentence-transformers llama-cpp-python
