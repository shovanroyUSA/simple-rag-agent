# Simple RAG Agent

A lightweight Retrieval-Augmented Generation (RAG) agent that listens to live UAV GPS telemetry and flags suspicious activity using a local quantized LLM plus a Chroma vector database.

## What It Does

- Subscribes to MQTT topic `iobt/uav/gps`
- Retrieves the top-3 relevant threat rules from Chroma
- Augments the LLM prompt with retrieved context
- Returns a JSON detection decision (attack type, confidence, reason)

## How to Run

### 1. Install Dependencies
```bash
pip install paho-mqtt chromadb sentence-transformers llama-cpp-python

### 2. Download a Quantized Model
```bash
python dl_models.py

### 3. Start a MQTT Broker
```bash
mosquitto -p 1883

### 4. Run the Agent
```bash
python simple_rag_agent.py



## How to Test

### Test Input (Publish GPS Telemetry)
Open a new terminal.
```bash
mosquitto_pub -t iobt/uav/gps -m '{"device":"UAV-GPS","latitude":65.1,"longitude":-97.6,"altitude":100.0,"accuracy":1.0,"timestamp":"2026-03-12T19:05:00Z"}'


