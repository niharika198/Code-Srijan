# SmartVolt AI: Kinetic Energy Telemetry & Prediction

SmartVolt AI is a high-fidelity energy telemetry platform that bridges the gap between IoT sensor data and predictive machine learning. By fusing real-time environmental metrics (light, noise, thermal) with historical consumption patterns, SmartVolt provides actionable insights to eliminate energy waste and optimize building efficiency.

---

## 🚀 Key Features

### 1. Live Telemetry Fusion
- **Sensor Logger Integration**: Stream real-time data from mobile devices (Lux, dBFS, Battery Temp).
- **Inferred Infrastructure Load**: Dynamically calculates lighting and appliance power draw based on ambient environment.
- **Thermal Intelligence**: Inferred room temperature and optimal AC setpoint recommendations based on phone battery thermal gradients.

### 2. Predictive AI Engine
- **Load Forecasting**: Linear regression models trained on multi-home datasets to predict daily and monthly energy consumption.
- **Washing Machine Analysis**: Dedicated tracking for high-load appliances with peak-hour optimization suggestions.
- **Baseline Anomaly Detection**: Identifies when current usage deviates significantly from predicted efficient baselines.

### 3. Kinetic Observatory Dashboard
- **Glassmorphic UI**: A premium, "no-line" editorial design system for complex telemetry.
- **Automated Sync**: Real-time fragments that update the dashboard every 2 seconds without full-page reloads.
- **Waste Stream Detection**: Real-time alerts for "Daylight Harvesting" waste and idle load anomalies.

---

## 🛠️ Project Structure

```bash
├── dashboard/
│   └── app.py              # Main Streamlit Application
├── src/
│   ├── ingest_server.py    # HTTP POST server for IoT data
│   ├── sensor_mapper.py    # Heuristic engine & sensor-to-kW logic
│   ├── predictor.py        # ML Prediction class
│   └── train_model_2.py    # AI Training pipeline
├── data/
│   ├── live_sensor_data.jsonl   # Raw ingestion buffer
│   └── live_energy_metrics.json # Mapped telemetry output
└── models/                 # Serialized ML model artifacts
```

---

## 🏃 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 2. Launch the Ingestion Server
This process receives the data from your phone.
```bash
python src/ingest_server.py
```
*The server runs on `http://localhost:8000/sensor`.*

### 3. Launch the Dashboard
Open a second terminal and run:
```bash
streamlit run dashboard/app.py
```

---

## 📱 Connecting Your Phone (Live Mode)

1.  Install the **Sensor Logger** app (available on iOS/Android).
2.  In Settings, enable **HTTP Push**.
3.  Set the URL to your computer's IP address (or use Ngrok for remote testing):
    - `http://<YOUR_IP>:8000/sensor`
4.  Select the following sensors: **Light**, **Microphone**, and **Battery**.
5.  Press **Start Logging** to see live updates in the SmartVolt Dashboard!

---

## 📜 Deployment

SmartVolt AI is designed for multi-process deployment.
- **Frontend**: Streamlit Community Cloud.
- **Backend**: Render or Railway (requires persistent storage for the `.jsonl` buffer).
- **Public Bridge**: Use Ngrok to expose the ingestion server during local development.

---

© 2026 SmartVolt AI. Built for the AntiGravity Project.