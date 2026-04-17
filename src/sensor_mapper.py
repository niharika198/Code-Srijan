import json
import os
import pandas as pd
import datetime

DATA_FILE = "data/live_sensor_data.jsonl"
MAPPED_FILE = "data/live_energy_metrics.json"

def parse_sensor_logger_payload(line):
    try:
        data = json.loads(line)
        metrics = {
            "light_lux": 0,
            "noise_db": 0,
            "battery_level": 100,
            "timestamp": data.get("server_timestamp", datetime.datetime.now().isoformat())
        }
        
        # Handle custom simple payload (like our test)
        if "lux" in data and "payload" not in data:
            metrics["light_lux"] = data.get("lux", 0)
            metrics["noise_db"] = data.get("decibels", 0)
            return metrics
            
        # Handle actual Sensor Logger payload
        if "payload" in data:
            for item in data["payload"]:
                name = item.get("name", "")
                values = item.get("values", {})
                
                if name == "light":
                    metrics["light_lux"] = values.get("lux", 0)
                elif name == "microphone":
                    metrics["noise_db"] = values.get("dBFS", 0)
                elif name == "battery":
                    metrics["battery_level"] = values.get("level", 100)
                    
        return metrics
    except Exception as e:
        print(f"Error parsing line: {e}")
        return None

def map_sensors_to_energy(metrics):
    if not metrics:
        return None
        
    # Heuristics for converting phone sensors to kW
    
    # 1. Lighting: High lux = lights on (if it's night) or natural light. 
    # For simplicity, let's say lux > 100 means lights are on (0.2 kW)
    lighting_kw = 0.2 if metrics["light_lux"] > 100 else 0.0
    
    # 2. Appliances: Loud noise (e.g. TV, washing machine) = power usage
    # dBFS is usually negative (0 is loudest). If it's above -40, something is running.
    # If custom test payload uses positive dB, handle that too.
    noise = metrics["noise_db"]
    appliance_kw = 0.0
    if noise > 0: # our mock data
        if noise > 50: appliance_kw = 1.5
        elif noise > 30: appliance_kw = 0.5
    else: # real sensor logger dBFS
        if noise > -30: appliance_kw = 1.5
        elif noise > -50: appliance_kw = 0.5
        
    # 3. Base Load (Fridge, etc.)
    base_kw = 0.3
    
    total_live_kw = lighting_kw + appliance_kw + base_kw
    
    return {
        "timestamp": metrics["timestamp"],
        "raw_lux": metrics["light_lux"],
        "raw_noise": metrics["noise_db"],
        "live_lighting_kw": lighting_kw,
        "live_appliance_kw": appliance_kw,
        "live_base_kw": base_kw,
        "total_live_kw": total_live_kw
    }

def get_latest_metrics():
    if not os.path.exists(DATA_FILE):
        return None
        
    # Read the last line of the file
    last_line = ""
    with open(DATA_FILE, 'r') as f:
        lines = f.readlines()
        if lines:
            last_line = lines[-1]
            
    if not last_line:
        return None
        
    raw_metrics = parse_sensor_logger_payload(last_line)
    energy_metrics = map_sensors_to_energy(raw_metrics)
    
    # Save the mapped output
    if energy_metrics:
        with open(MAPPED_FILE, 'w') as f:
            json.dump(energy_metrics, f)
            
    return energy_metrics

if __name__ == "__main__":
    latest = get_latest_metrics()
    print("Latest mapped metrics:", latest)
