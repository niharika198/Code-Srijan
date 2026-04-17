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
            max_noise = -160.0
            max_lux = 0.0
            
            for item in data["payload"]:
                name = item.get("name", "")
                values = item.get("values", {})
                
                if name == "light":
                    max_lux = max(max_lux, values.get("lux", 0))
                elif name == "microphone":
                    max_noise = max(max_noise, values.get("dBFS", -160))
                elif name == "battery":
                    metrics["battery_level"] = values.get("level", 100)
                    
            metrics["light_lux"] = max_lux
            metrics["noise_db"] = max_noise if max_noise > -160 else 0
            
        return metrics
    except Exception as e:
        print(f"Error parsing line: {e}")
        return None

def map_sensors_to_energy(metrics):
    if not metrics:
        return None
        
    # Heuristics for converting phone sensors to kW
    
    # 1. Lighting: High lux = lights on (if it's night) or natural light. 
    # Let's make it proportional so the dashboard feels highly responsive to your phone!
    # For every 1 Lux, we add 0.005 kW, capping out at 1.0 kW (to simulate max brightness)
    lux = metrics["light_lux"]
    lighting_kw = min(lux * 0.005, 1.0)
    
    # 2. Appliances (Voice/Noise): Loud noise = power usage
    # dBFS is usually negative (0 is loudest, -80 is very quiet).
    # Let's make this proportional so it reacts smoothly to your voice!
    noise = metrics["noise_db"]
    if noise > 0: # handling our earlier mock data test
        appliance_kw = min(noise * 0.02, 2.0)
    else: # real phone microphone data
        # Normalize the noise: Assume -80 is silence (0 kW) and 0 is max volume (2.0 kW)
        clamped_noise = max(noise, -80) # prevent it from dropping below -80
        # This converts -80 to 0.0, and 0 to 1.0
        intensity = (clamped_noise + 80) / 80.0 
        appliance_kw = intensity * 2.0
        
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

def get_recent_history(limit=50):
    if not os.path.exists(DATA_FILE):
        return []
        
    history = []
    from collections import deque
    with open(DATA_FILE, 'r') as f:
        lines = deque(f, limit)
        for line in lines:
            raw = parse_sensor_logger_payload(line)
            if raw:
                history.append(raw)
    return history

if __name__ == "__main__":
    latest = get_latest_metrics()
    print("Latest mapped metrics:", latest)
