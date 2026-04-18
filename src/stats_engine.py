import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from src.sensor_mapper import parse_sensor_logger_payload, map_sensors_to_energy

DATA_FILE = "data/live_sensor_data.jsonl"
STATS_CACHE = "data/statistical_insights.json"

class StatisticalEngine:
    def __init__(self, data_file=DATA_FILE):
        self.data_file = data_file
        self.df = None

    def load_data(self, limit=1000):
        """Loads the most recent records from the JSONL file and processes them."""
        if not os.path.exists(self.data_file):
            return False
            
        data_list = []
        try:
            with open(self.data_file, 'r') as f:
                # Read last 'limit' lines for performance
                lines = f.readlines()
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                
                for line in recent_lines:
                    raw = parse_sensor_logger_payload(line)
                    if raw:
                        mapped = map_sensors_to_energy(raw)
                        if mapped:
                            data_list.append(mapped)
            
            if not data_list:
                return False
                
            self.df = pd.DataFrame(data_list)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def get_summary_stats(self):
        """Calculates basic descriptive statistics."""
        if self.df is None or self.df.empty:
            return {}
            
        cols = ['live_lighting_kw', 'live_appliance_kw', 'live_ac_kw', 'total_live_kw']
        summary = self.df[cols].describe().to_dict()
        return summary

    def detect_anomalies(self, threshold=2.0):
        """Detects spikes using Z-Score."""
        if self.df is None or self.df.empty:
            return []
            
        mean = self.df['total_live_kw'].mean()
        std = self.df['total_live_kw'].std()
        
        if std == 0:
            return []
            
        self.df['z_score'] = (self.df['total_live_kw'] - mean) / std
        anomalies = self.df[abs(self.df['z_score']) > threshold]
        anomalies_list = anomalies[['timestamp', 'total_live_kw', 'z_score']].tail(5).to_dict(orient='records')
        # Convert Timestamps to strings for JSON serialization
        for a in anomalies_list:
            if isinstance(a['timestamp'], pd.Timestamp):
                a['timestamp'] = a['timestamp'].isoformat()
        
        return anomalies_list

    def calculate_waste_probability(self):
        """
        Uses heuristic-statistical approach to estimate waste probability.
        High Waste = High lights when lux is already high, or high noise with no activity.
        """
        if self.df is None or self.df.empty:
            return 0.0
            
        recent = self.df.iloc[-1]
        
        # 1. Daylight Waste: Lights are on despite high ambient light
        # If raw_lux > 300 and lighting_kw > 0.1, it's likely wasteful
        daylight_waste = 1.0 if (recent['raw_lux'] > 300 and recent['live_lighting_kw'] > 0.05) else 0.0
        
        # 2. Phantom Load / Standby Waste: 
        # If noise is very low (silence) but appliance_kw is significant
        phantom_waste = 1.0 if (recent['raw_noise'] < -60 and recent['live_appliance_kw'] > 0.5) else 0.0
        
        # 3. AC Overcooling:
        # If room temp < 22 and AC is still drawing high power
        cooling_waste = 1.0 if (recent['inferred_room_temp'] < 21.0 and recent['live_ac_kw'] > 0.5) else 0.0
        
        prob = (daylight_waste * 0.4) + (phantom_waste * 0.3) + (cooling_waste * 0.3)
        return round(prob * 100, 2)

    def get_correlations(self):
        """Analyzes correlation between sensors and energy."""
        if self.df is None or self.df.empty:
            return {}
            
        corr_matrix = self.df[['raw_lux', 'raw_noise', 'raw_temp', 'total_live_kw']].corr()
        return corr_matrix['total_live_kw'].to_dict()

    def generate_report(self):
        """Generates a full statistical report and saves it to a JSON file."""
        if not self.load_data():
            return {"error": "No data available"}
            
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_summary_stats(),
            "anomalies": self.detect_anomalies(),
            "waste_probability": self.calculate_waste_probability(),
            "correlations": self.get_correlations(),
            "sample_size": len(self.df)
        }
        
        with open(STATS_CACHE, 'w') as f:
            json.dump(report, f, indent=4)
            
        return report

if __name__ == "__main__":
    engine = StatisticalEngine()
    print("Running statistical analysis...")
    report = engine.generate_report()
    print(json.dumps(report, indent=2))
