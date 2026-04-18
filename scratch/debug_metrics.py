import sys
import os
sys.path.append(os.getcwd())
import src.sensor_mapper as sm

latest = sm.get_latest_metrics()
print(latest)
