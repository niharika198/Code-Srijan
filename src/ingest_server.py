import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import datetime

DATA_FILE = "data/live_sensor_data.jsonl"

class SensorDataHandler(BaseHTTPRequestHandler):
    def _set_response(self, code=200):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            payload = json.loads(post_data.decode('utf-8'))
            self._save_data(payload)
            self._set_response(200)
            self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
        except Exception as e:
            print(f"Error parsing data: {e}")
            self._set_response(400)
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))

    def _save_data(self, payload):
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        
        # We append each payload as a new line containing a JSON object (JSONL format)
        # to make it easy to parse dynamically without loading a huge array.
        payload['server_timestamp'] = datetime.datetime.now().isoformat()
        
        with open(DATA_FILE, 'a') as f:
            f.write(json.dumps(payload) + '\n')
        
        print(f"[{payload['server_timestamp']}] Received data. Keys: {list(payload.keys())}")

def run(server_class=HTTPServer, handler_class=SensorDataHandler, port=8000):
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting ingestion server on port {port}...")
    print(f"Configure Sensor Logger to send HTTP POST to: http://<your_computer_ip>:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Stopping ingestion server.")

if __name__ == '__main__':
    run()
