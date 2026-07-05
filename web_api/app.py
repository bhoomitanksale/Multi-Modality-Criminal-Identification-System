"""
Global Face Database API
Runs a standalone HTTP server that mimics a massive web database interpol-style check.
"""
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s [WEB-API] %(message)s')

# Mock "Global" Database (Famous criminals "on the web" but not in local DB)
WEB_CRIMINALS = [
    {
        "id": "INT-101",
        "name": "Dawood Ibrahim",
        "alias": "Bhai",
        "crime_type": "Organized Crime / Terrorism",
        "risk_level": "CRITICAL",
        "status": "Wanted - Interpol Red Notice"
    },
    {
        "id": "INT-102",
        "name": "Chhota Rajan",
        "alias": "Nana",
        "crime_type": "Murder / Extortion",
        "risk_level": "CRITICAL",
        "status": "In Custody (Simulated Match)"
    },
    {
        "id": "INT-103",
        "name": "Tiger Memon",
        "alias": "Tiger",
        "crime_type": "Terrorism",
        "risk_level": "CRITICAL",
        "status": "Wanted - Interpol"
    },
    {
        "id": "INT-901",
        "name": "Alex 'The Ghost' Mercer",
        "alias": "The Ghost",
        "crime_type": "International Cyber Extortion",
        "risk_level": "HIGH",
        "status": "At Large"
    }
]

class WebAPIHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps({"status": "Online", "database_size": "8,432,192 faces"}).encode('utf-8'))

    def do_POST(self):
        if self.path == "/api/v1/identify_faces":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data)
                batch_features = data.get("batch", [])
                
                logging.info(f"Received search request for batch of {len(batch_features)} faces.")
                
                results = []
                import random
                for i, features in enumerate(batch_features):
                    # logic: Professional Internet-Based Criminal Search
                    # We distinguish between "Live Webcam", "Internet Image", and "Sketch Reconstruction"
                    # using the hidden source flag in features[24].
                    
                    source_flag = features[24] if len(features) > 24 else 0
                    
                    # source_flag == 1.0 -> Internet Image (Famous)
                    # source_flag == 2.0 -> Sketch Reconstruction (Suspect)
                    is_famous_search = (source_flag == 1.0)
                    is_sketch_search = (source_flag == 2.0)
                    
                    if is_famous_search:
                        match = random.choice(WEB_CRIMINALS)
                        results.append({
                            "match_found": True, 
                            "confidence": 1.0, 
                            "data": match,
                            "source": "Global Interpol Database (Verified)"
                        })
                    elif is_sketch_search:
                        # Ensure deterministic match for the Dawood sketch demo
                        match = next((c for c in WEB_CRIMINALS if c["id"] == "INT-101"), WEB_CRIMINALS[0])
                        results.append({
                            "match_found": True, 
                            "confidence": 0.98, 
                            "data": match,
                            "source": "Neural Reconstruction Engine (Web Match)"
                        })
                    else:
                        # For normal live webcam scans, match with only 0.1% chance
                        if random.random() < 0.001:
                            match = random.choice(WEB_CRIMINALS)
                            results.append({
                                "match_found": True, 
                                "confidence": 1.0, 
                                "data": match,
                                "source": "Global Web Database"
                            })
                        else:
                            results.append({"match_found": False})
                
                self._set_headers(200)
                self.wfile.write(json.dumps({"results": results}).encode('utf-8'))
                
            except json.JSONDecodeError:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))

        elif self.path == "/api/v1/sketch_to_face":
            # Simulation: Takes a sketch and returns success metadata
            logging.info("Processing Sketch-to-Face reconstruction...")
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "success": True, 
                "reconstruction_id": "RECON-008",
                "message": "Face successfully reconstructed from sketch biometrics."
            }).encode('utf-8'))

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=WebAPIHandler, port=5000):
    server_address = ('127.0.0.1', port)
    httpd = server_class(server_address, handler_class)
    logging.info(f"Interpol Web API Data-Center Starting on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info("Web API Offline.")

if __name__ == '__main__':
    run()
