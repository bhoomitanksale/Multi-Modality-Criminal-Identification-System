import json
import urllib.request
import threading

API_URL = "http://127.0.0.1:5000/api/v1/identify_faces"

def search_web_faces_batch(features_list: list) -> dict:
    """
    Sends a batch of 32-dim face feature vectors to the Global Web API.
    Returns: {"success": True, "results": [{"match_found": bool, "data": ...}, ...]}
    """
    if not features_list:
        return {"success": True, "results": []}

    try:
        # Pre-process features for JSON friendliness
        batch_data = []
        for fv in features_list:
             batch_data.append([float(f) for f in fv])
        
        data = json.dumps({"batch": batch_data}).encode('utf-8')
        req = urllib.request.Request(API_URL, data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=5.0) as response:
            if response.status == 200:
                resp_json = json.loads(response.read().decode('utf-8'))
                return {"success": True, "results": resp_json.get("results", [])}
            
    except Exception as e:
        return {"success": False, "error": f"Web API unreachable: {e}"}
        
    return {"success": False, "error": "Unknown error during batch search."}

def search_web_faces_async(features_list: list, callback):
    """Run search_web_faces_batch in a background thread."""
    def worker():
        result = search_web_faces_batch(features_list)
        if callback:
            callback(result)
    threading.Thread(target=worker, daemon=True).start()
    
def translate_sketch_to_face(image_path: str) -> dict:
    """
    Sends a sketch image to the AI Reconstruction API.
    Returns: {"success": True, "reconstruction_id": str}
    """
    try:
        # Simulation: In a real system, we would upload the image file.
        # Since the API is a mock, we just call the endpoint.
        req = urllib.request.Request("http://127.0.0.1:5000/api/v1/sketch_to_face", data=b"{}", headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5.0) as response:
            if response.status == 200:
                return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        return {"success": False, "error": str(e)}
    return {"success": False, "error": "Unknown error"}

def translate_sketch_async(image_path: str, callback):
    def worker():
        result = translate_sketch_to_face(image_path)
        if callback:
            callback(result)
    threading.Thread(target=worker, daemon=True).start()

