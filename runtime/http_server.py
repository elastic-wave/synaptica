# runtime/http_server.py  (TRT-LLM version)
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, os
from tensorrt_llm.runtime import ModelRunner

ENGINE_DIR = os.environ.get("TRTLLM_ENGINE_DIR", "releases/TinyLlama-ctx2048-fp16-kvfp16-mb1-pg128")

runner = ModelRunner.from_dir(ENGINE_DIR)  # loads config + engine

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(size).decode())
        prompt = data.get("prompt", "")
        max_new = int(data.get("max_new_tokens", 128))
        temp    = float(data.get("temperature", 0.7))

        out = runner.generate([prompt], max_new_tokens=max_new, temperature=temp)
        text = out[0] if isinstance(out, list) else out

        self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(json.dumps({"response": text}).encode())

if __name__ == "__main__":
    print("TRT-LLM server starting…")
    HTTPServer(("0.0.0.0", 5000), H).serve_forever()
