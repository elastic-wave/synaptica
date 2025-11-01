from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class SimpleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode()
        data = json.loads(body)
        prompt = data.get('prompt', '(no prompt)')
        reply = f"[Dummy Engine Reply] Received prompt: {prompt}"
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'response': reply}).encode())

if __name__ == '__main__':
    print('Starting dummy LLM HTTP server on port 5000...')
    server = HTTPServer(('0.0.0.0', 5000), SimpleHandler)
    server.serve_forever()