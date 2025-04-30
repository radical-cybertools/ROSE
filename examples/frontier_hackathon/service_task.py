from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Initial mock state
service_state = {
    "status": "starting",
    "message": "Initializing service..."
}

class SimpleServiceHandler(BaseHTTPRequestHandler):

    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            self._set_headers()
            self.wfile.write(json.dumps(service_state).encode('utf-8'))
        elif self.path == "/":
            self._set_headers()
            self.wfile.write(json.dumps({"message": "Service is up and running."}).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not Found"}).encode('utf-8'))

    def do_POST(self):
        if self.path == "/status":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                new_state = json.loads(body.decode('utf-8'))
                service_state.update(new_state)
                self._set_headers()
                self.wfile.write(json.dumps(service_state).encode('utf-8'))
            except json.JSONDecodeError:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not Found"}).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=SimpleServiceHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Mock service running on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
