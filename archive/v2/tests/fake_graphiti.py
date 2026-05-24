from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

PORT = 8001


# Global in-memory store
episodes = []

class MockGraphiti(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/healthcheck':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            if self.path.startswith('/episodes/') or self.path.startswith('/v1/episodes/'):
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                data = json.loads(body.decode('utf-8'))
                print(f"Graphiti Received Episode: {data}")
                episodes.append(data)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"id": "mock-uuid"}).encode('utf-8'))
            
            elif self.path == '/search':
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                query_data = json.loads(body.decode('utf-8'))
                query = query_data.get("query", "").lower()
                print(f"Graphiti Received Search: {query}")
                results = []
                for ep in episodes:
                    ep_str = json.dumps(ep).lower()
                    if query in ep_str or True:
                        fact = ep.get("Content", ep.get("belief", ep.get("Body", str(ep))))
                        results.append({
                            "uuid": ep.get("uuid", "mock-uuid-123"),
                            "name": ep.get("Name", "Memory"),
                            "score": 1.0, 
                            "fact": fact
                        })
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(results).encode('utf-8'))

            elif self.path == '/messages' or self.path == '/v1/messages':
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                data = json.loads(body.decode('utf-8'))
                print(f"Graphiti Received Messages: {data}")
                # data is { "messages": [...] } usually
                msgs = data.get("messages", [])
                if not msgs and isinstance(data, list): msgs = data
                
                for m in msgs:
                    # Normalize to episode-like structure for search
                    ep = {"Content": m.get("content", str(m)), "Name": "Message"}
                    episodes.append(ep)

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"id": "mock-uuid"}).encode('utf-8'))

            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            print(f"ERROR in POST: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))


def run(server_class=HTTPServer, handler_class=MockGraphiti):
    server_address = ('', 8001)
    httpd = server_class(server_address, handler_class)
    print(f'Starting Mock Graphiti on {PORT}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
