import sys
import json

def log(msg):
    sys.stderr.write(f"[DUMMY] {msg}\n")
    sys.stderr.flush()

def main():
    log("Starting Dummy MCP Server")
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
                
            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method")
            
            response = {
                "jsonrpc": "2.0",
                "id": req_id
            }
            
            if method == "initialize":
                response["result"] = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "dummy-server", "version": "1.0"}
                }
            elif method == "tools/list":
                response["result"] = {
                    "tools": [
                        {
                            "name": "hello_world",
                            "description": "Returns a greeting. Input: string name.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}}
                            }
                        }
                    ]
                }
            elif method == "tools/call":
                params = request.get("params", {})
                name = params.get("name")
                args = params.get("arguments", {})
                
                if name == "hello_world":
                    user_name = args.get("name", "World")
                    response["result"] = {
                        "content": [
                            {"type": "text", "text": f"Hello, {user_name} from Dummy Server!"}
                        ],
                        "isError": False
                    }
                else:
                    response["error"] = {"code": -32601, "message": "Method not found"}
            else:
                # notifications or pings
                if req_id is not None:
                    response["result"] = {}
            
            if req_id is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
        except Exception as e:
            log(f"Error: {e}")

if __name__ == "__main__":
    main()
