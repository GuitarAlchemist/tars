import subprocess
import json
import sys
import os
import time

def send_request(process, request):
    json_req = json.dumps(request)
    sys.stderr.write(f"Sending: {json_req}\n")
    process.stdin.write(json_req + "\n")
    process.stdin.flush()

def read_response(process):
    # Read until we get a valid JSON object or EOF
    while True:
        line = process.stdout.readline()
        if not line:
            return None
        
        # trim whitespace
        line = line.strip()
        if not line:
            continue
            
        sys.stderr.write(f"Received raw: {line}\n")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"Failed to decode JSON: {line}\n")
            continue

def main():
    # Use the compiled exe directly
    cmd = [
        r"c:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\bin\Debug\net10.0\Tars.Interface.Cli.exe", 
        "mcp", "server"
    ]
    
    sys.stderr.write(f"Starting TARS MCP Server: {' '.join(cmd)}\n")
    
    # Needs to be text mode for line-buffering simplicity
    process = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=sys.stderr, # Forward stderr to see logs
        text=True,
        bufsize=1
    )

    try:
        # Give it a moment to start (compile etc)
        # Note: In a real scenario we might wait for a specific log message, 
        # but since logs go to stderr and we are forwarding them, we just see them.
        # We start by sending initialize.
        
        # 1. Initialize
        init_req = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {"name": "test-client", "version": "1.0"}
            },
            "id": 1
        }
        send_request(process, init_req)
        
        # The dotnet run might take a while to output anything if it needs to build.
        init_resp = read_response(process)
        if init_resp:
            print("Initialize Response:", json.dumps(init_resp, indent=2))
        else:
            print("No response to initialize")
            return

        # 2. Notify initialized
        notify_init = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        json_req = json.dumps(notify_init)
        process.stdin.write(json_req + "\n")
        process.stdin.flush()

        # 3. List Tools
        list_req = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        send_request(process, list_req)
        list_resp = read_response(process)
        if list_resp:
            print("List Tools Response:", json.dumps(list_resp, indent=2))
            
            # Check if we got tools
            tools = list_resp.get("result", {}).get("tools", [])
            print(f"Found {len(tools)} tools.")
            for t in tools:
                if "dummy" in t["name"]:
                    print(f"Confirmed Tool: {t['name']}")
                    
                    # 4. Call the tool
                    call_req = {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": t['name'],
                            "arguments": {
                                "name": "Antigravity"
                            }
                        },
                        "id": 3
                    }
                    send_request(process, call_req)
                    call_resp = read_response(process)
                    print("Tool Call Response:", json.dumps(call_resp, indent=2))
            # Check for Memory Tools
            tool_names = [t["name"] for t in tools]
            if "search_memory" in tool_names and "save_memory" in tool_names:
                print("SUCCESS: Memory Tools found via MCP!")
                
                # Call save_memory
                call_req = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "save_memory",
                        "arguments": {
                            "fact": "Project TARS v2 is progressing well."
                        }
                    },
                    "id": 99
                }
                send_request(process, call_req)
                call_resp = read_response(process)
                print("Save Memory Response:", json.dumps(call_resp, indent=2))
            else:
                print(f"WARNING: Memory tools missing! Tools found: {tool_names}")


    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    main()
