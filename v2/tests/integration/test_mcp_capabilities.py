import subprocess
import json
import sys
import os
import time

# --- Helper Functions ---

def send_request(process, request):
    json_req = json.dumps(request)
    # sys.stderr.write(f"Sending: {json_req}\n")
    process.stdin.write(json_req + "\n")
    process.stdin.flush()

def read_response(process, timeout=5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if not line:
            return None
        
        line = line.strip()
        if not line:
            continue
            
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None

def fail(msg):
    print(f"FAILED: {msg}")
    sys.exit(1)

# --- Main Test Logic ---

def main():
    print("Running Integration Test: MCP Capabilities...")
    
    # Path to TARS executable
    exe_path = r"c:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\bin\Debug\net10.0\Tars.Interface.Cli.exe"
    if not os.path.exists(exe_path):
        fail(f"Executable not found at {exe_path}. Please build the project.")

    cmd = [exe_path, "mcp", "server"]
    
    process = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=sys.stderr, 
        text=True,
        bufsize=1
    )

    try:
        # 1. Initialize
        print("Step 1: Handshake...")
        init_req = {
            "jsonrpc": "2.0", "method": "initialize", "id": 1,
            "params": {
                "protocolVersion": "2024-11-05", 
                "capabilities": {},
                "clientInfo": {"name": "integration-test", "version": "1.0"}
            }
        }
        send_request(process, init_req)
        init_resp = read_response(process)
        
        if not init_resp or "result" not in init_resp:
            fail("Handshake failed. No valid response.")
        
        # Notify initialized
        send_request(process, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        print("  -> Handshake OK")

        # 2. Verify Tool Registration
        print("Step 2: verifying Tool Registration...")
        list_req = {"jsonrpc": "2.0", "method": "tools/list", "id": 2, "params": {}}
        send_request(process, list_req)
        list_resp = read_response(process)
        
        if not list_resp or "result" not in list_resp:
            fail("Failed to list tools.")

        tools = list_resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        
        # Check for core memory tools
        required_tools = ["search_memory", "save_memory"]
        missing = [t for t in required_tools if t not in tool_names]
        
        if missing:
            print(f"  Warning: Missing memory tools: {missing}. Is Graphiti running?")
            # We don't fail here because integration tests might run without Graphiti
            # But we warn heavily.
        else:
            print(f"  -> Memory tools found: {required_tools}")

        # Check for standard tools
        if "list_files" not in tool_names:
            fail("Standard 'list_files' tool missing!")
        
        print("  -> Tool Registration OK")

        print("Integration Test PASSED.")
        sys.exit(0)

    except Exception as e:
        fail(f"Exception during test: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    main()
