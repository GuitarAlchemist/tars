import subprocess
import json
import sys
import os
import time
import uuid

# --- Reuse Helper Functions (In real implementation, move to shared module) ---

def send_request(process, request):
    json_req = json.dumps(request)
    process.stdin.write(json_req + "\n")
    process.stdin.flush()

def read_response(process, timeout=5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if not line: return None
        line = line.strip()
        if not line: continue
        try: return json.loads(line)
        except json.JSONDecodeError: continue
    return None

def fail(msg):
    print(f"[EVAL FAIL]: {msg}")
    sys.exit(1)

# --- Eval Logic ---

def main():
    print("Running Cognitive Eval: Memory Recall (Needle in Haystack)...")
    
    exe_path = r"c:\Users\spare\source\repos\tars\v2\src\Tars.Interface.Cli\bin\Debug\net10.0\Tars.Interface.Cli.exe"
    cmd = [exe_path, "mcp", "server"]
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)

    try:
        # 1. Handshake
        send_request(process, {"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "eval", "version": "1.0"}}})
        if not read_response(process): fail("Handshake failed")
        send_request(process, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

        # 2. Check Tools
        send_request(process, {"jsonrpc": "2.0", "method": "tools/list", "id": 2, "params": {}})
        tools_resp = read_response(process)
        tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
        
        if "save_memory" not in tool_names or "search_memory" not in tool_names:
            print("[EVAL SKIP]: Memory tools not available (Graphiti missing?).")
            sys.exit(0) # Skip, don't fail, if infra is missing

        # 3. Inject Needle
        needle = str(uuid.uuid4())
        fact = f"The secret eval code is {needle}"
        print(f"Injecting Needle: {fact}")
        
        save_req = {
            "jsonrpc": "2.0", "method": "tools/call", "id": 3,
            "params": {"name": "save_memory", "arguments": {"fact": fact}}
        }
        send_request(process, save_req)
        save_resp = read_response(process)
        if save_resp.get("error"): fail(f"Save failed: {save_resp['error']}")
        
        # 4. Search Needle
        print("Searching Needle...")
        # Add slight delay for ingestion if needed (Mock is instant, Real might need index refresh)
        time.sleep(10) 
        
        search_req = {
            "jsonrpc": "2.0", "method": "tools/call", "id": 4,
            "params": {"name": "search_memory", "arguments": {"query": "What is the secret eval code?"}}
        }
        send_request(process, search_req)
        search_resp = read_response(process)
        
        if search_resp.get("error"): fail(f"Search failed: {search_resp['error']}")
        
        content = search_resp["result"]["content"][0]["text"]
        print(f"Search Result: {content}")
        
        if needle in content:
            print("[EVAL PASS]: Needle found in haystack!")
            sys.exit(0)
        else:
            fail(f"Needle '{needle}' NOT found in response.")

    except Exception as e:
        fail(f"Exception: {e}")
    finally:
        process.terminate()

if __name__ == "__main__":
    main()
