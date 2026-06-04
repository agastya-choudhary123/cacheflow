#!/usr/bin/env python3
"""
Validate that custom llama KV cache save/restore works.
"""

import subprocess
import time
import json
import sys
import os
from pathlib import Path
import requests

def find_gguf():
    """Find the smallest GGUF model on disk."""
    search_paths = [
        Path.home() / ".ollama/models/blobs",
        Path.home() / "Library/Caches/llama.cpp",
        Path.cwd(),
    ]

    ggufs = []
    for path in search_paths:
        if path.exists():
            ggufs.extend(path.rglob("*.gguf"))

    if not ggufs:
        print("❌ No GGUF model found!")
        print(f"   Searched: {', '.join(str(p) for p in search_paths)}")
        print(f"   Run: ollama pull qwen2.5-coder:7b")
        sys.exit(1)

    # Find smallest
    model_path = min(ggufs, key=lambda p: p.stat().st_size)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Found model: {model_path} ({size_mb:.1f} MB)")
    return str(model_path)

def main():
    # Find model
    model_path = find_gguf()

    # Create .agentgit directory and snapshots directory
    agentgit_dir = Path.cwd() / ".agentgit"
    agentgit_dir.mkdir(exist_ok=True)
    snapshots_dir = agentgit_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)

    # Start custom llama server
    print("\n[1/6] Starting custom llama server...")
    server_log = agentgit_dir / "server.log"

    # Find the custom server script
    custom_server = Path(__file__).parent.parent / "agentgit" / "llama_server_custom.py"
    if not custom_server.exists():
        print(f"❌ Custom server script not found at {custom_server}")
        sys.exit(1)

    proc = subprocess.Popen(
        [
            sys.executable,
            str(custom_server),
            model_path,
            "--ctx-size", "2048",
            "--port", "8080",
            "--slot-save-path", str(snapshots_dir),
        ],
        stdout=open(server_log, "w"),
        stderr=subprocess.STDOUT,
    )

    # Wait for startup
    time.sleep(5)
    for i in range(30):
        try:
            r = requests.get("http://127.0.0.1:8080/health", timeout=1)
            if r.status_code == 200:
                print("✓ Custom llama server started")
                break
        except:
            pass
        if i < 29:
            time.sleep(0.5)
    else:
        print("❌ Custom llama server did not start in 30 seconds")
        proc.terminate()
        with open(server_log) as f:
            print("Server log:")
            print(f.read()[-500:])
        sys.exit(1)

    api_schema = {}

    try:
        # Test completion endpoint
        print("\n[2/6] Testing /completion endpoint...")
        r = requests.post(
            "http://127.0.0.1:8080/completion",
            json={
                "prompt": "Hello in 3 words:",
                "slot_id": 0,
                "n_predict": 10,
                "cache_prompt": True,
            },
            timeout=60,
        )
        if r.status_code != 200:
            print(f"❌ /completion returned {r.status_code}")
            print(r.text)
            sys.exit(1)

        completion_response = r.json()
        api_schema["completion"] = {k: type(v).__name__ for k, v in completion_response.items()}

        usage = completion_response.get("usage", {})
        print(f"✓ Completion endpoint works")
        print(f"  - Usage: {usage}")
        print(f"  - Response: {completion_response.get('content', '')[:50]}...")

        # Test slot list
        print("\n[3/6] Testing /slots endpoint...")
        r = requests.get("http://127.0.0.1:8080/slots", timeout=30)
        if r.status_code != 200:
            print(f"❌ /slots returned {r.status_code}")
            sys.exit(1)

        slots_response = r.json()
        api_schema["slots"] = "list"
        print(f"✓ Slot list endpoint works")
        print(f"  - Slots: {len(slots_response)}")

        # Test slot save
        print("\n[4/6] Testing /slots/0/save endpoint...")
        r = requests.post("http://127.0.0.1:8080/slots/0/save", timeout=60)
        if r.status_code != 200:
            print(f"❌ /slots/0/save returned {r.status_code}: {r.text}")
            sys.exit(1)

        save_response = r.json()
        api_schema["save"] = {k: type(v).__name__ for k, v in save_response.items()}
        saved_filename = save_response.get("filename")
        save_time_ms = save_response.get("save_time_ms")
        print(f"✓ Slot save endpoint works")
        print(f"  - Filename: {saved_filename}")
        print(f"  - Save time: {save_time_ms}ms")

        # Verify file exists
        saved_file = snapshots_dir / saved_filename
        if saved_file.exists():
            size_mb = saved_file.stat().st_size / (1024 * 1024)
            print(f"  - File size: {size_mb:.1f} MB")
        else:
            print(f"❌ Saved file not found: {saved_file}")
            sys.exit(1)

        # Test slot restore
        print("\n[5/6] Testing /slots/0/restore endpoint...")
        r = requests.post(
            "http://127.0.0.1:8080/slots/0/restore",
            json={"filename": saved_filename},
            timeout=60,
        )
        if r.status_code != 200:
            print(f"❌ /slots/0/restore returned {r.status_code}: {r.text}")
            sys.exit(1)

        restore_response = r.json()
        api_schema["restore"] = {k: type(v).__name__ for k, v in restore_response.items()}
        restore_time_ms = restore_response.get("restore_time_ms")
        print(f"✓ Slot restore endpoint works")
        print(f"  - Restore time: {restore_time_ms}ms")

        # Final summary
        print("\n[6/6] Summary")
        print("✓ All prerequisite checks passed!")

        if restore_time_ms > 10000:
            print(f"\n⚠ Warning: Restore took {restore_time_ms}ms (>10s)")
            print("  Consider reducing ctx_size if this becomes a bottleneck")

        # Save schema
        with open(agentgit_dir / "api_schema.json", "w") as f:
            json.dump(api_schema, f, indent=2)
        print(f"\n✓ API schema saved to {agentgit_dir / 'api_schema.json'}")

        # Save server info
        with open(agentgit_dir / "server_info.json", "w") as f:
            json.dump({
                "server_type": "custom_llama_cpp_python",
                "model_path": model_path,
                "ctx_size": 2048,
                "save_time_ms": save_time_ms,
                "restore_time_ms": restore_time_ms,
            }, f, indent=2)

    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
