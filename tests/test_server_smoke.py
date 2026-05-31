"""Smoke test for LlamaServer with save/restore."""

import sys
from pathlib import Path
import tempfile

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentgit.server import LlamaServer
from agentgit.config import find_gguf_for_model, compute_model_hash


def find_model():
    """Find a GGUF model on disk."""
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
        print(f"   Run: ollama pull llama3.1:8b")
        sys.exit(1)

    # Find smallest
    model_path = min(ggufs, key=lambda p: p.stat().st_size)
    return str(model_path)


def main():
    """Run smoke test."""
    print("🔥 AgentGit Smoke Test\n")

    # Find model
    model_path = find_model()
    size_gb = Path(model_path).stat().st_size / (1024**3)
    print(f"✓ Found model: {Path(model_path).name} ({size_gb:.1f} GB)")

    # Create temp directory for snapshots
    with tempfile.TemporaryDirectory() as tmpdir:
        slot_save_path = Path(tmpdir) / "snapshots"
        slot_save_path.mkdir()

        # Start server
        print("\n[1/5] Starting LlamaServer...")
        server = LlamaServer()
        server.start(
            model_path=model_path,
            slot_save_path=str(slot_save_path),
            ctx_size=2048,
            n_gpu_layers=99,
        )
        print(f"✓ Server started on port {server.port}")

        try:
            # Test completion
            print("\n[2/5] Testing completion...")
            response = server.completion(
                prompt="Say hello in exactly 3 words.",
                slot_id=0,
                max_tokens=20,
            )

            content = response["content"].strip()
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]

            print(f"✓ Completion received")
            print(f"  Response: {content}")
            print(f"  Tokens: {prompt_tokens} prompt + {completion_tokens} completion")

            # Verify response is non-empty
            if not content:
                print("❌ Response is empty!")
                sys.exit(1)

            # Test save_slot
            print("\n[3/5] Testing save_slot()...")
            save_result = server.save_slot(slot_id=0)
            save_time_ms = save_result["save_time_ms"]
            snapshot_size_mb = save_result["size_bytes"] / (1024**2)
            filename = save_result["filename"]

            print(f"✓ Slot saved")
            print(f"  Filename: {filename}")
            print(f"  Size: {snapshot_size_mb:.1f} MB")
            print(f"  Save time: {save_time_ms}ms")

            # Verify file exists
            snapshot_file = slot_save_path / filename
            if not snapshot_file.exists():
                print(f"❌ Snapshot file not found: {snapshot_file}")
                sys.exit(1)

            # Test restore_slot
            print("\n[4/5] Testing restore_slot()...")
            restore_result = server.restore_slot(filename=filename, slot_id=0)
            restore_time_ms = restore_result["restore_time_ms"]

            print(f"✓ Slot restored")
            print(f"  Restore time: {restore_time_ms}ms")

            if restore_time_ms > 10000:
                print(f"\n⚠ Warning: Restore took {restore_time_ms}ms (>10s)")
                print("  Consider reducing ctx_size if this becomes a bottleneck")

            # Stop server
            print("\n[5/5] Cleanup...")
            server.stop()
            print("✓ Server stopped")

            # Final result
            print("\n" + "=" * 50)
            print("PASS ✓")
            print("=" * 50)
            print(f"\nSummary:")
            print(f"  Model: {Path(model_path).name}")
            print(f"  Context size: 2048 tokens")
            print(f"  Response: {content}")
            print(f"  Completion tokens: {completion_tokens}")
            print(f"  Snapshot size: {snapshot_size_mb:.1f} MB")
            print(f"  Save time: {save_time_ms}ms")
            print(f"  Restore time: {restore_time_ms}ms")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            server.stop()
            sys.exit(1)


if __name__ == "__main__":
    main()
