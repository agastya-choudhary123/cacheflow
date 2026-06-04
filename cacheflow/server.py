"""
LlamaServer: Process manager for the custom llama.cpp server with KV cache save/restore.
"""

import subprocess
import time
import httpx
import shutil
from pathlib import Path
from typing import Optional, Dict, Any


class LlamaServer:
    """Manages a llama server subprocess and provides the completion API."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.base_url: Optional[str] = None
        self.http_client: Optional[httpx.Client] = None
        self.ctx_size: Optional[int] = None
        self.log_file: Optional[object] = None

    def start(
        self,
        model_path: str,
        slot_save_path: str,
        ctx_size: int = 8192,
        n_gpu_layers: int = 99,
    ) -> None:
        """
        Start llama server as a subprocess.

        Args:
            model_path: Path to GGUF model file
            slot_save_path: Directory to save KV cache snapshots
            ctx_size: Context size (IMMUTABLE once set)
            n_gpu_layers: Number of GPU layers to use (-1 = CPU only)
        """
        # Find custom server script
        custom_server = Path(__file__).parent / "llama_server_custom.py"
        if not custom_server.exists():
            raise FileNotFoundError(f"Custom server script not found: {custom_server}")

        # Find available port
        self.port = self._find_available_port(8080, 8090)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.ctx_size = ctx_size

        # Create slot save directory
        Path(slot_save_path).mkdir(parents=True, exist_ok=True)

        # Get project root for logging
        project_root = Path(__file__).parent.parent
        log_dir = project_root / ".cacheflow"
        log_dir.mkdir(exist_ok=True)
        server_log = log_dir / "server.log"

        # Start subprocess (store log file handle for cleanup)
        self.log_file = open(server_log, "w")
        self.process = subprocess.Popen(
            [
                "python3",
                str(custom_server),
                model_path,
                "--ctx-size",
                str(ctx_size),
                "--port",
                str(self.port),
                "--slot-save-path",
                slot_save_path,
            ],
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        self._wait_for_health(timeout=60)
        self.http_client = httpx.Client(timeout=300.0)

    def _find_available_port(self, start: int, end: int) -> int:
        """Find an available port in the given range."""
        import socket

        for port in range(start, end + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No available ports in range {start}-{end}")

    def _wait_for_health(self, timeout: int = 30) -> None:
        """Poll until server is healthy."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=1)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"Server did not start within {timeout} seconds")

    def is_running(self) -> bool:
        """Check if server is running and healthy."""
        if not self.process or self.process.poll() is not None:
            return False
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=1)
            return r.status_code == 200
        except:
            return False

    def stop(self) -> None:
        """Stop the server."""
        if self.http_client:
            self.http_client.close()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log_file:
            self.log_file.close()

    def completion(
        self,
        prompt: str,
        slot_id: int = 0,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Run a completion request.

        Args:
            prompt: Input prompt
            slot_id: Slot ID to use
            max_tokens: Maximum tokens to generate

        Returns:
            Response dict with keys: content, tokens_evaluated, tokens_predicted, usage, etc.
        """
        if not self.http_client:
            raise RuntimeError("Server not started")

        response = self.http_client.post(
            f"{self.base_url}/completion",
            json={
                "prompt": prompt,
                "slot_id": slot_id,
                "n_predict": max_tokens,
                "cache_prompt": True,
            },
        )
        response.raise_for_status()
        return response.json()

    def prime_slot(self, prefix: str, slot_id: int = 0) -> Dict[str, Any]:
        """Reset the model and eval a stable prefix, establishing the KV baseline.

        Args:
            prefix: Stable prefix text (system prompt + codebase)
            slot_id: Slot ID to prime

        Returns:
            Dict with keys: n_tokens, prime_time_ms
        """
        if not self.http_client:
            raise RuntimeError("Server not started")

        response = self.http_client.post(
            f"{self.base_url}/slots/{slot_id}/prime",
            json={"prefix": prefix},
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()

    def save_slot(self, slot_id: int = 0) -> Dict[str, Any]:
        """
        Save KV cache for a slot to disk.

        Args:
            slot_id: Slot ID to save

        Returns:
            Dict with keys: filename, save_time_ms, size_bytes
        """
        if not self.http_client:
            raise RuntimeError("Server not started")

        response = self.http_client.post(f"{self.base_url}/slots/{slot_id}/save")
        response.raise_for_status()
        return response.json()

    def restore_slot(self, filename: str, slot_id: int = 0) -> Dict[str, Any]:
        """
        Restore KV cache for a slot from disk.

        Args:
            filename: Snapshot filename to restore
            slot_id: Slot ID to restore to

        Returns:
            Dict with keys: filename, restore_time_ms
        """
        if not self.http_client:
            raise RuntimeError("Server not started")

        response = self.http_client.post(
            f"{self.base_url}/slots/{slot_id}/restore",
            json={"filename": filename},
        )
        response.raise_for_status()
        return response.json()

    def erase_slot(self, slot_id: int = 0) -> None:
        """Clear KV cache for a slot."""
        if not self.http_client:
            raise RuntimeError("Server not started")

        response = self.http_client.post(f"{self.base_url}/slots/{slot_id}/erase")
        response.raise_for_status()

    def list_slots(self) -> list:
        """List all available slots."""
        if not self.http_client:
            raise RuntimeError("Server not started")

        response = self.http_client.get(f"{self.base_url}/slots")
        response.raise_for_status()
        return response.json()
