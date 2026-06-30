"""Repo cloning, measurement, and cf init for benchmark corpus."""

import subprocess
import sys
from pathlib import Path

from benchmarks.config import BenchConfig, REPO_CORPUS


def count_loc(repo_path: Path) -> tuple[int, int]:
    """Return (total_lines, file_count) for Python source files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        files = [f for f in result.stdout.splitlines() if f]
        total_lines = 0
        for f in files:
            fpath = repo_path / f
            if fpath.exists():
                try:
                    total_lines += sum(1 for _ in fpath.open(errors="ignore"))
                except OSError:
                    pass
        return total_lines, len(files)
    except Exception:
        return 0, 0


def clone_repo(repo_key: str, repos_dir: Path) -> Path:
    """Clone repo if not already present; return its local path."""
    info = REPO_CORPUS[repo_key]
    url = info["url"]
    dest = repos_dir / repo_key
    if dest.exists():
        return dest

    cmd = ["git", "clone", "--depth=1"]
    sparse = info.get("sparse_checkout")
    if sparse:
        cmd += ["--filter=blob:none", "--sparse"]
    cmd += [url, str(dest)]

    print(f"  cloning {url} → {dest}")
    subprocess.run(cmd, check=True, capture_output=True)

    if sparse:
        subprocess.run(
            ["git", "sparse-checkout", "set"] + sparse,
            cwd=dest, check=True, capture_output=True
        )
    return dest


def init_repo(repo_path: Path, cfg: BenchConfig) -> bool:
    """Initialize a repo's .cacheflow config. Returns True if succeeded."""
    import json as _json
    cf_dir = repo_path / ".cacheflow"
    config_path = cf_dir / "config.json"
    if config_path.exists():
        return True

    # Resolve the active model directly — avoids interactive cf init prompt.
    try:
        from cacheflow.cli import _discover_models
        from cacheflow.config import compute_model_hash
        models = _discover_models()  # list of (label, name, path)
        if not models:
            print(f"  [warn] no models discovered for {repo_path}")
            return False
        # Use the active model from cf model list (first entry is active in ollama order)
        # Prefer qwen3 if available, otherwise first model
        chosen = models[0]
        for label, name, path in models:
            if "qwen3" in name.lower():
                chosen = (label, name, path)
                break
        _, model_name, model_path = chosen
        model_hash = compute_model_hash(model_path) if model_path else ""
        cf_dir.mkdir(parents=True, exist_ok=True)
        (cf_dir / "snapshots").mkdir(exist_ok=True)
        snap_path = str(cf_dir / "snapshots")
        config = {
            "model_path": model_path,
            "model_name": model_name,
            "model_hash": model_hash,
            "ctx_size": cfg.ctx_size,
            "n_gpu_layers": cfg.n_gpu_layers if cfg.n_gpu_layers >= 0 else 99,
            "slot_save_path": snap_path,
        }
        config_path.write_text(_json.dumps(config, indent=2))
        return True
    except Exception as e:
        print(f"  [warn] init failed for {repo_path}: {e}")
        return False


def apply_mutation(repo_path: Path, mutation_type: str) -> dict:
    """Apply a synthetic codebase mutation and return metadata."""
    import time, random, string

    tag = "".join(random.choices(string.ascii_lowercase, k=6))
    metadata = {"mutation_type": mutation_type, "tag": tag}

    if mutation_type == "add_file":
        target = repo_path / f"_bench_added_{tag}.py"
        target.write_text(f'"""Benchmark-injected file {tag}."""\ndef bench_{tag}(): pass\n')
        metadata["path"] = str(target.relative_to(repo_path))

    elif mutation_type == "delete_file":
        py_files = list(repo_path.rglob("*.py"))
        py_files = [f for f in py_files if ".cacheflow" not in str(f) and f.stat().st_size < 5000]
        if py_files:
            target = random.choice(py_files)
            metadata["path"] = str(target.relative_to(repo_path))
            target.rename(target.with_suffix(".py.bak"))

    elif mutation_type == "rename_function":
        py_files = list(repo_path.rglob("*.py"))
        py_files = [f for f in py_files if ".cacheflow" not in str(f)]
        for f in random.sample(py_files, min(10, len(py_files))):
            try:
                text = f.read_text(errors="ignore")
                lines = text.splitlines()
                func_lines = [(i, l) for i, l in enumerate(lines) if l.strip().startswith("def ")]
                if func_lines:
                    idx, line = random.choice(func_lines)
                    old_name = line.strip().split("(")[0].replace("def ", "")
                    new_name = f"{old_name}_{tag}"
                    new_text = text.replace(f"def {old_name}(", f"def {new_name}(", 1)
                    f.write_text(new_text)
                    metadata["path"] = str(f.relative_to(repo_path))
                    metadata["old_name"] = old_name
                    metadata["new_name"] = new_name
                    break
            except OSError:
                continue

    return metadata


def revert_mutation(repo_path: Path, metadata: dict):
    """Attempt to revert a mutation (best-effort)."""
    mtype = metadata.get("mutation_type")
    path_str = metadata.get("path")
    if not path_str:
        return
    target = repo_path / path_str
    if mtype == "add_file" and target.exists():
        target.unlink()
    elif mtype == "delete_file":
        bak = target.with_suffix(".py.bak")
        if bak.exists():
            bak.rename(target)
    elif mtype == "rename_function":
        old_name = metadata.get("old_name")
        new_name = metadata.get("new_name")
        if old_name and new_name and target.exists():
            text = target.read_text(errors="ignore")
            target.write_text(text.replace(f"def {new_name}(", f"def {old_name}(", 1))


def setup_repos(cfg: BenchConfig) -> dict[str, Path]:
    """Clone and init all configured repos. Returns {repo_key: path}."""
    paths = {}
    for key in cfg.repos:
        print(f"[repos] setting up {key}")
        path = clone_repo(key, cfg.repos_dir)
        loc, files = count_loc(path)
        print(f"  {key}: {loc:,} LOC, {files:,} files")
        init_repo(path, cfg)
        paths[key] = path
    return paths
