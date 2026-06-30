"""Snapshot garbage collector: removes old/unused KV cache files."""

import time
from pathlib import Path

from cacheflow.store import CacheFlowStore

# Grace period before an unreferenced snapshot file becomes eligible for
# reaping. A freshly saved snapshot is written under a transient name
# (`slot_<id>_<hex>.bin`) and only becomes referenced once the owning session
# renames it to its agent-HEAD name and updates the store — a window of a few
# milliseconds. With concurrent agents (W3/W5), one agent's post-session GC can
# otherwise observe another agent's in-flight save as "unreferenced" and delete
# it out from under it, causing "Snapshot file not created by server". Only
# reaping files older than this grace period closes that race while still
# cleaning genuine orphans (which are always older) on a later GC pass.
DEFAULT_GRACE_SECONDS = 60.0


class SnapshotGC:
    """Garbage-collects unused snapshot files.

    Keeps the current snapshot for each agent and removes old files.
    """

    def __init__(self, store: CacheFlowStore, snapshots_dir: Path,
                 grace_seconds: float = DEFAULT_GRACE_SECONDS):
        self.store = store
        self.snapshots_dir = snapshots_dir
        self.grace_seconds = grace_seconds

    def collect(self, dry_run: bool = False) -> list[Path]:
        """Remove snapshot files no longer referenced by any agent's HEAD,
        plus orphaned .tmp_ files from crashed sessions.

        Files modified within `grace_seconds` are left alone so a concurrent
        session's in-flight save (not yet promoted to its agent HEAD) is never
        reaped mid-rename.

        Args:
            dry_run: If True, return the list without deleting anything

        Returns:
            List of paths that were deleted (or would be deleted on dry_run)
        """
        referenced: set[str] = set()

        # Keep the current snapshot for each agent
        agents = self.store.list_agents()
        for agent in agents:
            if agent.current_snapshot_path:
                snapshot_name = Path(agent.current_snapshot_path).name
                referenced.add(snapshot_name)

        deleted: list[Path] = []

        if not self.snapshots_dir.exists():
            return deleted

        now = time.time()

        def _too_young(f: Path) -> bool:
            try:
                return now - f.stat().st_mtime < self.grace_seconds
            except OSError:
                # Vanished between glob and stat (another GC raced us) — skip it.
                return True

        for f in self.snapshots_dir.glob("*.bin"):
            if f.name.startswith(".tmp_"):
                # Orphaned temp files from crashed sessions
                if _too_young(f):
                    continue
                if not dry_run:
                    f.unlink(missing_ok=True)
                deleted.append(f)
            elif f.name not in referenced:
                if _too_young(f):
                    continue
                if not dry_run:
                    f.unlink(missing_ok=True)
                deleted.append(f)

        return deleted
