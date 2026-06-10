"""Snapshot garbage collector: removes old/unused KV cache files."""

from pathlib import Path

from cacheflow.store import CacheFlowStore


class SnapshotGC:
    """Garbage-collects unused snapshot files.

    Keeps the current snapshot for each agent and removes old files.
    """

    def __init__(self, store: CacheFlowStore, snapshots_dir: Path):
        self.store = store
        self.snapshots_dir = snapshots_dir

    def collect(
        self,
        keep_latest_n: int = 1,
        dry_run: bool = False,
        older_than_days: int | None = None,
    ) -> list[Path]:
        """Remove old snapshot files.

        Args:
            keep_latest_n: Keep only the current snapshot (keep_latest_n is ignored)
            dry_run: If True, return the list without deleting anything
            older_than_days: Not used in simplified mode

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

        for f in self.snapshots_dir.glob("*.bin"):
            if f.name.startswith(".tmp_"):
                # Orphaned temp files from crashed sessions
                if not dry_run:
                    f.unlink(missing_ok=True)
                deleted.append(f)
            elif f.name not in referenced:
                if not dry_run:
                    f.unlink(missing_ok=True)
                deleted.append(f)

        return deleted
