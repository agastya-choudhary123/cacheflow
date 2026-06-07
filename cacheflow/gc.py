"""Snapshot garbage collector: removes unreferenced KV cache files."""

from pathlib import Path

from cacheflow.store import CacheFlowStore


class SnapshotGC:
    """Garbage-collects unreferenced snapshot files.

    A snapshot is reclaimable when no commit record's snapshot_path references
    it. The GC always retains the latest keep_latest_n snapshots per agent
    (for fast restore) and the current HEAD snapshot.
    """

    def __init__(self, store: CacheFlowStore, snapshots_dir: Path):
        self.store = store
        self.snapshots_dir = snapshots_dir

    def collect(self, keep_latest_n: int = 3, dry_run: bool = False) -> list[Path]:
        """Remove snapshots not referenced by any commit.

        Args:
            keep_latest_n: Minimum number of most-recent snapshots to keep per agent.
            dry_run: If True, return the list without deleting anything.

        Returns:
            List of paths that were deleted (or would be deleted on dry_run).
        """
        referenced: set[str] = set()

        agents = self.store.list_agents()
        for agent in agents:
            commits = self.store.get_commit_history(agent)

            # Always keep the HEAD snapshot
            if agent.head_commit_id:
                head = self.store.get_commit(agent.head_commit_id)
                if head:
                    referenced.add(Path(head.snapshot_path).name)

            # Keep the latest N commits' snapshots as a warm cache
            keep_commits = commits[-keep_latest_n:] if len(commits) > keep_latest_n else commits
            for c in keep_commits:
                referenced.add(Path(c.snapshot_path).name)

        deleted: list[Path] = []

        if not self.snapshots_dir.exists():
            return deleted

        for f in self.snapshots_dir.glob("*.bin"):
            if f.name.startswith(".tmp_"):
                # Orphaned temp files from crashed sessions — always safe to delete
                if not dry_run:
                    f.unlink(missing_ok=True)
                deleted.append(f)
            elif f.name not in referenced:
                if not dry_run:
                    f.unlink(missing_ok=True)
                deleted.append(f)

        return deleted
