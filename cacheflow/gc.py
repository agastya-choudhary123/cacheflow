"""Snapshot garbage collector: removes unreferenced KV cache files."""

import datetime
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

    def collect(
        self,
        keep_latest_n: int = 3,
        dry_run: bool = False,
        older_than_days: int | None = None,
    ) -> list[Path]:
        """Remove snapshots not referenced by any commit.

        Args:
            keep_latest_n: Minimum number of most-recent snapshots to keep per agent.
            dry_run: If True, return the list without deleting anything.
            older_than_days: If set, also delete snapshots whose commit's created_at
                is older than this many days. HEAD is always protected regardless.

        Returns:
            List of paths that were deleted (or would be deleted on dry_run).
        """
        cutoff: datetime.datetime | None = None
        if older_than_days is not None:
            cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=older_than_days)

        referenced: set[str] = set()
        head_snapshots: set[str] = set()

        agents = self.store.list_agents()
        for agent in agents:
            commits = self.store.get_commit_history(agent)

            # Always keep the HEAD snapshot regardless of age
            if agent.head_commit_id:
                head = self.store.get_commit(agent.head_commit_id)
                if head:
                    head_snapshots.add(Path(head.snapshot_path).name)
                    referenced.add(Path(head.snapshot_path).name)

            # Keep the latest N commits' snapshots as a warm cache
            keep_commits = commits[-keep_latest_n:] if len(commits) > keep_latest_n else commits
            for c in keep_commits:
                # Skip time-based expiry check for keep_latest_n set; apply cutoff to older commits
                if cutoff is None:
                    referenced.add(Path(c.snapshot_path).name)
                else:
                    # Keep if within the time window
                    created = c.created_at
                    if isinstance(created, str):
                        try:
                            created = datetime.datetime.fromisoformat(created)
                            if created.tzinfo is None:
                                created = created.replace(tzinfo=datetime.timezone.utc)
                        except ValueError:
                            created = None
                    if created is None or created >= cutoff:
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
            elif f.name not in referenced and f.name not in head_snapshots:
                if not dry_run:
                    f.unlink(missing_ok=True)
                deleted.append(f)

        return deleted
