"""SQLite DAG store for commits and agent state."""

import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Uuid, event, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session as SQLSession
from sqlalchemy.exc import IntegrityError

Base = declarative_base()


class Agent(Base):
    """An agent that runs tasks and accumulates KV cache state."""

    __tablename__ = "agents"

    id = Column(Uuid, primary_key=True, default=uuid4)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    model_hash = Column(String, nullable=False)  # sha256 of model file
    model_name = Column(String, nullable=False)  # e.g. "qwen2.5-coder:7b"
    ctx_size = Column(Integer, nullable=False)
    baseline_tokens_evaluated = Column(Integer, nullable=True)  # total prompt tokens on first session (N + task1), used to compute savings
    stable_context = Column(String, nullable=True)  # exact stable prefix text used for current HEAD snapshot
    head_commit_id = Column(
        Uuid, ForeignKey("commits.id"), nullable=True
    )  # current HEAD


class Commit(Base):
    """A snapshot of agent state at a point in time."""

    __tablename__ = "commits"

    id = Column(Uuid, primary_key=True)
    agent_id = Column(Uuid, ForeignKey("agents.id"), nullable=False)
    parent_id = Column(
        Uuid, ForeignKey("commits.id"), nullable=True
    )  # previous commit
    forked_from_id = Column(
        Uuid, ForeignKey("commits.id"), nullable=True
    )  # set when forked
    snapshot_path = Column(String, nullable=False)  # relative path to .bin file
    snapshot_size_bytes = Column(Integer, nullable=False)
    task = Column(String, nullable=False)  # what the agent did
    tokens_this_session = Column(Integer, nullable=False)
    tokens_saved = Column(Integer, nullable=False)
    llama_cpp_version = Column(String, nullable=False)
    snapshot_save_time_ms = Column(Integer, nullable=False)
    snapshot_restore_time_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class SessionLog(Base):
    """Log of a single session with an agent."""

    __tablename__ = "sessions"

    id = Column(Uuid, primary_key=True, default=uuid4)
    agent_id = Column(Uuid, ForeignKey("agents.id"), nullable=False)
    commit_id = Column(Uuid, ForeignKey("commits.id"), nullable=True)
    prompt = Column(String, nullable=False)
    response = Column(String, nullable=False)
    tokens_in = Column(Integer, nullable=False)
    tokens_out = Column(Integer, nullable=False)
    duration_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class SnapshotEmbedding(Base):
    """Semantic embedding and knowledge facets for a snapshot."""

    __tablename__ = "snapshot_embeddings"

    commit_id = Column(Uuid, ForeignKey("commits.id"), primary_key=True)
    agent_id = Column(Uuid, ForeignKey("agents.id"), nullable=False)
    short_summary = Column(String, nullable=False)  # 2-3 sentence NL summary derived from facets
    facets = Column(String, nullable=False)  # JSON: {functions: [...], bugs: [...], patterns: [...], facts: [...]}
    embedding = Column(String, nullable=False)  # JSON: list[float] 384-dim of short_summary
    facet_embeddings = Column(String, nullable=False)  # JSON: {facet_name: list[float]}
    deep_summary = Column(String, nullable=True)  # Populated on-demand, cached
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class CacheFlowStore:
    """Manages the SQLite DAG database."""

    def __init__(self, db_path: Path):
        """Initialize store with a database path."""
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            isolation_level="SERIALIZABLE",
            connect_args={"timeout": 10},
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Enable WAL mode for better concurrency
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    def init_db(self) -> None:
        """Create all tables, running any needed schema migrations."""
        Base.metadata.create_all(self.engine, checkfirst=True)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Apply additive schema migrations safely."""
        with self.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(agents)"))
            cols = {row[1] for row in result}
            if "baseline_tokens_evaluated" not in cols:
                conn.execute(text("ALTER TABLE agents ADD COLUMN baseline_tokens_evaluated INTEGER"))
                conn.commit()
            if "stable_context" not in cols:
                conn.execute(text("ALTER TABLE agents ADD COLUMN stable_context TEXT"))
                conn.commit()

    def _get_session(self) -> SQLSession:
        """Get a new database session."""
        return self.SessionLocal()

    def create_agent(
        self, name: str, model_name: str, model_hash: str, ctx_size: int
    ) -> Agent:
        """Create a new agent. Raises ValueError if agent name already exists."""
        session = self._get_session()
        try:
            agent = Agent(
                name=name,
                model_name=model_name,
                model_hash=model_hash,
                ctx_size=ctx_size,
            )
            session.add(agent)
            session.commit()
            session.refresh(agent)
            return agent
        except IntegrityError as e:
            session.rollback()
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Agent '{name}' already exists")
            raise
        finally:
            session.close()

    def update_agent_stable_context(self, agent: Agent, stable_context: str) -> None:
        """Persist the stable prefix text used for the current HEAD snapshot."""
        session = self._get_session()
        try:
            agent.stable_context = stable_context
            session.merge(agent)
            session.commit()
        finally:
            session.close()

    def update_agent_baseline(self, agent: Agent, baseline: int) -> None:
        """Persist baseline_tokens_evaluated on first session completion.

        Args:
            agent: Agent to update
            baseline: tokens_evaluated from first session (must be > 0)
        """
        if baseline <= 0:
            raise ValueError(f"Baseline tokens must be positive, got {baseline}")

        session = self._get_session()
        try:
            agent.baseline_tokens_evaluated = baseline
            session.merge(agent)
            session.commit()
        finally:
            session.close()

    def get_agent(self, name: str) -> Agent | None:
        """Get an agent by name."""
        session = self._get_session()
        try:
            return session.query(Agent).filter(Agent.name == name).first()
        finally:
            session.close()

    def get_agent_by_id(self, agent_id: UUID) -> Agent | None:
        """Get an agent by ID."""
        session = self._get_session()
        try:
            return session.query(Agent).filter(Agent.id == agent_id).first()
        finally:
            session.close()

    def list_agents(self) -> list[Agent]:
        """List all agents."""
        session = self._get_session()
        try:
            return session.query(Agent).all()
        finally:
            session.close()

    def create_commit(
        self,
        agent: Agent,
        snapshot_path: str,
        task: str,
        tokens_this_session: int,
        tokens_saved: int,
        parent_id: UUID | None = None,
        forked_from_id: UUID | None = None,
        llama_cpp_version: str = "0.0.0",
        snapshot_save_time_ms: int = 0,
        snapshot_restore_time_ms: int = 0,
    ) -> Commit:
        """
        Create a new commit from a snapshot file.

        Args:
            All parameters as specified

        Returns:
            Commit object

        Raises:
            FileNotFoundError: if snapshot file doesn't exist
            ValueError: if snapshot file is empty
        """
        # Validate snapshot file exists and is readable
        snapshot_full_path = Path(snapshot_path)
        if not snapshot_full_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

        # Verify snapshot is not empty
        snapshot_size_bytes = snapshot_full_path.stat().st_size
        if snapshot_size_bytes == 0:
            raise ValueError(f"Snapshot file is empty: {snapshot_path}")

        # Compute commit ID as sha256(snapshot content + agent_id + timestamp)
        # This ensures different commits even if snapshot content is identical (e.g., forks)
        with open(snapshot_full_path, "rb") as f:
            file_contents = f.read()

        # Hash includes snapshot content, agent ID, and timestamp for uniqueness
        hash_input = file_contents + str(agent.id).encode() + str(int(time.time() * 1e9)).encode()
        commit_hash = hashlib.sha256(hash_input).digest()
        commit_id = UUID(bytes=commit_hash[:16])

        session = self._get_session()
        try:
            commit = Commit(
                id=commit_id,
                agent_id=agent.id,
                parent_id=parent_id,
                forked_from_id=forked_from_id,
                snapshot_path=snapshot_path,
                snapshot_size_bytes=snapshot_size_bytes,
                task=task,
                tokens_this_session=tokens_this_session,
                tokens_saved=tokens_saved,
                llama_cpp_version=llama_cpp_version,
                snapshot_save_time_ms=snapshot_save_time_ms,
                snapshot_restore_time_ms=snapshot_restore_time_ms,
            )
            session.add(commit)

            # Update agent's head commit
            agent.head_commit_id = commit_id
            session.merge(agent)

            session.commit()
            session.refresh(commit)
            return commit
        except IntegrityError as e:
            session.rollback()
            raise RuntimeError(f"Failed to create commit: {e}")
        finally:
            session.close()

    def get_commit(self, commit_id: UUID) -> Commit | None:
        """Get a commit by ID."""
        session = self._get_session()
        try:
            return session.query(Commit).filter(Commit.id == commit_id).first()
        finally:
            session.close()

    def get_commit_by_id_prefix(self, commit_id_prefix: str) -> Commit | None:
        """Get a commit by ID prefix (short hash)."""
        session = self._get_session()
        try:
            # Try parsing as UUID first
            try:
                commit_id = UUID(commit_id_prefix)
                return session.query(Commit).filter(Commit.id == commit_id).first()
            except ValueError:
                # If not a valid UUID, treat as a prefix
                all_commits = session.query(Commit).all()
                for commit in all_commits:
                    if str(commit.id).startswith(commit_id_prefix):
                        return commit
                return None
        finally:
            session.close()

    def get_commit_history(self, agent: Agent) -> list[Commit]:
        """Get commits from HEAD back to root, oldest last."""
        commits = []
        session = self._get_session()
        try:
            current_id = agent.head_commit_id
            while current_id:
                commit = session.query(Commit).filter(Commit.id == current_id).first()
                if not commit:
                    break
                commits.append(commit)
                current_id = commit.parent_id
            return list(reversed(commits))
        finally:
            session.close()

    def log_session(
        self,
        agent: Agent,
        commit: Commit,
        prompt: str,
        response: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: int,
    ) -> SessionLog:
        """Log a session."""
        session = self._get_session()
        try:
            session_log = SessionLog(
                agent_id=agent.id,
                commit_id=commit.id,
                prompt=prompt,
                response=response,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=duration_ms,
            )
            session.add(session_log)
            session.commit()
            session.refresh(session_log)
            return session_log
        finally:
            session.close()

    def save_snapshot_embedding(
        self,
        commit_id: UUID,
        agent_id: UUID,
        short_summary: str,
        facets: str,  # JSON string
        embedding: str,  # JSON string: list[float]
        facet_embeddings: str,  # JSON string: {facet_name: list[float]}
    ) -> SnapshotEmbedding:
        """Save semantic embedding and knowledge facets for a snapshot."""
        session = self._get_session()
        try:
            snapshot_emb = SnapshotEmbedding(
                commit_id=commit_id,
                agent_id=agent_id,
                short_summary=short_summary,
                facets=facets,
                embedding=embedding,
                facet_embeddings=facet_embeddings,
            )
            session.add(snapshot_emb)
            session.commit()
            session.refresh(snapshot_emb)
            return snapshot_emb
        finally:
            session.close()

    def get_snapshot_embedding(self, commit_id: UUID) -> SnapshotEmbedding | None:
        """Get semantic embedding for a specific commit."""
        session = self._get_session()
        try:
            return session.query(SnapshotEmbedding).filter(
                SnapshotEmbedding.commit_id == commit_id
            ).first()
        finally:
            session.close()

    def get_all_embeddings(self, agent_name: str | None = None) -> list[SnapshotEmbedding]:
        """Get all snapshot embeddings, optionally filtered by agent name."""
        session = self._get_session()
        try:
            query = session.query(SnapshotEmbedding)
            if agent_name:
                query = query.join(Agent).filter(Agent.name == agent_name)
            return query.all()
        finally:
            session.close()

    def update_deep_summary(self, commit_id: UUID, deep_summary: str) -> None:
        """Update the deep summary for a snapshot (generated on-demand, cached)."""
        session = self._get_session()
        try:
            snapshot_emb = session.query(SnapshotEmbedding).filter(
                SnapshotEmbedding.commit_id == commit_id
            ).first()
            if snapshot_emb:
                snapshot_emb.deep_summary = deep_summary
                session.merge(snapshot_emb)
                session.commit()
        finally:
            session.close()
