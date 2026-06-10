"""SQLite store for agent state and snapshots."""

import hashlib
from pathlib import Path
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Uuid, event, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session as SQLSession
from sqlalchemy.exc import IntegrityError

Base = declarative_base()


def _hash_context(text_content: str) -> str:
    """Return SHA-256 hex digest of a string (64 chars)."""
    return hashlib.sha256(text_content.encode()).hexdigest()


class Agent(Base):
    """An agent that runs tasks and accumulates KV cache state."""

    __tablename__ = "agents"

    id = Column(Uuid, primary_key=True, default=uuid4)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    model_hash = Column(String, nullable=False)  # sha256 of model file
    model_name = Column(String, nullable=False)  # e.g. "qwen2.5-coder:7b"
    ctx_size = Column(Integer, nullable=False)
    baseline_tokens_evaluated = Column(Integer, nullable=True)
    stable_context_hash = Column(String, nullable=True)  # SHA-256 of stable prefix
    current_snapshot_path = Column(String, nullable=True)  # path to .bin file
    current_snapshot_size_bytes = Column(Integer, nullable=False, default=0)
    last_tokens_saved = Column(Integer, nullable=False, default=0)
    parent_agent_id = Column(Uuid, ForeignKey("agents.id"), nullable=True)  # for forking


class CacheFlowStore:
    """Manages the SQLite database for agents and snapshots."""

    def __init__(self, db_path: Path):
        """Initialize store with a database path."""
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            isolation_level="SERIALIZABLE",
            connect_args={"timeout": 10},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    def init_db(self) -> None:
        """Create all tables."""
        Base.metadata.create_all(self.engine, checkfirst=True)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Apply schema migrations."""
        with self.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(agents)"))
            cols = {row[1] for row in result}

            migrations = [
                ("baseline_tokens_evaluated", "ALTER TABLE agents ADD COLUMN baseline_tokens_evaluated INTEGER"),
                ("current_snapshot_path", "ALTER TABLE agents ADD COLUMN current_snapshot_path TEXT"),
                ("current_snapshot_size_bytes", "ALTER TABLE agents ADD COLUMN current_snapshot_size_bytes INTEGER DEFAULT 0"),
                ("last_tokens_saved", "ALTER TABLE agents ADD COLUMN last_tokens_saved INTEGER DEFAULT 0"),
                ("parent_agent_id", "ALTER TABLE agents ADD COLUMN parent_agent_id TEXT"),
                ("stable_context_hash", "ALTER TABLE agents ADD COLUMN stable_context_hash TEXT"),
            ]

            for col_name, migration_sql in migrations:
                if col_name not in cols:
                    conn.execute(text(migration_sql))
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
        """Persist the SHA-256 hash of the stable prefix for change detection.

        Accepts the full stable_context text for API compatibility but only
        stores the hash — the full text is never written to the database.
        """
        context_hash = _hash_context(stable_context)
        session = self._get_session()
        try:
            agent.stable_context_hash = context_hash
            session.merge(agent)
            session.commit()
        finally:
            session.close()

    def get_stable_context_hash(self, agent: Agent) -> str | None:
        """Return the stored stable_context_hash for an agent."""
        return agent.stable_context_hash

    def update_agent_baseline(self, agent: Agent, baseline: int) -> None:
        """Persist baseline_tokens_evaluated on first session completion."""
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

    def update_agent_snapshot(
        self,
        agent: Agent,
        snapshot_path: str,
        snapshot_size_bytes: int,
        tokens_saved: int,
    ) -> None:
        """Update agent's current snapshot and metrics."""
        session = self._get_session()
        try:
            agent.current_snapshot_path = snapshot_path
            agent.current_snapshot_size_bytes = snapshot_size_bytes
            agent.last_tokens_saved = tokens_saved
            session.merge(agent)
            session.commit()
        finally:
            session.close()
