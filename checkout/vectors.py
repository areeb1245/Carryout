"""
Face embeddings in Postgres with pgvector: the index the gate searches to see whether it knows a face.

Each face the app keeps is a 128-number embedding from SFace (identity.py makes them on this machine). They are
stored in one table, `checkout_faces`, with an HNSW index on cosine distance, so "whose face is this?" is one query
however many people have registered:

    SELECT profile, 1 - (embedding <=> $face) AS score ... ORDER BY embedding <=> $face LIMIT n

Only the embeddings and the profile ids go to the database: no pictures, names, emails or payment details. SQLite
keeps a copy of the embeddings too, so the shop floor can match the few people who are inside without a network
call, and so everything still works when `DATABASE_URL` is not set.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .sam_client import load_env

log = logging.getLogger("checkout")
DIMS = 128
TABLE = "checkout_faces"


def literal(signature: np.ndarray) -> str:
    """An embedding as pgvector reads it: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.6f}" for v in np.asarray(signature, np.float32).ravel()) + "]"


class FaceIndex:
    def __init__(self, env: dict | None = None, url: str | None = None):
        env = load_env() if env is None else env
        self.url = url if url is not None else env.get("DATABASE_URL", "")
        self.enabled, self.problem = bool(self.url), "" if self.url else "DATABASE_URL is not set in .env; faces are matched from the local copy"
        self.conn = None
        self.lock = threading.Lock()
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-index")
        if self.enabled:
            try:
                self._run(self._setup)
            except Exception as e:  # no database is not a reason to keep the shop shut
                self.enabled, self.problem = False, f"the face database is not reachable ({type(e).__name__}); faces are matched from the local copy"
                log.warning("face index off: %s", e)

    # ------------------------------------------------------------------ plumbing

    def _connect(self):
        import psycopg

        # the pooled Neon address runs PgBouncer in transaction mode, which cannot keep prepared statements
        return psycopg.connect(self.url, autocommit=True, prepare_threshold=None, connect_timeout=15)

    def _run(self, work):
        """Run `work(connection)`, reconnecting once: Neon suspends an idle database and drops the connection."""
        import psycopg

        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.conn is None or self.conn.closed:
                        self.conn = self._connect()
                    return work(self.conn)
                except (psycopg.OperationalError, psycopg.InterfaceError):
                    self.conn = None
                    if attempt:
                        raise

    def _setup(self, conn) -> None:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} (id BIGSERIAL PRIMARY KEY, profile TEXT NOT NULL, local_id BIGINT NOT NULL, "
            f"embedding vector({DIMS}) NOT NULL, created TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE (profile, local_id))"
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding ON {TABLE} USING hnsw (embedding vector_cosine_ops)")

    def close(self) -> None:
        self.writer.shutdown(wait=True)
        with self.lock:
            if self.conn is not None and not self.conn.closed:
                self.conn.close()

    # ------------------------------------------------------------------ writing

    def add(self, profile: str, local_id: int, signature: np.ndarray) -> None:
        self._run(lambda c: c.execute(f"INSERT INTO {TABLE} (profile, local_id, embedding) VALUES (%s, %s, %s::vector) ON CONFLICT (profile, local_id) DO NOTHING", (profile, local_id, literal(signature))))

    def add_later(self, profile: str, local_id: int, signature: np.ndarray) -> None:
        """From the frame loop: store it without waiting for the database."""
        if not self.enabled:
            return

        def work():
            try:
                self.add(profile, local_id, signature)
            except Exception as e:
                log.warning("face embedding not stored: %s", e)

        self.writer.submit(work)

    def sync(self, rows: list[tuple[int, str, np.ndarray]]) -> dict:
        """Make the table hold exactly the local embeddings, given as (local id, profile, embedding)."""

        def work(conn) -> dict:
            there = {(p, i) for p, i in conn.execute(f"SELECT profile, local_id FROM {TABLE}")}
            here = {(profile, local_id): signature for local_id, profile, signature in rows}
            for (profile, local_id) in there - set(here):
                conn.execute(f"DELETE FROM {TABLE} WHERE profile = %s AND local_id = %s", (profile, local_id))
            for (profile, local_id) in set(here) - there:
                conn.execute(f"INSERT INTO {TABLE} (profile, local_id, embedding) VALUES (%s, %s, %s::vector) ON CONFLICT DO NOTHING", (profile, local_id, literal(here[(profile, local_id)])))
            return {"added": len(set(here) - there), "removed": len(there - set(here)), "total": len(here)}

        return self._run(work)

    def forget_all(self) -> None:
        self._run(lambda c: c.execute(f"TRUNCATE {TABLE}"))

    # ------------------------------------------------------------------ searching

    def search(self, signature: np.ndarray, nearest: int = 24) -> list[tuple[str, float]]:
        """The profiles whose stored faces are nearest this one, as (profile id, cosine similarity), best first.
        A profile keeps several views of a face; its score is that of its best view."""
        face = literal(signature)
        sql = (
            f"SELECT profile, MAX(1 - distance) AS score FROM (SELECT profile, embedding <=> %s::vector AS distance FROM {TABLE} "
            f"ORDER BY embedding <=> %s::vector LIMIT %s) nearest GROUP BY profile ORDER BY score DESC"
        )
        return [(profile, float(score)) for profile, score in self._run(lambda c: c.execute(sql, (face, face, nearest)).fetchall())]

    def count(self) -> int:
        return self._run(lambda c: c.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])
