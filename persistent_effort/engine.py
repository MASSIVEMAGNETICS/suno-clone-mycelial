from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_HOME = Path(os.environ.get("MYCELIUM_HOME", str(Path.home() / ".mycelium")))
DEFAULT_DB = DEFAULT_HOME / "persistent_effort.db"
DEFAULT_ARTIFACTS = DEFAULT_HOME / "persistent_effort" / "artifacts"

# These are intentionally small inference-search arms.  The base model remains
# frozen during normal operation.  The values are conservative ACE-Step 1.5
# controls and can be expanded without migrating the database.
DEFAULT_ARMS: tuple[dict[str, Any], ...] = (
    {"model": "acestep-v15-turbo", "inference_steps": 8, "guidance_scale": 5.0, "audio_cover_strength": 0.20},
    {"model": "acestep-v15-turbo", "inference_steps": 8, "guidance_scale": 6.0, "audio_cover_strength": 0.25},
    {"model": "acestep-v15-turbo", "inference_steps": 8, "guidance_scale": 7.0, "audio_cover_strength": 0.30},
    {"model": "acestep-v15-turbo", "inference_steps": 8, "guidance_scale": 5.5, "audio_cover_strength": 0.35},
    {"model": "acestep-v15-turbo", "inference_steps": 8, "guidance_scale": 6.5, "audio_cover_strength": 0.40},
    {"model": "acestep-v15-base", "inference_steps": 24, "guidance_scale": 5.5, "audio_cover_strength": 0.25},
    {"model": "acestep-v15-base", "inference_steps": 32, "guidance_scale": 6.5, "audio_cover_strength": 0.30},
    {"model": "acestep-v15-base", "inference_steps": 40, "guidance_scale": 7.0, "audio_cover_strength": 0.35},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_ts() -> int:
    return int(time.time())


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ffprobe(path: Path) -> dict[str, Any]:
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    cmd = [
        exe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name,size:stream=codec_name,sample_rate,channels,channel_layout",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {"ffprobe_error": proc.stderr.strip()[:1000]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ffprobe_error": "invalid JSON"}


def probe_audio(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "suffix": path.suffix.lower(),
    }
    probe = _ffprobe(path)
    if probe:
        result["ffprobe"] = probe
        try:
            result["duration_s"] = float(probe.get("format", {}).get("duration"))
        except (TypeError, ValueError):
            pass
    return result


@dataclass(frozen=True)
class SongRecord:
    song_id: str
    sha256: str
    path: str
    rights_state: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    song_id: str
    arm_key: str
    seed: int
    task_type: str
    prompt: str
    lyrics: str
    params: dict[str, Any]
    artifact_path: Optional[str] = None
    artifact_sha256: Optional[str] = None
    score: Optional[float] = None


class PersistentStore:
    """Append-oriented SQLite state for cheap, durable learning.

    The generator may be replaced.  This database is the experience that must
    survive model swaps.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path).expanduser().resolve()
        ensure_parent(self.db_path)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "PersistentStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS songs (
                song_id TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                rights_state TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS arms (
                arm_key TEXT PRIMARY KEY,
                params_json TEXT NOT NULL,
                pulls INTEGER NOT NULL DEFAULT 0,
                reward_sum REAL NOT NULL DEFAULT 0.0,
                reward_sq_sum REAL NOT NULL DEFAULT 0.0,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                parent_experiment_id TEXT,
                song_id TEXT NOT NULL REFERENCES songs(song_id),
                arm_key TEXT NOT NULL REFERENCES arms(arm_key),
                seed INTEGER NOT NULL,
                task_type TEXT NOT NULL,
                prompt TEXT NOT NULL,
                lyrics TEXT NOT NULL,
                params_json TEXT NOT NULL,
                backend_request_json TEXT,
                backend_result_json TEXT,
                artifact_path TEXT,
                artifact_sha256 TEXT,
                score REAL,
                status TEXT NOT NULL,
                failure TEXT,
                created_at INTEGER NOT NULL,
                completed_at INTEGER,
                FOREIGN KEY(parent_experiment_id) REFERENCES experiments(experiment_id)
            );

            CREATE TABLE IF NOT EXISTS pairwise_feedback (
                feedback_id TEXT PRIMARY KEY,
                song_id TEXT NOT NULL REFERENCES songs(song_id),
                winner_experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                loser_experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS adaptation_events (
                adaptation_id TEXT PRIMARY KEY,
                song_id TEXT,
                method TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                recommendation_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY(song_id) REFERENCES songs(song_id)
            );
            """
        )
        self.conn.commit()

    def ensure_arm(self, params: dict[str, Any]) -> str:
        arm_key = sha256_bytes(canonical_json(params).encode("utf-8"))[:20]
        self.conn.execute(
            """
            INSERT INTO arms(arm_key, params_json, pulls, reward_sum, reward_sq_sum, updated_at)
            VALUES(?, ?, 0, 0, 0, ?)
            ON CONFLICT(arm_key) DO NOTHING
            """,
            (arm_key, canonical_json(params), utc_ts()),
        )
        self.conn.commit()
        return arm_key

    def ensure_default_arms(self) -> None:
        for arm in DEFAULT_ARMS:
            self.ensure_arm(dict(arm))

    def ingest_song(self, path: Path | str, rights_state: str, metadata: Optional[dict[str, Any]] = None) -> SongRecord:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        allowed = {"owned", "licensed", "public-domain"}
        if rights_state not in allowed:
            raise ValueError(f"rights_state must be one of {sorted(allowed)}")
        digest = sha256_file(path)
        merged_meta = probe_audio(path)
        if metadata:
            merged_meta.update(metadata)
        row = self.conn.execute("SELECT * FROM songs WHERE sha256=?", (digest,)).fetchone()
        if row:
            if row["rights_state"] != rights_state:
                raise ValueError("re-ingested content must preserve its recorded rights_state")
            self.conn.execute(
                "UPDATE songs SET path=?, metadata_json=? WHERE sha256=?",
                (str(path), canonical_json(merged_meta), digest),
            )
            self.conn.commit()
            return SongRecord(
                song_id=row["song_id"],
                sha256=row["sha256"],
                path=str(path),
                rights_state=row["rights_state"],
                metadata=merged_meta,
            )
        song_id = f"song_{digest[:16]}"
        self.conn.execute(
            "INSERT INTO songs(song_id,sha256,path,rights_state,metadata_json,created_at) VALUES(?,?,?,?,?,?)",
            (song_id, digest, str(path), rights_state, canonical_json(merged_meta), utc_ts()),
        )
        self.conn.commit()
        return SongRecord(song_id, digest, str(path), rights_state, merged_meta)

    def get_song(self, song_id: str) -> SongRecord:
        row = self.conn.execute("SELECT * FROM songs WHERE song_id=?", (song_id,)).fetchone()
        if not row:
            raise KeyError(song_id)
        return SongRecord(row["song_id"], row["sha256"], row["path"], row["rights_state"], json.loads(row["metadata_json"]))

    def choose_arm_ucb(self, exploration: float = 1.1, allowed_models: Optional[set[str]] = None) -> tuple[str, dict[str, Any]]:
        self.ensure_default_arms()
        rows = self.conn.execute(
            """
            SELECT a.*,
                   COUNT(CASE
                       WHEN e.score IS NULL AND e.status IN ('queued','running','complete') THEN 1
                   END) AS reservations
            FROM arms AS a
            LEFT JOIN experiments AS e ON e.arm_key=a.arm_key
            GROUP BY a.arm_key
            ORDER BY a.arm_key
            """
        ).fetchall()
        candidates: list[sqlite3.Row] = []
        for row in rows:
            params = json.loads(row["params_json"])
            if allowed_models and params.get("model") not in allowed_models:
                continue
            candidates.append(row)
        if not candidates:
            raise RuntimeError("No generation arms available")

        untried = [r for r in candidates if int(r["pulls"]) + int(r["reservations"]) == 0]
        if untried:
            row = random.choice(untried)
            return row["arm_key"], json.loads(row["params_json"])

        total = sum(int(r["pulls"]) + int(r["reservations"]) for r in candidates)
        best_row = None
        best_value = float("-inf")
        for row in candidates:
            pulls = int(row["pulls"])
            effective_pulls = pulls + int(row["reservations"])
            mean = float(row["reward_sum"]) / pulls if pulls else 0.0
            bonus = exploration * math.sqrt(math.log(total + 1.0) / effective_pulls)
            value = mean + bonus
            if value > best_value:
                best_value = value
                best_row = row
        assert best_row is not None
        return best_row["arm_key"], json.loads(best_row["params_json"])

    def create_experiment(
        self,
        *,
        song_id: str,
        arm_key: str,
        seed: int,
        task_type: str,
        prompt: str,
        lyrics: str,
        params: dict[str, Any],
        parent_experiment_id: Optional[str] = None,
    ) -> str:
        experiment_id = f"exp_{uuid.uuid4().hex[:20]}"
        self.conn.execute(
            """
            INSERT INTO experiments(
                experiment_id,parent_experiment_id,song_id,arm_key,seed,task_type,prompt,lyrics,
                params_json,status,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                experiment_id,
                parent_experiment_id,
                song_id,
                arm_key,
                int(seed),
                task_type,
                prompt,
                lyrics,
                canonical_json(params),
                "queued",
                utc_ts(),
            ),
        )
        self.conn.commit()
        return experiment_id

    def mark_running(self, experiment_id: str, request: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE experiments SET status='running', backend_request_json=? WHERE experiment_id=?",
            (canonical_json(request), experiment_id),
        )
        self.conn.commit()

    def mark_complete(self, experiment_id: str, result: dict[str, Any], artifact_path: Path) -> None:
        digest = sha256_file(artifact_path)
        self.conn.execute(
            """
            UPDATE experiments
            SET status='complete', backend_result_json=?, artifact_path=?, artifact_sha256=?, completed_at=?
            WHERE experiment_id=?
            """,
            (canonical_json(result), str(artifact_path.resolve()), digest, utc_ts(), experiment_id),
        )
        self.conn.commit()

    def mark_failed(self, experiment_id: str, failure: str) -> None:
        self.conn.execute(
            "UPDATE experiments SET status='failed', failure=?, completed_at=? WHERE experiment_id=?",
            (failure[:4000], utc_ts(), experiment_id),
        )
        self.conn.commit()

    def get_experiment(self, experiment_id: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        if not row:
            raise KeyError(experiment_id)
        return row

    def apply_score(self, experiment_id: str, score: float) -> None:
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be in [0,1]")
        row = self.get_experiment(experiment_id)
        if row["status"] != "complete" or not row["artifact_path"] or not row["artifact_sha256"]:
            raise ValueError("feedback requires a complete experiment with a committed artifact")
        old_score = row["score"]
        if old_score is not None:
            raise ValueError("experiment already scored; immutable feedback prevents accidental double-counting")
        with self.conn:
            claimed = self.conn.execute(
                """
                UPDATE experiments
                SET score=?
                WHERE experiment_id=?
                  AND score IS NULL
                  AND status='complete'
                  AND artifact_path IS NOT NULL AND artifact_path<>''
                  AND artifact_sha256 IS NOT NULL AND artifact_sha256<>''
                """,
                (score, experiment_id),
            )
            if claimed.rowcount != 1:
                raise ValueError("experiment was concurrently scored or is no longer eligible for feedback")
            self.conn.execute(
                """
                UPDATE arms
                SET pulls=pulls+1,
                    reward_sum=reward_sum+?,
                    reward_sq_sum=reward_sq_sum+?,
                    updated_at=?
                WHERE arm_key=?
                """,
                (score, score * score, utc_ts(), row["arm_key"]),
            )

    def pairwise(self, winner_id: str, loser_id: str, source: str = "human", confidence: float = 1.0) -> str:
        if winner_id == loser_id:
            raise ValueError("winner and loser must be different experiments")
        winner = self.get_experiment(winner_id)
        loser = self.get_experiment(loser_id)
        if winner["song_id"] != loser["song_id"]:
            raise ValueError("pairwise feedback must compare experiments from the same reference song")
        for row in (winner, loser):
            if row["status"] != "complete" or not row["artifact_path"] or not row["artifact_sha256"]:
                raise ValueError("pairwise feedback requires two complete experiments with committed artifacts")
            if row["score"] is not None:
                raise ValueError("pairwise feedback requires previously unscored experiments")
        confidence = float(confidence)
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in (0,1]")
        winner_score = 0.5 + (0.5 * confidence)
        loser_score = 0.5 - (0.5 * confidence)
        feedback_id = f"fb_{uuid.uuid4().hex[:20]}"
        with self.conn:
            for row, score in ((winner, winner_score), (loser, loser_score)):
                claimed = self.conn.execute(
                    """
                    UPDATE experiments
                    SET score=?
                    WHERE experiment_id=?
                      AND score IS NULL
                      AND status='complete'
                      AND artifact_path IS NOT NULL AND artifact_path<>''
                      AND artifact_sha256 IS NOT NULL AND artifact_sha256<>''
                    """,
                    (score, row["experiment_id"]),
                )
                if claimed.rowcount != 1:
                    raise ValueError("pairwise experiment was concurrently scored or is no longer eligible")
            self.conn.execute(
                "INSERT INTO pairwise_feedback VALUES(?,?,?,?,?,?,?)",
                (feedback_id, winner["song_id"], winner_id, loser_id, source, confidence, utc_ts()),
            )
            for row, score in ((winner, winner_score), (loser, loser_score)):
                self.conn.execute(
                    """
                    UPDATE arms
                    SET pulls=pulls+1,
                        reward_sum=reward_sum+?,
                        reward_sq_sum=reward_sq_sum+?,
                        updated_at=?
                    WHERE arm_key=?
                    """,
                    (score, score * score, utc_ts(), row["arm_key"]),
                )
        return feedback_id

    def top_experiments(self, song_id: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM experiments
            WHERE song_id=? AND status='complete'
            ORDER BY CASE WHEN score IS NULL THEN 1 ELSE 0 END, score DESC, completed_at DESC
            LIMIT ?
            """,
            (song_id, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

    def adaptation_recommendation(self, song_id: str, min_scored: int = 24, window: int = 12) -> dict[str, Any]:
        rows = self.conn.execute(
            """
            SELECT score, completed_at FROM experiments
            WHERE song_id=? AND score IS NOT NULL AND status='complete'
            ORDER BY completed_at ASC
            """,
            (song_id,),
        ).fetchall()
        scores = [float(r["score"]) for r in rows]
        if len(scores) < min_scored:
            return {
                "action": "keep_searching",
                "reason": f"Only {len(scores)} scored experiments; persistent inference search has not been exhausted.",
                "scored_experiments": len(scores),
            }
        recent = scores[-window:]
        prior = scores[-2 * window : -window] if len(scores) >= 2 * window else scores[:window]
        recent_best = max(recent)
        prior_best = max(prior) if prior else 0.0
        gain = recent_best - prior_best
        if recent_best >= 0.90:
            return {
                "action": "freeze_weights",
                "reason": "Search already reaches high quality; weight updates would add cost and forgetting risk.",
                "recent_best": recent_best,
                "gain": gain,
            }
        if gain >= 0.03:
            return {
                "action": "keep_searching",
                "reason": "Recent non-parametric search is still improving.",
                "recent_best": recent_best,
                "gain": gain,
            }
        recommendation = {
            "action": "targeted_adapter_candidate",
            "reason": "Search appears plateaued; escalate only to sensitivity-targeted low-rank adaptation.",
            "recent_best": recent_best,
            "gain": gain,
            "procedure": [
                "Preprocess only owned/licensed winners and representative failures once.",
                "Run ACE-Step Side-Step gradient sensitivity estimation on 3-5 batches.",
                "Target only the highest-sensitivity projections/modules.",
                "Use rank-16 LoRA first for production compatibility; test LoKR separately for speed.",
                "Accept the adapter only if blind A/B evaluation beats the frozen-base persistent loop.",
            ],
        }
        adaptation_id = f"adapt_{uuid.uuid4().hex[:20]}"
        self.conn.execute(
            "INSERT INTO adaptation_events VALUES(?,?,?,?,?)",
            (adaptation_id, song_id, "sensitivity_targeted_adapter", canonical_json({"scores": scores[-24:]}), canonical_json(recommendation), utc_ts()),
        )
        self.conn.commit()
        recommendation["adaptation_id"] = adaptation_id
        return recommendation


class AceStepClient:
    """Minimal stdlib client for the documented ACE-Step 1.5 async REST API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001", api_key: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)

    def _request_json(self, method: str, path: str, body: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"ACE-Step HTTP {exc.code}: {detail[:2000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ACE-Step unavailable at {self.base_url}: {exc}") from exc
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("ACE-Step returned invalid JSON") from exc
        if isinstance(parsed, dict) and parsed.get("error"):
            raise RuntimeError(f"ACE-Step error: {parsed['error']}")
        return parsed

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/health")

    def submit(self, payload: dict[str, Any]) -> str:
        response = self._request_json("POST", "/release_task", payload)
        data = response.get("data", response)
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not task_id:
            raise RuntimeError(f"No task_id in ACE-Step response: {response}")
        return str(task_id)

    def wait(self, task_id: str, timeout_s: float = 900.0, poll_s: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            response = self._request_json("POST", "/query_result", {"task_id_list": [task_id]})
            data = response.get("data", response)
            if not isinstance(data, list) or not data:
                time.sleep(poll_s)
                continue
            item = data[0]
            status = int(item.get("status", 0))
            if status == 1:
                result = item.get("result", "[]")
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        result = [{"raw_result": result}]
                return {"task_id": task_id, "status": 1, "result": result, "response": response}
            if status == 2:
                raise RuntimeError(f"ACE-Step task failed: {item}")
            time.sleep(poll_s)
        raise TimeoutError(f"ACE-Step task {task_id} exceeded {timeout_s}s")

    def download(self, file_ref: str, destination: Path) -> None:
        ensure_parent(destination)
        url = file_ref if file_ref.startswith("http://") or file_ref.startswith("https://") else urllib.parse.urljoin(self.base_url + "/", file_ref.lstrip("/"))
        req = urllib.request.Request(url, method="GET")
        if self.api_key and self._origin(url) == self._origin(self.base_url):
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=max(self.timeout, 120.0)) as resp, destination.open("wb") as f:
            shutil.copyfileobj(resp, f)

    @staticmethod
    def _origin(url: str) -> tuple[str, str, int | None]:
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        port = parsed.port
        if port is None and scheme == "https":
            port = 443
        elif port is None and scheme == "http":
            port = 80
        return scheme, host, port


class PersistentEffortEngine:
    def __init__(self, store: PersistentStore, client: AceStepClient, artifact_root: Path | str = DEFAULT_ARTIFACTS):
        self.store = store
        self.client = client
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _seed() -> int:
        return random.SystemRandom().randint(1, 2_147_483_646)

    def _execute(self, experiment_id: str, payload: dict[str, Any]) -> ExperimentRecord:
        row = self.store.get_experiment(experiment_id)
        self.store.mark_running(experiment_id, payload)
        try:
            task_id = self.client.submit(payload)
            result = self.client.wait(task_id)
            items = result.get("result", [])
            if not items or not isinstance(items, list):
                raise RuntimeError(f"ACE-Step completed without an audio result: {result}")
            file_ref = items[0].get("file")
            if not file_ref:
                raise RuntimeError(f"ACE-Step result has no file field: {items[0]}")
            out_dir = self.artifact_root / row["song_id"] / experiment_id
            out_path = out_dir / "mix.wav"
            self.client.download(str(file_ref), out_path)
            self.store.mark_complete(experiment_id, result, out_path)
            complete = self.store.get_experiment(experiment_id)
            return ExperimentRecord(
                experiment_id=experiment_id,
                song_id=complete["song_id"],
                arm_key=complete["arm_key"],
                seed=int(complete["seed"]),
                task_type=complete["task_type"],
                prompt=complete["prompt"],
                lyrics=complete["lyrics"],
                params=json.loads(complete["params_json"]),
                artifact_path=complete["artifact_path"],
                artifact_sha256=complete["artifact_sha256"],
                score=complete["score"],
            )
        except Exception as exc:
            self.store.mark_failed(experiment_id, repr(exc))
            raise

    def generate_preview(
        self,
        *,
        song_id: str,
        prompt: str,
        lyrics: str = "",
        duration_s: float = 30.0,
        seed: Optional[int] = None,
        exploration: float = 1.1,
    ) -> ExperimentRecord:
        song = self.store.get_song(song_id)
        arm_key, arm = self.store.choose_arm_ucb(exploration=exploration)
        seed = int(seed if seed is not None else self._seed())
        params = dict(arm)
        params["audio_duration"] = float(duration_s)
        experiment_id = self.store.create_experiment(
            song_id=song_id,
            arm_key=arm_key,
            seed=seed,
            task_type="text2music",
            prompt=prompt,
            lyrics=lyrics,
            params=params,
        )
        payload = {
            "prompt": prompt,
            "lyrics": lyrics,
            "task_type": "text2music",
            "reference_audio_path": song.path,
            "model": arm["model"],
            "inference_steps": int(arm["inference_steps"]),
            "guidance_scale": float(arm["guidance_scale"]),
            "audio_cover_strength": float(arm["audio_cover_strength"]),
            "audio_duration": float(duration_s),
            "use_random_seed": False,
            "seed": seed,
            "thinking": False,
        }
        return self._execute(experiment_id, payload)

    def preview_batch(self, *, song_id: str, prompt: str, lyrics: str = "", count: int = 8, duration_s: float = 30.0) -> list[ExperimentRecord]:
        if count < 1 or count > 32:
            raise ValueError("count must be between 1 and 32")
        results: list[ExperimentRecord] = []
        for _ in range(count):
            results.append(self.generate_preview(song_id=song_id, prompt=prompt, lyrics=lyrics, duration_s=duration_s))
        return results

    def promote(self, experiment_id: str, duration_s: float) -> ExperimentRecord:
        parent = self.store.get_experiment(experiment_id)
        if parent["status"] != "complete":
            raise ValueError("only a complete preview can be promoted")
        song = self.store.get_song(parent["song_id"])
        params = json.loads(parent["params_json"])
        params["audio_duration"] = float(duration_s)
        promoted_id = self.store.create_experiment(
            song_id=parent["song_id"],
            arm_key=parent["arm_key"],
            seed=int(parent["seed"]),
            task_type="text2music",
            prompt=parent["prompt"],
            lyrics=parent["lyrics"],
            params=params,
            parent_experiment_id=experiment_id,
        )
        payload = {
            "prompt": parent["prompt"],
            "lyrics": parent["lyrics"],
            "task_type": "text2music",
            "reference_audio_path": song.path,
            "model": params["model"],
            "inference_steps": int(params["inference_steps"]),
            "guidance_scale": float(params["guidance_scale"]),
            "audio_cover_strength": float(params["audio_cover_strength"]),
            "audio_duration": float(duration_s),
            "use_random_seed": False,
            "seed": int(parent["seed"]),
            "thinking": False,
        }
        return self._execute(promoted_id, payload)

    def repaint(self, experiment_id: str, start_s: float, end_s: float, instruction: str) -> ExperimentRecord:
        parent = self.store.get_experiment(experiment_id)
        if parent["status"] != "complete" or not parent["artifact_path"]:
            raise ValueError("repaint requires a completed source artifact")
        if start_s < 0 or end_s <= start_s:
            raise ValueError("invalid repaint interval")
        params = json.loads(parent["params_json"])
        child_id = self.store.create_experiment(
            song_id=parent["song_id"],
            arm_key=parent["arm_key"],
            seed=int(parent["seed"]),
            task_type="repaint",
            prompt=parent["prompt"],
            lyrics=parent["lyrics"],
            params={**params, "repainting_start": start_s, "repainting_end": end_s, "instruction": instruction},
            parent_experiment_id=experiment_id,
        )
        payload = {
            "prompt": parent["prompt"],
            "lyrics": parent["lyrics"],
            "task_type": "repaint",
            "src_audio_path": str(Path(parent["artifact_path"]).resolve()),
            "model": params.get("model", "acestep-v15-turbo"),
            "inference_steps": int(params.get("inference_steps", 8)),
            "guidance_scale": float(params.get("guidance_scale", 7.0)),
            "repainting_start": float(start_s),
            "repainting_end": float(end_s),
            "chunk_mask_mode": "explicit",
            "instruction": instruction,
            "use_random_seed": False,
            "seed": int(parent["seed"]),
            "thinking": False,
        }
        return self._execute(child_id, payload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="persistent-effort",
        description="Poor-man compute strategy for MYCELIUM: remember, search, repaint, then adapt only if plateaued.",
    )
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    p.add_argument("--ace-url", default=os.environ.get("ACE_STEP_URL", "http://127.0.0.1:8001"))
    p.add_argument("--ace-key", default=os.environ.get("ACE_STEP_API_KEY"))
    sub = p.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Seal one owned/licensed/public-domain song into persistent memory")
    ingest.add_argument("song")
    ingest.add_argument("--rights", required=True, choices=["owned", "licensed", "public-domain"])

    health = sub.add_parser("health", help="Probe local ACE-Step")

    preview = sub.add_parser("preview", help="Generate one cheap reference-conditioned preview")
    preview.add_argument("song_id")
    preview.add_argument("--prompt", required=True)
    preview.add_argument("--lyrics", default="")
    preview.add_argument("--duration", type=float, default=30.0)

    batch = sub.add_parser("batch", help="Generate a persistent-search preview batch")
    batch.add_argument("song_id")
    batch.add_argument("--prompt", required=True)
    batch.add_argument("--lyrics", default="")
    batch.add_argument("--count", type=int, default=8)
    batch.add_argument("--duration", type=float, default=30.0)

    score = sub.add_parser("score", help="Commit immutable human/AI-Ear reward in [0,1]")
    score.add_argument("experiment_id")
    score.add_argument("value", type=float)

    pair = sub.add_parser("pair", help="Record winner/loser preference")
    pair.add_argument("winner")
    pair.add_argument("loser")
    pair.add_argument("--source", default="human")
    pair.add_argument("--confidence", type=float, default=1.0)

    top = sub.add_parser("top", help="Show best remembered experiments for one reference song")
    top.add_argument("song_id")
    top.add_argument("--limit", type=int, default=10)

    promote = sub.add_parser("promote", help="Spend full-render compute only on a winning preview")
    promote.add_argument("experiment_id")
    promote.add_argument("--duration", type=float, required=True)

    repaint = sub.add_parser("repaint", help="Repair only the failed time range")
    repaint.add_argument("experiment_id")
    repaint.add_argument("--start", type=float, required=True)
    repaint.add_argument("--end", type=float, required=True)
    repaint.add_argument("--instruction", required=True)

    adapt = sub.add_parser("adaptation", help="Decide whether persistence is exhausted enough to justify sparse training")
    adapt.add_argument("song_id")

    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    with PersistentStore(args.db) as store:
        if args.command == "ingest":
            print(json.dumps(asdict(store.ingest_song(args.song, args.rights)), indent=2, ensure_ascii=False))
            return 0
        client = AceStepClient(args.ace_url, args.ace_key)
        engine = PersistentEffortEngine(store, client, args.artifacts)
        if args.command == "health":
            print(json.dumps(client.health(), indent=2, ensure_ascii=False))
        elif args.command == "preview":
            print(json.dumps(asdict(engine.generate_preview(song_id=args.song_id, prompt=args.prompt, lyrics=args.lyrics, duration_s=args.duration)), indent=2, ensure_ascii=False))
        elif args.command == "batch":
            print(json.dumps([asdict(x) for x in engine.preview_batch(song_id=args.song_id, prompt=args.prompt, lyrics=args.lyrics, count=args.count, duration_s=args.duration)], indent=2, ensure_ascii=False))
        elif args.command == "score":
            store.apply_score(args.experiment_id, args.value)
            print(json.dumps({"ok": True, "experiment_id": args.experiment_id, "score": args.value}))
        elif args.command == "pair":
            feedback_id = store.pairwise(args.winner, args.loser, source=args.source, confidence=args.confidence)
            print(json.dumps({"ok": True, "feedback_id": feedback_id}))
        elif args.command == "top":
            print(json.dumps(store.top_experiments(args.song_id, args.limit), indent=2, ensure_ascii=False))
        elif args.command == "promote":
            print(json.dumps(asdict(engine.promote(args.experiment_id, args.duration)), indent=2, ensure_ascii=False))
        elif args.command == "repaint":
            print(json.dumps(asdict(engine.repaint(args.experiment_id, args.start, args.end, args.instruction)), indent=2, ensure_ascii=False))
        elif args.command == "adaptation":
            print(json.dumps(store.adaptation_recommendation(args.song_id), indent=2, ensure_ascii=False))
        else:
            raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
