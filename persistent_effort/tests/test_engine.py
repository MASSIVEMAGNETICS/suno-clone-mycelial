import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from persistent_effort.engine import AceStepClient, PersistentStore


class PersistentEffortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "state.db"
        self.song_file = self.root / "song.wav"
        self.song_file.write_bytes(b"RIFF" + b"\x00" * 128)
        self.store = PersistentStore(self.db)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_ingest_is_content_addressed(self):
        first = self.store.ingest_song(self.song_file, "owned")
        second = self.store.ingest_song(self.song_file, "owned")
        self.assertEqual(first.song_id, second.song_id)
        self.assertEqual(first.sha256, second.sha256)

    def test_reingest_refreshes_moved_source_path(self):
        first = self.store.ingest_song(self.song_file, "owned")
        moved = self.root / "moved.wav"
        self.song_file.rename(moved)

        refreshed = self.store.ingest_song(moved, "owned")

        self.assertEqual(refreshed.song_id, first.song_id)
        self.assertEqual(refreshed.path, str(moved.resolve()))
        self.assertEqual(self.store.get_song(first.song_id).path, str(moved.resolve()))

        with self.assertRaises(ValueError):
            self.store.ingest_song(moved, "licensed")
        self.assertEqual(self.store.get_song(first.song_id).rights_state, "owned")

    def test_invalid_rights_state_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.ingest_song(self.song_file, "unknown")

    def test_ucb_explores_untried_arms(self):
        self.store.ensure_default_arms()
        seen = set()
        for _ in range(8):
            key, params = self.store.choose_arm_ucb()
            self.assertIn("model", params)
            seen.add(key)
            song = self.store.ingest_song(self.song_file, "owned")
            exp = self.store.create_experiment(
                song_id=song.song_id,
                arm_key=key,
                seed=123,
                task_type="text2music",
                prompt="test",
                lyrics="",
                params=params,
            )
            # mark a synthetic completed artifact so scoring uses production path
            artifact = self.root / f"{exp}.wav"
            artifact.write_bytes(b"audio")
            self.store.mark_complete(exp, {"result": []}, artifact)
            self.store.apply_score(exp, 0.5)
        self.assertEqual(len(seen), 8)

    def test_ucb_reserves_unscored_arms_across_a_batch(self):
        song = self.store.ingest_song(self.song_file, "owned")
        seen = set()
        for seed in range(8):
            key, params = self.store.choose_arm_ucb()
            seen.add(key)
            self.store.create_experiment(
                song_id=song.song_id,
                arm_key=key,
                seed=seed,
                task_type="text2music",
                prompt="batch",
                lyrics="",
                params=params,
            )
        self.assertEqual(len(seen), 8)

    def test_score_is_immutable(self):
        song = self.store.ingest_song(self.song_file, "owned")
        key, params = self.store.choose_arm_ucb()
        exp = self.store.create_experiment(
            song_id=song.song_id,
            arm_key=key,
            seed=1,
            task_type="text2music",
            prompt="x",
            lyrics="",
            params=params,
        )
        artifact = self.root / "x.wav"
        artifact.write_bytes(b"audio")
        self.store.mark_complete(exp, {"result": []}, artifact)
        self.store.apply_score(exp, 0.7)
        with self.assertRaises(ValueError):
            self.store.apply_score(exp, 0.9)

    def test_score_rejects_unfinished_or_artifactless_experiment(self):
        song = self.store.ingest_song(self.song_file, "owned")
        key, params = self.store.choose_arm_ucb()
        exp = self.store.create_experiment(
            song_id=song.song_id,
            arm_key=key,
            seed=1,
            task_type="text2music",
            prompt="x",
            lyrics="",
            params=params,
        )
        with self.assertRaises(ValueError):
            self.store.apply_score(exp, 0.5)

        self.store.conn.execute("UPDATE experiments SET status='complete' WHERE experiment_id=?", (exp,))
        self.store.conn.commit()
        with self.assertRaises(ValueError):
            self.store.apply_score(exp, 0.5)

        row = self.store.get_experiment(exp)
        arm = self.store.conn.execute("SELECT pulls FROM arms WHERE arm_key=?", (key,)).fetchone()
        self.assertIsNone(row["score"])
        self.assertEqual(arm["pulls"], 0)

    def test_pairwise_requires_same_reference_song(self):
        other_file = self.root / "other.wav"
        other_file.write_bytes(b"RIFF" + b"\x01" * 128)
        a = self.store.ingest_song(self.song_file, "owned")
        b = self.store.ingest_song(other_file, "owned")
        key, params = self.store.choose_arm_ucb()
        exp_a = self.store.create_experiment(song_id=a.song_id, arm_key=key, seed=1, task_type="text2music", prompt="a", lyrics="", params=params)
        exp_b = self.store.create_experiment(song_id=b.song_id, arm_key=key, seed=2, task_type="text2music", prompt="b", lyrics="", params=params)
        with self.assertRaises(ValueError):
            self.store.pairwise(exp_a, exp_b)

    def test_pairwise_feedback_updates_scores_and_arm_learning(self):
        song = self.store.ingest_song(self.song_file, "owned")
        key_a, params_a = self.store.choose_arm_ucb()
        exp_a = self.store.create_experiment(song_id=song.song_id, arm_key=key_a, seed=1, task_type="text2music", prompt="a", lyrics="", params=params_a)
        key_b, params_b = self.store.choose_arm_ucb()
        exp_b = self.store.create_experiment(song_id=song.song_id, arm_key=key_b, seed=2, task_type="text2music", prompt="b", lyrics="", params=params_b)
        self.assertNotEqual(key_a, key_b)
        artifact_a = self.root / "a.wav"
        artifact_b = self.root / "b.wav"
        artifact_a.write_bytes(b"a")
        artifact_b.write_bytes(b"b")
        self.store.mark_complete(exp_a, {"result": []}, artifact_a)
        self.store.mark_complete(exp_b, {"result": []}, artifact_b)

        self.store.pairwise(exp_a, exp_b, confidence=1.0)

        self.assertEqual(self.store.get_experiment(exp_a)["score"], 1.0)
        self.assertEqual(self.store.get_experiment(exp_b)["score"], 0.0)
        arm_a = self.store.conn.execute("SELECT pulls,reward_sum FROM arms WHERE arm_key=?", (key_a,)).fetchone()
        arm_b = self.store.conn.execute("SELECT pulls,reward_sum FROM arms WHERE arm_key=?", (key_b,)).fetchone()
        self.assertEqual((arm_a["pulls"], arm_a["reward_sum"]), (1, 1.0))
        self.assertEqual((arm_b["pulls"], arm_b["reward_sum"]), (1, 0.0))

    def test_download_never_forwards_bearer_token_cross_origin(self):
        seen = []

        class Response:
            def __enter__(self):
                return io.BytesIO(b"audio")

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_urlopen(request, timeout):
            seen.append(request)
            return Response()

        client = AceStepClient("https://ace.example:8443/api", api_key="secret")
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.download("/same.wav", self.root / "same.wav")
            client.download("https://cdn.example/file.wav", self.root / "cross.wav")

        self.assertEqual(seen[0].get_header("Authorization"), "Bearer secret")
        self.assertIsNone(seen[1].get_header("Authorization"))

    def test_adaptation_gate_stays_closed_early(self):
        song = self.store.ingest_song(self.song_file, "owned")
        rec = self.store.adaptation_recommendation(song.song_id)
        self.assertEqual(rec["action"], "keep_searching")


if __name__ == "__main__":
    unittest.main()
