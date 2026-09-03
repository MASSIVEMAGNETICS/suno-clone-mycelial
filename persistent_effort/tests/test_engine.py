import json
import tempfile
import unittest
from pathlib import Path

from persistent_effort.engine import PersistentStore


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

    def test_adaptation_gate_stays_closed_early(self):
        song = self.store.ingest_song(self.song_file, "owned")
        rec = self.store.adaptation_recommendation(song.song_id)
        self.assertEqual(rec["action"], "keep_searching")


if __name__ == "__main__":
    unittest.main()
