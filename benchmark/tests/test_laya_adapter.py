from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_laya_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_laya_benchmark", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


class LayaAdapterTests(unittest.TestCase):
    def test_call_supported_drops_none_for_var_kwargs(self) -> None:
        captured = {}

        def loader(*args, **kwargs):
            captured.update(kwargs)
            return args

        mod.call_supported(loader, "model", dtype=None, batch_size=None, device="gpu")
        self.assertEqual(captured, {"device": "gpu"})

    def test_choice_normalization(self) -> None:
        questions = {
            "route": {
                "type": "choice",
                "instructions": "route",
                "criteria": {"a": "A", "b": "B"},
            }
        }
        result = {"answers": {"route": {"probabilities": {"a": 0.2, "b": 0.8}}}}
        out = mod.validate_and_normalize_result(result, questions, "test/model")
        self.assertEqual(out["answers"]["route"]["choice"], "b")
        self.assertAlmostEqual(out["answers"]["route"]["confidence"], 0.8)

    def test_noul_and_score_normalization(self) -> None:
        questions = {
            "flag": {"type": "noul", "instructions": "flag"},
            "level": {"type": "score", "instructions": "level", "criteria": ["low", "high"]},
        }
        result = {
            "answers": {
                "flag": {"noul": 0.25},
                "level": {"probabilities": {"0": 0.4, "1": 0.6}},
            }
        }
        out = mod.validate_and_normalize_result(result, questions, "test/model")
        self.assertEqual(out["answers"]["flag"]["type"], "noul")
        self.assertAlmostEqual(out["answers"]["level"]["score"], 0.6)

    def test_mock_backend_all_primitives(self) -> None:
        backend = mod.MockBackend("mock/model")
        questions = {
            "n": {"type": "noul", "instructions": "n"},
            "c": {"type": "choice", "instructions": "c", "criteria": {"x": "x", "y": "y"}},
            "s": {"type": "score", "instructions": "s", "criteria": ["a", "b", "c"]},
        }
        out = mod.validate_and_normalize_result(backend.predict("state", questions), questions, "mock/model")
        self.assertEqual(set(out["answers"]), set(questions))


    def test_truncated_input_classification(self) -> None:
        response = {"usage": {"input_tokens": 1024}}
        questions = {"q": {"type": "noul", "instructions": "q"}}
        status, comparable, reason, details = mod.classify_success_compatibility(
            response, questions, {"max_len": 1024, "head_max_len": 256}
        )
        self.assertEqual(status, "truncated_input")
        self.assertFalse(comparable)
        self.assertIn("context ceiling", reason or "")
        self.assertEqual(details["aggregate_full_input_ceiling"], 1024)

    def test_full_input_classification(self) -> None:
        response = {"usage": {"input_tokens": 811}}
        questions = {"q": {"type": "noul", "instructions": "q"}}
        status, comparable, reason, _ = mod.classify_success_compatibility(
            response, questions, {"max_len": 1024, "head_max_len": 256}
        )
        self.assertEqual(status, "full_input")
        self.assertTrue(comparable)
        self.assertIsNone(reason)

    def test_unsupported_cardinality_classification(self) -> None:
        status, comparable, reason = mod.classify_error_compatibility(
            "Question 'target' has too many options for the token budget", "ValueError"
        )
        self.assertEqual(status, "unsupported_cardinality")
        self.assertFalse(comparable)
        self.assertIn("too many options", reason or "")


    def test_legacy_backfill_marks_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.jsonl"
            mod.write_jsonl(raw, [{
                "case_id": "c1",
                "ok": True,
                "request": {"questions": {"q": {"type": "noul", "instructions": "q"}}},
                "response": {"usage": {"input_tokens": 1024}},
            }])
            changed = mod.upgrade_legacy_compatibility(raw, {"max_len": 1024, "head_max_len": 256})
            self.assertEqual(changed, 1)
            row = mod.read_jsonl(raw)[0]
            self.assertEqual(row["compatibility_status"], "truncated_input")
            self.assertFalse(row["comparable"])

    def test_selected_cases_file_matches_max_cases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "results" / "run_test"
            source.mkdir(parents=True)
            (root / "benchmark").mkdir()
            cases = []
            for i in range(3):
                cases.append({
                    "case_id": f"case_{i}",
                    "experiment": "test",
                    "state": "state",
                    "questions": {"q": {"type": "noul", "instructions": "q"}},
                    "expected": {"q": {"type": "noul", "label": 1}},
                    "metadata": {},
                })
            mod.write_jsonl(source / "cases.jsonl", cases)
            (source / "manifest.json").write_text(json.dumps({"case_count": 3}))
            out = mod.run_source(
                study_root=root,
                source_run=source,
                backend=mod.MockBackend("mock/model"),
                checkpoint="typed-decisions",
                output_root=root / "out",
                max_cases=2,
                resume=False,
                fail_fast=True,
                progress_every=100,
                budget_overrides={},
                do_analyze=False,
            )
            self.assertEqual(len(mod.read_jsonl(out / "cases.jsonl")), 2)
            self.assertEqual(len(mod.read_jsonl(out / "raw.jsonl")), 2)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["case_count"], 2)


if __name__ == "__main__":
    unittest.main()
