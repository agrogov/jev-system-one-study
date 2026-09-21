from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("run_semif_benchmark", SCRIPTS / "run_semif_benchmark.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)
rl = mod.rl

QUESTIONS = {
    "flag": {"type": "noul", "instructions": "Is it paid?"},
    "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": "Bugs"}},
    "urgency": {"type": "score", "instructions": "Rate urgency", "criteria": ["none", "mild", "high"]},
}


def args(**overrides):
    ns = mod.parse_args(["--study-root", ".", "--backend", "mock"])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class SemIfAdapterTests(unittest.TestCase):
    def test_build_rows_maps_each_primitive(self) -> None:
        rows = {r["id"]: r for r in mod.build_rows("state", QUESTIONS)}
        self.assertEqual([o["id"] for o in rows["flag"]["options"]], ["yes", "no"])
        self.assertEqual([o["id"] for o in rows["route"]["options"]], ["billing", "tech"])
        self.assertEqual([o["id"] for o in rows["urgency"]["options"]], ["0", "1", "2"])
        self.assertEqual(rows["route"]["question"], "Which team?")
        self.assertTrue(all(r["state"] == "state" for r in rows.values()))

    def test_state_object_is_preserved(self) -> None:
        state = {"a": 1}
        self.assertEqual(mod.build_rows(state, QUESTIONS)[0]["state"], state)

    def test_unsupported_type_and_empty_criteria_raise(self) -> None:
        with self.assertRaises(ValueError):
            mod.build_rows("s", {"q": {"type": "rank", "instructions": "x"}})
        with self.assertRaises(ValueError):
            mod.build_rows("s", {"q": {"type": "choice", "instructions": "x", "criteria": {}}})

    def test_assemble_answers(self) -> None:
        results = [
            {"id": "flag", "option_ids": ["yes", "no"], "probabilities": [0.9, 0.1]},
            {"id": "route", "option_ids": ["billing", "tech"], "probabilities": [0.25, 0.75]},
            {"id": "urgency", "option_ids": ["0", "1", "2"], "probabilities": [0.0, 0.5, 0.5]},
        ]
        out = mod.assemble_answers(QUESTIONS, results)
        self.assertAlmostEqual(out["flag"]["noul"], 0.9)
        self.assertEqual(out["route"]["choice"], "tech")
        self.assertAlmostEqual(out["urgency"]["score"], 1.5)
        self.assertEqual(out["urgency"]["legend"], {"0": "none", "1": "mild", "2": "high"})

    def test_backend_predict_normalizes_and_selects_mode(self) -> None:
        backend = mod.mock_backend(args())
        single = backend.predict("s", {"flag": QUESTIONS["flag"]})
        self.assertEqual(single["provider_meta"]["mode"], "direct")
        multi = backend.predict("s", QUESTIONS)
        self.assertEqual(multi["provider_meta"]["mode"], "shared")
        normalized = rl.validate_and_normalize_result(multi, QUESTIONS, backend.info.model_ref)
        self.assertEqual(set(normalized["answers"]), set(QUESTIONS))
        self.assertIn("provider_meta", normalized)
        self.assertEqual(normalized["usage"]["output_tokens"], 0)

    def test_cardinality_and_length_refusals_are_classified(self) -> None:
        status, comparable, _ = rl.classify_error_compatibility("options must contain 2-16 entries", "ValueError")
        self.assertEqual((status, comparable), ("unsupported_cardinality", False))
        status, comparable, _ = rl.classify_error_compatibility(
            "Row q: 9000 input tokens exceed limit 4096; no truncation allowed", "ValueError"
        )
        self.assertEqual((status, comparable), ("input_too_long", False))
        self.assertEqual(rl.classify_error_compatibility("boom", "RuntimeError")[0], "backend_error")

    def test_oversized_choice_is_recorded_not_repaired(self) -> None:
        backend = mod.mock_backend(args())
        big = {"q": {"type": "choice", "instructions": "x", "criteria": {str(i): "d" for i in range(32)}}}
        with self.assertRaises(ValueError) as ctx:
            backend.predict("s", big)
        self.assertEqual(rl.classify_error_compatibility(str(ctx.exception), "ValueError")[0], "unsupported_cardinality")

    def test_end_to_end_mock_replay_writes_jev_compatible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmark").mkdir()
            run = root / "jev-results" / "run_test"
            run.mkdir(parents=True)
            cases = [
                {"case_id": "c1", "experiment": "e", "state": "s", "questions": {"flag": QUESTIONS["flag"]}, "expected": {}, "metadata": {}},
                {"case_id": "c2", "experiment": "e", "state": "s",
                 "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {str(i): "d" for i in range(20)}}},
                 "expected": {}, "metadata": {}},
            ]
            (run / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
            (run / "manifest.json").write_text("{}")
            out = root / "out"
            mod.main(["--study-root", str(root), "--backend", "mock", "--source-run", "run_test",
                      "--output-root", str(out), "--no-analyze", "--no-warmup"])
            (run_dir,) = list(out.iterdir())
            self.assertTrue(run_dir.name.startswith("run_test__semif_"))
            rows = [json.loads(l) for l in (run_dir / "raw.jsonl").read_text().splitlines()]
            self.assertEqual([r["ok"] for r in rows], [True, False])
            self.assertEqual(rows[0]["compatibility_status"], "full_input")
            self.assertEqual(rows[1]["compatibility_status"], "unsupported_cardinality")
            self.assertEqual(rows[0]["response_headers"]["x-local-provider"], "semif")
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["provider"], "semif-local")
            self.assertEqual(manifest["requests_comparable"], 1)


if __name__ == "__main__":
    unittest.main()
