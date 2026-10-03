"""Check W&B trial logging and separation of selection from test evaluation."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rank3_sweep as sweep


def fake_run(config):
    run = MagicMock()
    run.__enter__.return_value = run
    run.config, run.summary = config, {}
    run.url = "https://wandb.ai/test/fsnm/runs/eval"
    return run


class SweepTests(unittest.TestCase):
    def test_learner_parameters(self):
        tree = sweep.fit_parameters({**sweep.DEFAULTS, "min_samples_leaf": 301.7})
        self.assertEqual(tree["min_samples_leaf"], 302)
        self.assertNotIn("n_knots", tree)
        spline = sweep.fit_parameters({**sweep.DEFAULTS, "learner_type": "linear_spline"})
        self.assertNotIn("min_samples_leaf", spline)
        self.assertEqual(spline["spline_degree"], 3)

    def test_both_learners_log_best_checkpoint_without_test_data(self):
        for learner in ("tree", "linear_spline"):
            config = {**sweep.DEFAULTS, "learner_type": learner, "n_train": 1000,
                      "n_validation": 400, "n_iterations": 5, "patience": 2,
                      "min_samples_leaf": 30, "n_knots": 5}
            run = fake_run(config)
            with patch.object(sweep.wandb, "init", return_value=run), \
                 patch.object(sweep, "sample_joint", wraps=sweep.sample_joint) as draw:
                sweep.train()
            self.assertEqual([call.args[1] for call in draw.call_args_list], [12, 100])
            validation = [call.args[0]["validation_loss"] for call in run.log.call_args_list]
            self.assertEqual(run.summary["best_validation_loss"], min(validation))
            self.assertEqual(run.summary["best_iteration"], int(np.argmin(validation))+1)
            self.assertEqual(run.summary["status"], "ok")

    def test_numerical_failure_is_failed_run(self):
        run = fake_run(sweep.DEFAULTS)
        with patch.object(sweep.wandb, "init", return_value=run), \
             patch.object(sweep, "fit", side_effect=np.linalg.LinAlgError("singular")):
            sweep.train()
        self.assertEqual(run.summary["status"], "failed")
        run.finish.assert_called_once_with(exit_code=1)
        self.assertNotIn("best_validation_loss", run.summary)
        run.log.assert_not_called()

    def test_active_or_all_failed_sweep_does_not_evaluate(self):
        for runs, message in (([SimpleNamespace(state="running")], "active"),
                              ([SimpleNamespace(state="failed")], "successful")):
            api = MagicMock()
            api.sweep.return_value.runs = runs
            with patch.object(sweep.wandb, "Api", return_value=api), \
                 patch.object(sweep, "fit", side_effect=AssertionError("No fitting")):
                with self.assertRaisesRegex(ValueError, message):
                    sweep.evaluate_best("test/fsnm/sweep", Path("unused"))

    def test_evaluation_selects_validation_winner(self):
        config = {**sweep.DEFAULTS, "n_train": 1000, "n_validation": 400,
                  "n_iterations": 5, "patience": 2, "min_samples_leaf": 30}
        model = sweep.fit(config)
        score = float(model[3]["validation_loss"].min())
        best = SimpleNamespace(state="finished", id="best", url="best-url", config=config,
                               summary={"status": "ok", "best_validation_loss": score,
                                        "best_iteration": model[3]["best_iteration"]})
        worse = SimpleNamespace(state="finished", id="worse", config=config,
                                summary={"status": "ok", "best_validation_loss": score+1})
        api = MagicMock()
        api.sweep.return_value.runs = [worse, best, SimpleNamespace(state="failed")]
        run = fake_run(config)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(sweep.wandb, "Api", return_value=api), \
             patch.object(sweep.wandb, "init", return_value=run), \
             patch.object(sweep.wandb, "Image"), \
             patch.object(sweep, "comparison_figure", return_value=MagicMock()), \
             patch("matplotlib.pyplot.close"), patch("builtins.print"), \
             patch.object(sweep, "sample_joint", wraps=sweep.sample_joint) as draw:
            summary = sweep.evaluate_best("test/fsnm/sweep", Path(directory))
            self.assertEqual(summary["selected_run"], "best-url")
            self.assertEqual([c.args[1] for c in draw.call_args_list], [12, 100, 12, 101])
            self.assertTrue((Path(directory) / "summary.json").exists())
            run.log.assert_called_once()


if __name__ == "__main__":
    unittest.main()
