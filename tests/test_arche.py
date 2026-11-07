import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import arche


def sample_logs():
    rows = [
        ["2024-09-01 10:00:00", "a", "cours", "Système", "Cours consulté"],
        ["2024-09-02 10:00:00", "a", "fiche", "Fichier", "Module de cours consulté"],
        ["2024-09-03 00:00:00", "a", "quiz", "Test", "Tentative de test commencée"],
        ["2024-09-01 12:00:00", "b", "cours", "Système", "Cours consulté"],
        ["2024-09-02 12:00:00", "c", "cours", "Test", "Utilisateur évalué"],
    ]
    logs = pd.DataFrame(rows, columns=arche.LOG_COLUMNS)
    logs["heure"] = pd.to_datetime(logs["heure"])
    return logs


class PreparationTests(unittest.TestCase):
    def test_cutoff_is_strict(self):
        features, info = arche.prepare_features(sample_logs(), "2024-09-03")
        self.assertEqual(features.loc["a", "n_test"], 0)
        self.assertEqual(features.loc["a", "n_fichier"], 1)
        self.assertEqual(features.loc["a", "jours_actifs"], 2)
        self.assertEqual(info["observed_events"], 4)
        self.assertEqual(info["evaluation_events_excluded"], 1)

    def test_future_activity_cannot_change_features(self):
        logs = sample_logs()
        features, _ = arche.prepare_features(logs, "2024-09-03")
        future = logs.iloc[[0]].copy()
        future["heure"] = pd.Timestamp("2026-01-01")
        future["pseudo"] = "nouveau"
        changed, _ = arche.prepare_features(pd.concat([logs, future]), "2024-09-03")
        pd.testing.assert_frame_equal(features, changed)

    def test_missing_traces_not_zero_filled(self):
        features, _ = arche.prepare_features(sample_logs(), "2024-09-03")
        notes = pd.DataFrame({"pseudo": ["a", "b", "c", "absent"], "note": [8, 12, 10, 9]})
        dataset = arche.make_dataset(features, notes)
        self.assertEqual(list(dataset.index), ["a", "b"])

    def test_no_ids_or_labels_as_features(self):
        self.assertNotIn("pseudo", arche.FEATURES)
        self.assertNotIn("note", arche.FEATURES)
        features, _ = arche.prepare_features(sample_logs())
        self.assertEqual(list(features.columns), arche.FEATURES)

    def test_reader_audits_duplicates_and_keeps_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            logs = sample_logs()
            pd.concat([logs, logs.iloc[[0]]]).to_csv(path / "logs.csv", index=False)
            pd.DataFrame({"pseudo": ["a", "b"], "note": [9, 11]}).to_csv(
                path / "notes.csv", index=False
            )
            original = (path / "logs.csv").read_bytes()
            cleaned, _, audit = arche.read_data(path)
            self.assertEqual(len(cleaned), len(logs))
            self.assertEqual(audit["duplicate_events_removed"], 1)
            self.assertEqual((path / "logs.csv").read_bytes(), original)

    def test_invalid_grades_rejected(self):
        for grades in [[9, 30], [9, np.inf], [9, np.nan]]:
            with self.subTest(grades=grades), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                sample_logs().to_csv(path / "logs.csv", index=False)
                pd.DataFrame({"pseudo": ["a", "b"], "note": grades}).to_csv(
                    path / "notes.csv", index=False
                )
                with self.assertRaises(ValueError):
                    arche.read_data(path)

    def test_duplicate_grade_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            sample_logs().to_csv(path / "logs.csv", index=False)
            pd.DataFrame({"pseudo": ["a", "a"], "note": [9, 11]}).to_csv(
                path / "notes.csv", index=False
            )
            with self.assertRaises(ValueError):
                arche.read_data(path)

    def test_empty_cutoff(self):
        features, _ = arche.prepare_features(sample_logs(), "2020-01-01")
        self.assertTrue(features.empty)

    def test_missing_columns_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            sample_logs().drop(columns="heure").to_csv(path / "logs.csv", index=False)
            pd.DataFrame({"pseudo": ["a"], "note": [9]}).to_csv(path / "notes.csv", index=False)
            with self.assertRaisesRegex(ValueError, "colonnes requises"):
                arche.read_data(path)

    def test_bad_dates_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            logs = sample_logs().astype({"heure": "string"})
            logs.loc[0, "heure"] = "invalide"
            logs.to_csv(path / "logs.csv", index=False)
            pd.DataFrame({"pseudo": ["a"], "note": [9]}).to_csv(path / "notes.csv", index=False)
            with self.assertRaises(ValueError):
                arche.read_data(path)


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def dataset():
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            rng.uniform(1, 15, size=(20, len(arche.FEATURES))),
            columns=arche.FEATURES, index=[str(i) for i in range(20)],
        )
        frame.index.name = "pseudo"
        frame["note"] = [9.0, 11.0] * 10
        return frame

    def test_splits_disjoint_complete_and_repeatable(self):
        labels = np.array([0, 1] * 10)
        splits = arche.make_splits(labels, 5, 2, 42)
        again = arche.make_splits(labels, 5, 2, 42)
        for repeat in range(2):
            held_out = []
            for i in range(repeat * 5, (repeat + 1) * 5):
                train, test = splits[i]
                self.assertFalse(set(train) & set(test))
                self.assertEqual(len(set(train) | set(test)), 20)
                np.testing.assert_array_equal(test, again[i][1])
                held_out.extend(test)
            self.assertEqual(sorted(held_out), list(range(20)))

    def test_too_few_in_class_rejected(self):
        with self.assertRaises(ValueError):
            arche.make_splits([0] * 8 + [1] * 2, 5, 2, 42)

    def test_failure_is_positive_for_recall(self):
        result = arche.score_predictions([9, 8, 11], [9, 12, 12], [0, 0, 1], [0, 1, 1])
        self.assertAlmostEqual(result["rappel_echec"], 0.5)
        self.assertAlmostEqual(result["precision_echec"], 1)

    def test_scaling_only_fitted_on_training(self):
        dataset = self.dataset()
        model = arche.make_models()["lineaire"]
        model.fit(dataset.iloc[:15][arche.FEATURES], dataset.iloc[:15]["note"])
        scaler = model.named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, dataset.iloc[:15][arche.FEATURES].mean())
        self.assertEqual(scaler.n_samples_seen_, 15)

    def test_individual_target_excluded(self):
        dataset = self.dataset()
        features = dataset[arche.FEATURES]
        grade, training = arche.predict_learner(features, dataset, "0", "lineaire")
        self.assertNotIn("0", training.index)
        changed = dataset.copy()
        changed.loc["0", "note"] = 20
        second, _ = arche.predict_learner(features, changed, "0", "lineaire")
        self.assertEqual(grade, second)
        self.assertEqual(len(training), 19)

    def test_unknown_target_rejected(self):
        dataset = self.dataset()
        with self.assertRaises(ValueError):
            arche.predict_learner(dataset[arche.FEATURES], dataset, "absent", "lineaire")

    def test_cross_validation_outputs(self):
        data = self.dataset()
        predictions, metrics, summary, matrices, paired = arche.evaluate(
            data, folds=2, repeats=2
        )
        self.assertEqual(len(predictions), 20 * 4 * 2)
        self.assertEqual(len(metrics), 4 * 2)
        for _, part in predictions.groupby(["model", "repeat"]):
            self.assertEqual(part.pseudo.nunique(), 20)
            self.assertFalse(part.pseudo.duplicated().any())
        self.assertEqual(set(summary), {"moyenne", "majoritaire", "lineaire", "foret"})
        self.assertNotIn("mae", summary["majoritaire"])
        self.assertEqual(np.array(matrices["lineaire"][0]).sum(), 20)
        self.assertIn("balanced_accuracy", paired)

    def test_recommendation_does_not_claim_forest_winner_on_tie(self):
        values = {
            "balanced_accuracy": {"mean": 0.6, "min": 0.55, "max": 0.65},
            "rappel_echec": {"mean": 0.6}, "mae": {"mean": 1.0},
        }
        summary = {
            "lineaire": values, "foret": values,
            "majoritaire": {"balanced_accuracy": {"mean": 0.5}},
        }
        self.assertIn("sans déploiement", arche.recommendation(summary))


if __name__ == "__main__":
    unittest.main()
