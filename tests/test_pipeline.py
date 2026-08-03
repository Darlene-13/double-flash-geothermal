import unittest

from src.pipeline import build_dataset, validate_dataset
from src.train_surrogate import chronological_slices, safe_features


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = build_dataset()

    def test_generator_observations_are_aligned(self):
        self.assertEqual(len(self.frame), 186)
        self.assertEqual(self.frame["observation_id"].nunique(), 186)

    def test_history_features_do_not_use_current_load(self):
        expected = self.frame["gen1_load_MW"].shift(1).rolling(3).mean()
        self.assertTrue(expected.equals(self.frame["gen1_load_roll3"]))

    def test_model_features_are_leakage_safe(self):
        features = safe_features(self.frame, "gen1")
        self.assertNotIn("gen1_load_MW", features)
        self.assertFalse(any("steam_utilization" in value for value in features))
        self.assertFalse(any("turbine_isentropic_eff" in value for value in features))
        self.assertFalse(any(value.endswith("exergy_efficiency") for value in features))

    def test_physical_validation(self):
        report = validate_dataset(self.frame)
        self.assertGreater(report["repeated_date_observations"], 0)
        self.assertEqual(report["quality_outside_0_1"], 0)

    def test_chronological_splits_do_not_overlap(self):
        train, validation, test = chronological_slices(183)
        self.assertEqual(train.stop, validation.start)
        self.assertEqual(validation.stop, test.start)


if __name__ == "__main__":
    unittest.main()
