import unittest

from posture import (
    compute_posture_score,
    initialize_calibration_state,
    update_calibration_state,
)


class Phase6Tests(unittest.TestCase):
    def test_calibration_state_collects_baseline(self):
        state = initialize_calibration_state()
        update_calibration_state(state, {"neck_angle": 10.0, "spine_angle": 8.0, "shoulder_tilt": 3.0, "head_tilt": 4.0})
        update_calibration_state(state, {"neck_angle": 12.0, "spine_angle": 9.0, "shoulder_tilt": 4.0, "head_tilt": 5.0})

        self.assertTrue(state["is_calibrated"])
        self.assertAlmostEqual(state["baseline"]["neck_angle"], 11.0)
        self.assertAlmostEqual(state["baseline"]["spine_angle"], 8.5)

    def test_posture_score_is_clamped(self):
        score = compute_posture_score(120.0)
        self.assertEqual(score, 0)

        score = compute_posture_score(10.0)
        self.assertEqual(score, 100)


if __name__ == "__main__":
    unittest.main()
