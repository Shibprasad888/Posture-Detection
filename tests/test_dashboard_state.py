import unittest

from posture import (
    initialize_dashboard_state,
    initialize_session_analytics,
    update_dashboard_state,
    update_session_analytics,
)


class DashboardStateTests(unittest.TestCase):
    def test_tracks_good_and_bad_posture_durations(self):
        state = initialize_dashboard_state()

        update_dashboard_state(state, "GOOD POSTURE", 0.25)
        update_dashboard_state(state, "FORWARD HEAD", 0.25)

        self.assertEqual(state["current_status"], "FORWARD HEAD")
        self.assertAlmostEqual(state["posture_durations"]["good"], 0.25)
        self.assertAlmostEqual(state["posture_durations"]["bad"], 0.25)

    def test_alerts_once_for_a_continuous_bad_posture_event(self):
        analytics = initialize_session_analytics(alert_duration_seconds=0.5)

        update_session_analytics(analytics, "FORWARD HEAD", 0.25)
        self.assertEqual(analytics["alerts_triggered"], 0)

        update_session_analytics(analytics, "FORWARD HEAD", 0.25)
        self.assertEqual(analytics["alerts_triggered"], 1)

        update_session_analytics(analytics, "GOOD POSTURE", 0.1)
        self.assertEqual(analytics["posture_corrections"], 1)
        self.assertEqual(analytics["active_alert"], False)


if __name__ == "__main__":
    unittest.main()
