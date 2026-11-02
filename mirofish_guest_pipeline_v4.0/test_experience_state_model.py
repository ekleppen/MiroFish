import unittest
from experience_state_model import build_experience_state, need_kind


class TestExperienceState(unittest.TestCase):
    def test_repair(self):
        e = {"service_type": "Repair", "category": "HVAC", "item": "A/C Unit"}
        x = build_experience_state(e, severity="moderate", resolution="unresolved")
        self.assertEqual(x["events"][0]["need_kind"], "problem")
        self.assertEqual(x["relative_satisfaction_pressure"], -0.25)
        self.assertIn("not been resolved", x["guest_context"])

    def test_resolution_changes_problem_pressure(self):
        e = {"service_type": "Repair", "category": "HVAC", "item": "A/C Unit"}
        u = build_experience_state(e, resolution="unresolved")
        r = build_experience_state(e, resolution="resolved")
        self.assertGreater(r["relative_satisfaction_pressure"], u["relative_satisfaction_pressure"])

    def test_routine_request_not_problem(self):
        e = {"service_type": "Request", "category": "Towels & Terry", "item": "Bath Towel"}
        x = build_experience_state(e)
        self.assertEqual(x["events"][0]["need_kind"], "request")
        self.assertIsNone(x["relative_satisfaction_pressure"])
        self.assertIn("not necessarily", x["guest_context"])

    def test_complaint_is_problem_even_if_request(self):
        e = {"service_type": "Request", "category": "Guest Services", "item": "Noise Complaint"}
        self.assertEqual(need_kind(e), "problem")

    def test_unspecified_experience(self):
        x = build_experience_state(None)
        self.assertEqual(x["events"], [])
        self.assertIsNone(x["relative_satisfaction_pressure"])

    def test_invalid_assumptions_rejected(self):
        with self.assertRaises(ValueError):
            build_experience_state(None, severity="extreme")


if __name__ == '__main__':
    unittest.main()
