import unittest

from tianji_lerobot_replay.arm_quality import summarize, Thresholds


class QualityStatisticsTests(unittest.TestCase):
    def row(self, reachable, eta):
        row = {"both_reachable": reachable}
        for side in ("left", "right"):
            row.update({f"{side}_reachable": reachable,
                        f"{side}_isotropy_ratio": eta,
                        f"{side}_condition_number": 1 / eta if eta else None,
                        f"{side}_position_error_m": 0. if reachable else .5,
                        f"{side}_orientation_error_deg": 0.,
                        f"{side}_joint_margin_fraction": .1})
        return row

    def test_low_manip_denominator_contains_reachable_frames_only(self):
        result = summarize("independent", [self.row(True, .03), self.row(True, .04),
                                            self.row(False, None)], Thresholds())
        self.assertAlmostEqual(result["both_reachable_rate"], 2 / 3)
        self.assertEqual(result["arms"]["left"]["low_manipulability_frames"], 1)
        self.assertEqual(result["arms"]["left"]["low_manipulability_rate_among_reachable"], .5)

    def test_all_unreachable_is_na_not_zero_low_manip(self):
        result = summarize("independent", [self.row(False, None)], Thresholds())
        self.assertIsNone(result["arms"]["left"]["low_manipulability_rate_among_reachable"])
        self.assertIsNone(result["arms"]["left"]["min_isotropy_ratio"])
