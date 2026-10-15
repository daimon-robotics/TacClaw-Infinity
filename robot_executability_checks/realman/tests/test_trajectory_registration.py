import unittest
from tianji_lerobot_replay.trajectory_registration import assess_registration


def report(left, right):
    return {'sides': [{'side': side, 'coordinate_frame': frame,
                       'trajectory_status': 'success'}
                      for side, frame in [('left', left), ('right', right)]]}


class RegistrationTests(unittest.TestCase):
    def test_mixed_frames_rejected_despite_success(self):
        result = assess_registration(report('apriltag_frame', 'slam_world_frame'),
                                     reference_identity_verified=True,
                                     robot_from_reference_verified=True)
        self.assertFalse(result['eligible_for_robot_evaluation'])

    def test_two_slam_worlds_are_not_shared(self):
        result = assess_registration(report('slam_world_frame', 'slam_world_frame'),
                                     reference_identity_verified=True,
                                     robot_from_reference_verified=True)
        self.assertFalse(result['eligible_for_robot_evaluation'])

    def test_tag_frame_label_is_not_physical_registration(self):
        self.assertFalse(assess_registration(report('apriltag_frame', 'apriltag_frame'))[
            'eligible_for_robot_evaluation'])

    def test_verified_common_reference_accepted(self):
        self.assertTrue(assess_registration(report('apriltag_frame', 'apriltag_frame'),
                                           reference_identity_verified=True,
                                           robot_from_reference_verified=True)[
            'eligible_for_robot_evaluation'])
