from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from tianji_lerobot_replay.model import prepare_urdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF = (
    PROJECT_ROOT
    / "models"
    / "ros_packages"
    / "daimon_tj_fx_description"
    / "urdf"
    / "daimon_tj_fx_dual_arm_with_gripper.urdf"
)


class ModelPreparationTests(unittest.TestCase):
    def test_generated_urdf_uses_vendor_gripper_cad_zero_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared = prepare_urdf(
                URDF,
                Path(temporary) / "prepared.urdf",
                [PROJECT_ROOT / "models" / "ros_packages"],
            )
            root = ET.parse(prepared).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        for name in (
            "left_endeffector_joint",
            "left_endeffector_mimic_joint",
            "right_endeffector_joint",
            "right_endeffector_mimic_joint",
        ):
            joint = joints[name]
            self.assertEqual(joint.find("origin").get("rpy"), "0 0 0")
            self.assertEqual(joint.find("axis").get("xyz"), "-1 0 0")
            self.assertEqual(joint.find("limit").get("upper"), "1e-9")
            self.assertIsNone(joint.find("mimic"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
