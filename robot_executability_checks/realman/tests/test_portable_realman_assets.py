from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from tianji_lerobot_replay.workspace_specs import get_workspace_spec


class PortableRealManAssetsTests(unittest.TestCase):
    def test_model_is_packaged_not_an_output_artifact(self):
        path = get_workspace_spec('realman').urdf
        self.assertTrue(path.is_file())
        self.assertIn('models/ros_packages/dual_rm75_b_description/', path.as_posix())
        self.assertNotIn('outputs', path.parts)

    def test_all_meshes_resolve_inside_package(self):
        path = get_workspace_spec('realman').urdf
        package = path.parents[1]
        meshes = ET.parse(path).findall('.//mesh')
        self.assertTrue(meshes)
        prefix = 'package://dual_rm75_b_description/'
        for mesh in meshes:
            uri = mesh.get('filename', '')
            with self.subTest(uri=uri):
                self.assertTrue(uri.startswith(prefix))
                target = (package / uri[len(prefix):]).resolve()
                self.assertTrue(target.is_relative_to(package.resolve()))
                self.assertTrue(target.is_file(), target)

    def test_tcp_and_seven_joint_names_match_evaluator(self):
        spec = get_workspace_spec('realman')
        tree = ET.parse(spec.urdf)
        links = {link.get('name') for link in tree.findall('link')}
        joints = {joint.get('name') for joint in tree.findall('joint')}
        for side in ('left', 'right'):
            self.assertIn(spec.ee_bodies[side], links)
            self.assertEqual(len(spec.joint_names[side]), 7)
            self.assertTrue(set(spec.joint_names[side]).issubset(joints))
