# Dual RM75-B description

`urdf/dual_rm75_b.urdf` is the assembled model used by the action-space
visualization. It contains two official RealMan RM75-B seven-axis chains, the
complete in-house `URDF_DM25PD03-DA_20260803` UMI grippers, and the simulation
pedestal assembled from the four supplied DM-TR01 STEP parts. Each gripper
includes the full jaw geometry, main camera, binocular camera, and the
DM25PD04-EG Link2 mounting plate.

This is a simulation assembly rather than an official RealMan dual-arm product
URDF. The arm mounting transforms are fixed at:

- left: `xyz="-0.060000 0.050000 -0.089500"`,
  `rpy="-1.57079632679 0 0"`;
- right: `xyz="-0.060000 -0.050000 -0.089500"`,
  `rpy="-1.57079632679 0 3.14159265359"`.

The stand meshes in `meshes/stand` are tessellated from
`DM-TR01-01-01A.STEP` through `DM-TR01-01-04A.STEP`; their original millimetre
units are converted using the URDF mesh scale `0.001 0.001 0.001`. For a more
compact assembly, the original 450-mm cross beam is scaled to 100 mm along its
length axis (`0.001 0.001 0.000222222222`), without changing its cross-section. The arm
base interface planes are coincident with the two cross-beam end planes; the
arms are not suspended from visually disconnected frames.

Moving joints are `left_joint_1`--`left_joint_7` and
`right_joint_1`--`right_joint_7`; the gripper camera joints are fixed. The data
TCP links `left_tcp` and `right_tcp` are the UMI gripper URDF's `base_link`.
Each RM75-B `link_7` is attached to the DM25 mounting plate at identity, then
the plate-to-TCP transform is `xyz="0.010346 0 0.074711"`. Mesh references use
`package://dual_rm75_b_description/...`; there are no developer-machine
absolute paths. The UMI gripper mesh is deliberately renamed
`umi_gripper_base_link.STL` because MuJoCo keys imported assets by basename and
would otherwise display the RM75-B `base_link.STL` in its place.

The RM75-B arm meshes originate from the official RealMan `rm_models`
repository and retain its Apache-2.0 license. The gripper description originates
from the local `ugripper_urdf` package, whose package metadata declares BSD.
