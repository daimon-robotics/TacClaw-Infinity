# Native analytic IK backend

This directory packages the second Tianji IK implementation for the standalone
replay tool: the `Aik7_ow` closed-form 7-axis solver used by the ROS workspace
whole-body/WBC path.  It intentionally does **not** include the first
Pinocchio/HPIPM incremental HQP teleoperation controller, the old `WbcType`
robot/torso placement constants, Ruckig, or ROS runtime code.

The solver source in `vendor/` was copied verbatim from Git commit `1dc1bde`
of the adjacent `ros2_ws_tianji` repository:

- `Aik7_ow.h` and left/right `aik7_ow_*_tj.cpp`;
- the shared analytic model type declarations.

`src/current_urdf_model.cpp` is a standard-DH model derived specifically from
this repository's complete Tianji URDF.  The ROS workspace's historical
`Dm25Model` uses a different base/end-frame convention and must not be written
directly into this MuJoCo model.  The bridge exports both Aik7 IK and that
model's FK; Python refuses to run if FK disagrees with the loaded URDF.

`src/aik7_bridge.cpp` is new glue with a small C ABI for Python `ctypes`.  Its
input is a row-major `base_link ← ee_link` transform.  The Python replay layer
first obtains the fixed `gripper_base_link (TCP) → Link7 → ee_link` transforms
dynamically from the same MuJoCo/URDF model, so UMI TCP data is never passed
directly to the analytic solver.

Build it with `./setup.sh`, or manually:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release
```

The host needs a C++17 compiler, CMake, and Eigen headers (`libeigen3-dev` on
Debian/Ubuntu).  `native/build/` is generated and intentionally ignored by Git.
