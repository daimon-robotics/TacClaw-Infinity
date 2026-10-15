# Robot code import

The three robot versions are snapshots of `tianji_lerobot_replay`, imported on 2026-10-13 into the existing TacClaw-Infinity repository. The public repository keeps all three directories under `robot_executability_checks/` on `main`.

| Directory | Source branch | Source commit |
| --- | --- | --- |
| `robot_executability_checks/tianji/` | `main` | `7b52be740a8d6975f744db3579a7c571d2ed0583` |
| `robot_executability_checks/dm-sparky/` | `DM-Sparky` | `1de318e32e521aaf1a838618c844d141f23ee18c` |
| `robot_executability_checks/realman/` | `realman` | `37b0900f9d70abc900a3acf484925b0f8a6d205a` |

## Layout and compatibility

The three robot directories were grouped under `robot_executability_checks/` after the initial import. Each robot retains the same internal layout; documentation links and clone/working-directory examples follow the new paths.

- Each directory retains the source version's code, tests, configuration, and robot model assets. Existing model metadata and license notices are retained.
- Imports are snapshots; the source repository's commit history and unrelated branches are not imported.
- Installation and navigation examples use this public repository and the robot directories. Optional stereo-depth processing requires a separate `stereo_depth` installation and model weights.
- The code locates models and configuration relative to its own robot directory. No developer-specific absolute paths are required.
- All three versions use the Python package name `tianji_lerobot_replay`. Run them in separate processes and environments; do not combine their `src/` paths.
- DM-Sparky and RealMan retain the original Tianji compatibility code and defaults. Follow their robot-specific entry points in the root README instead of assuming every generic entry point selects the directory's robot.
- Coordinate conventions, scene calibration, algorithms, and evaluation thresholds are unchanged by the import.
- Datasets, generated videos, evaluation outputs, caches, compiled libraries, and Python environments are not included.

## Tests

From the robot directory, after installation and environment activation:

```bash
PYTHONPATH=src python -m unittest -v
```

The native Aik7 tests require a compiled library. Actual dataset replay and rendering additionally require a correctly registered LeRobot v3 dataset and, for interactive viewing, a graphics context.

## Import validation (2026-10-13)

- All 504 source files are present; only the three robot README files were edited.
- 115 existing tests passed across the three versions (36 Tianji, 38 DM-Sparky, 41 RealMan).
- The shared C++ Aik7 backend compiled on macOS arm64. The 12 native tests were run separately against the resulting `.dylib`, because the upstream test discovery checks for the Linux `.so` filename.
- All nine quick-start/replay/workspace CLI help checks and the six shell syntax checks passed.
- No real-robot execution, full dataset evaluation, or video rendering was performed as part of the import.
