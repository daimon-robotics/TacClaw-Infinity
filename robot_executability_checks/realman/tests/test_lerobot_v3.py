from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tianji_lerobot_replay.lerobot_v3 import (
    CANONICAL_LEFT_FIELDS,
    CANONICAL_RIGHT_FIELDS,
    load_episode,
    open_dataset,
)


class LeRobotV3Tests(unittest.TestCase):
    def test_named_layout_is_resolved_even_when_feature_columns_are_reordered(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))

        names = [
            "unused",
            *CANONICAL_RIGHT_FIELDS,
            "another_unused",
            *CANONICAL_LEFT_FIELDS,
        ]
        left = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        right = np.array([4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (root / "data" / "chunk-000").mkdir(parents=True)
            info = {
                "codebase_version": "v3.0",
                "fps": 30,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "features": {"observation.state": {"shape": [16], "names": names}},
            }
            (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
            rows = []
            for frame in range(2):
                values = np.zeros(16, dtype=float)
                for name, value in zip(CANONICAL_LEFT_FIELDS, left + frame, strict=True):
                    values[names.index(name)] = value
                for name, value in zip(CANONICAL_RIGHT_FIELDS, right + frame, strict=True):
                    values[names.index(name)] = value
                rows.append(values.tolist())
            pq.write_table(
                pa.table(
                    {
                        "observation.state": rows,
                        "timestamp": [2.0, 2.1],
                        "episode_index": [7, 7],
                        "index": [30, 31],
                    }
                ),
                root / "data" / "chunk-000" / "file-000.parquet",
            )
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [7],
                        "length": [2],
                        "data/chunk_index": [0],
                        "data/file_index": [0],
                        "dataset_from_index": [30],
                        "dataset_to_index": [32],
                    }
                ),
                root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            )
            dataset = open_dataset(root)
            episode = load_episode(dataset, 7)

        np.testing.assert_allclose(episode.timestamps_s, [0.0, 0.1])
        np.testing.assert_allclose(episode.positions_world_m[0, 0], left[:3])
        np.testing.assert_allclose(episode.positions_world_m[0, 1], right[:3])
        np.testing.assert_allclose(episode.quaternions_world_xyzw[1, 0], (left + 1.0)[3:])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
