"""Cross-embodiment dual-TCP replay and offline kinematic evaluation.

The MuJoCo path replays a trajectory with Aik7. The numerical evaluation path
independently checks pose reachability, joint margin and Jacobian metrics. An
optional workspace path estimates position occupancy with explicit MuJoCo
collision filtering. None of these tools constitutes real-robot safety validation.
"""

from .errors import ReplayError

__all__ = ["ReplayError"]
