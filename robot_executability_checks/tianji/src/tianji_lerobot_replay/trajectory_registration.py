"""Fail-closed registration checks for newly reconstructed raw trajectories."""


def assess_registration(report, *, reference_identity_verified=False,
                        robot_from_reference_verified=False):
    sides = {row['side']: row for row in report.get('sides', [])}
    frames = {side: sides.get(side, {}).get('coordinate_frame', 'missing')
              for side in ('left', 'right')}
    reasons = []
    for side in ('left', 'right'):
        row = sides.get(side, {})
        if row.get('trajectory_status') != 'success':
            reasons.append(f'{side}: trajectory reconstruction did not succeed')
    if any(frame not in ('apriltag_frame', 'head_world_frame') for frame in frames.values()):
        reasons.append('At least one arm has no common reference; local SLAM worlds cannot be combined')
    elif frames['left'] != frames['right']:
        reasons.append('Left and right use different coordinate frames')
    if not reference_identity_verified:
        reasons.append('Physical reference identity and origin/axes are not verified')
    if not robot_from_reference_verified:
        reasons.append('Rigid reference-to-robot/table registration is missing')
    return dict(coordinate_frames=frames, eligible_for_robot_evaluation=not reasons,
                blocking_reasons=reasons, reachability=None, manipulability=None)
