"""First-person camera turning for the Open3D legacy viewer.

ViewControl.rotate() orbits the view around a target point, so it cannot turn
you on the spot while walking. This rotates the camera about its own position
instead, by editing the pinhole camera parameters directly.
"""
import numpy as np


def _axis_angle(axis, deg):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def turn(extrinsic, yaw_deg=0.0, pitch_deg=0.0, up=(0.0, 1.0, 0.0), max_pitch=0.98):
    """Return a new world->camera extrinsic turned about the camera's own position.

    Positive yaw turns right and positive pitch looks up (OpenCV camera: +z forward,
    +x right, +y down). Yaw spins about the world `up` axis so the horizon stays
    level; pitch tips about the camera's own right axis and is refused once the view
    would go within ~11 degrees of straight up/down (max_pitch is the cosine limit).
    """
    E = np.array(extrinsic, dtype=float)
    R, t = E[:3, :3], E[:3, 3]
    centre = -R.T @ t
    c2w = R.T                                    # columns: camera x (right), y (down), z (forward) in world
    up = np.asarray(up, float)

    if yaw_deg:
        c2w = _axis_angle(up, -yaw_deg) @ c2w    # sign: +yaw = turn right when +y is up
    if pitch_deg:
        tipped = _axis_angle(c2w[:, 0], pitch_deg) @ c2w
        if abs(tipped[:, 2] @ up) <= max_pitch:  # don't flip over the top
            c2w = tipped

    out = np.eye(4)
    out[:3, :3] = c2w.T
    out[:3, 3] = -c2w.T @ centre
    return out
