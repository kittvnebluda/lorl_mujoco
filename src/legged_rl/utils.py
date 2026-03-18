import numpy as np


def sanitize_for_hparams(d: dict) -> dict:
    clean = {}
    for k, v in d.items():
        match v:
            case None:
                clean[k] = "None"
            case int() | float() | str() | bool():
                clean[k] = v
    return clean


def quat2euler(quat):
    """
    Converts quaternion (w in first place) to euler roll, pitch, yaw
    quaternion = [w, x, y, z]
    Source: https://gist.github.com/salmagro/2e698ad4fbf9dae40244769c5ab74434
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
