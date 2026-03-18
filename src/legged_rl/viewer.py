from importlib.resources import files

import mujoco
import mujoco.viewer
import numpy as np


def viewer():
    model = mujoco.MjModel.from_xml_path(
        str(files("legged_rl").joinpath("mujoco_menagerie/unitree_go2/scene.xml"))
    )
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)  # 0 = first keyframe ("home")

    print("Initial qpos:", data.qpos)
    print("Joints range:", model.jnt_range)
    print("Joints limited:", model.jnt_limited)
    print("Joints number:", model.njnt)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()

    tuned_qpos = data.qpos.copy()
    print("Your tuned stable qpos:\n", tuned_qpos)
    # np.save("stable_go2_qpos.npy", tuned_qpos)


if __name__ == "__main__":
    viewer()
