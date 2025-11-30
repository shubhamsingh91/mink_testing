import numpy as np
import time

from robot_descriptions.loaders.mujoco import load_robot_description
from mink.configuration import Configuration
from mink.tasks.frame_task import FrameTask
from mink.lie.se3 import SE3
from mink.lie.so3 import SO3
from mink.solve_ik import solve_ik
import mujoco


def main():
    # 1) Load a MuJoCo model (UR5e) via robot_descriptions
    model = load_robot_description("ur5e_mj_description")  # MJCF model

    # 2) Wrap in Mink's Configuration
    config = Configuration(model)
    config.update()  # run FK once

    print(f"nq = {config.nq}, nv = {config.nv}")

    # print frame names
    print("Bodies:")
    for i in range(model.nbody):
        print(" -", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

    # 3) Define an end-effector frame task
    ee_frame_name = "wrist_3_link"

    frame_task = FrameTask(
        frame_name=ee_frame_name,
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
        gain=0.5,
    )

    # 4) Set a target a bit offset from the current EE pose
    current_T = config.get_transform_frame_to_world(
        frame_name=ee_frame_name,
        frame_type="body",
    )
    print("\n---------------------")

    print("Current EE pose: \t", current_T)

    # Create a target with rotation: 45 degrees around Z-axis
    rotation = SO3.from_z_radians(np.pi / 4)  # 45 deg rotation around Z
    translation = np.array([0.03, 0.0, 0.0])  # small offset in x
    offset = SE3.from_rotation_and_translation(rotation, translation)
    target_T = current_T @ offset
    frame_task.set_target(target_T)
    print("Target EE pose: \t", target_T)
    print("\n---------------------")
    
    # 5) Solve IK in a little loop
    dt = 0.01
    max_steps = 500
    start_time = time.perf_counter()
    for step in range(max_steps):
        if np.linalg.norm(frame_task.compute_error(config)) < 1e-6:
            print("\nConverged in: ", step, " steps")
            break
        v = solve_ik(
            configuration=config,
            tasks=[frame_task],
            dt=dt,
            solver="daqp",
            damping=1e-6,
            safety_break=False,
        )
        # integrate v into q
        config.integrate_inplace(v, dt)
        config.update()

        # monitor EE error norm
        e = frame_task.compute_error(config)

    elapsed = time.perf_counter() - start_time
    if step == max_steps - 1:
        print("\nDid not converge within the step limit.")
    print(f"Time elapsed: {elapsed*1000:.2f} ms")
    print("error norm:", np.linalg.norm(e))
    print("Final q:", config.q)
    print("final v:", v)


if __name__ == "__main__":
    main()
