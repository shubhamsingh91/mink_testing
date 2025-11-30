import numpy as np
import time

from robot_descriptions.loaders.mujoco import load_robot_description
from mink.configuration import Configuration
from mink.tasks.frame_task import FrameTask
from mink.lie.se3 import SE3
from mink.lie.so3 import SO3
from mink.solve_ik import solve_ik, build_ik
import mujoco


def main(verbose: bool = False):
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

    # For tracking history (when verbose)
    history = {"step": [], "error": [], "qp_cost": [], "dq": []}

    if verbose:
        print(f"{'step':>4} | {'||error||':>10} | {'dq':^40} | {'QP cost':>10}")
        print("-" * 75)

    start_time = time.perf_counter()
    for step in range(max_steps):
        e = frame_task.compute_error(config)
        error_norm = np.linalg.norm(e)
        if error_norm < 1e-6:
            print(f"\nConverged in {step} steps")
            break
        v = solve_ik(
            configuration=config,
            tasks=[frame_task],
            dt=dt,
            solver="daqp",
            damping=1e-6,
            safety_break=False,
        )

        dq = v * dt

        if verbose:
            # Build QP to get the actual cost matrices
            qp = build_ik(config, [frame_task], dt, damping=1e-6)
            dq_str = np.array2string(dq, precision=4, suppress_small=True, floatmode='fixed')
            # Actual QP cost: 0.5 * v^T H v + c^T v
            qp_cost = 0.5 * v @ qp.P @ v + qp.q @ v
            print(f"{step:>4} | {error_norm:>10.6f} | {dq_str:>40} | {qp_cost:>10.6f}")

            # Store history
            history["step"].append(step)
            history["error"].append(error_norm)
            history["qp_cost"].append(qp_cost)
            history["dq"].append(dq.copy())

        # integrate v into q
        config.integrate_inplace(v, dt)
        config.update()

    elapsed = time.perf_counter() - start_time
    if step == max_steps - 1:
        print("\nDid not converge within the step limit.")
    print(f"Time elapsed: {elapsed*1000:.2f} ms")
    print("error norm:", np.linalg.norm(e))
    print("Final q:", config.q)
    print("final v:", v)

    # Plot convergence if verbose
    if verbose and len(history["step"]) > 0:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Error norm (log scale)
        axes[0].semilogy(history["step"], history["error"], 'b.-', label="||error||")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Error norm (log)")
        axes[0].set_title("Task Error Convergence")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # QP cost (log scale)
        axes[1].semilogy(history["step"], history["qp_cost"], 'r.-', label="QP cost")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("QP cost (log)")
        axes[1].set_title("QP Cost Convergence")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("convergence.png", dpi=150)
        print("\nSaved convergence plot to convergence.png")
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed iteration info and plot convergence")
    args = parser.parse_args()
    main(verbose=args.verbose)
