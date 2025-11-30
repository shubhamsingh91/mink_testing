# Mink IK Test

Inverse kinematics example using [Mink](https://github.com/kevinzakka/mink) with a UR5e robot.

## Setup

```bash
uv sync
```

## Run

```bash
# Normal mode (fast)
uv run python main.py

# Verbose mode (detailed iteration output + convergence plot)
uv run python main.py -v
```

The `-v` flag enables:
- Per-iteration printout of error norm, delta-q, and QP cost
- Saves a log-scale convergence plot to `convergence.png`

## Overview

This example:
1. Loads a UR5e robot model via `robot_descriptions`
2. Creates a `FrameTask` to track `wrist_3_link`
3. Sets a target pose (45deg rotation + 3cm offset)
4. Solves IK iteratively using the DAQP QP solver

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gain` | Convergence speed (0-1). Higher = faster but jerkier | 0.5 |
| `dt` | Integration timestep | 0.01 |
| `damping` | Levenberg-Marquardt regularization | 1e-6 |
| `solver` | QP solver (`daqp`, `quadprog`, `osqp`) | `daqp` |

## Performance

- ~0.4 ms per IK iteration (DAQP solver)
- Converges in 6-22 steps depending on gain
- Total solve time: 4-18 ms

## Available Frames

**Bodies:** `world`, `base`, `shoulder_link`, `upper_arm_link`, `forearm_link`, `wrist_1_link`, `wrist_2_link`, `wrist_3_link`

**Sites:** `attachment_site` (end-effector)