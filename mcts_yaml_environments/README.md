# MCTS planning evaluation environments

This archive contains 30 YAML environments: 3 variants per experimental setting.

## Experiment 1 — State-space size
- `exp1_state_small_*.yaml`: 5 x 5 x 4 = 100 cells/states
- `exp1_state_medium_*.yaml`: 5 x 8 x 5 = 200 cells/states
- `exp1_state_large_*.yaml`: 8 x 10 x 5 = 400 cells/states
- One drone, 3D movement enabled, zero obstacles.

## Experiment 2 — Action-space size (number of drones)
- `exp2_action_1drone_*.yaml` through `exp2_action_4drone_*.yaml`
- Fixed 10 x 10 x 10 grid, zero obstacles, 3D movement enabled.
- Three variants for each drone count.

## Experiment 3 — Obstacle clutter
- `exp3_clutter_0obs_*.yaml`, `exp3_clutter_2obs_*.yaml`, `exp3_clutter_4obs_*.yaml`
- Fixed 5 x 5 x 1 grid, one drone, 2D movement (`change_altitude: False`).
- Obstacle counts are 0, 2, and 4 (0%, 8%, and 16% of 25 cells).
- Start and goal cells are not obstacles; the listed layouts preserve a route.

## MDP parameters
All files use exactly:
- `step_cost: -1.0`
- `collision_penalty: -50.0`
- `goal_reward: 100.0`
- `discount_factor: 0.98`
- `max_num_steps: 100`
- `alpha: 0.5`

## Notes
Coordinates are `[x, y, z]`. Experiment 1 and 2 use `change_altitude: True`;
Experiment 3 uses `change_altitude: False`.
