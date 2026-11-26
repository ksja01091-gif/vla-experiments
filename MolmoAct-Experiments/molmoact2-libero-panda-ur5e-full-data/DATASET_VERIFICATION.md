# Dataset conversion verification

## Full exports

| Robot | Training episodes / frames | Validation episodes / frames | Test episodes / frames |
| --- | ---: | ---: | ---: |
| Panda | 400 / 269,553 | 50 / 33,567 | 50 / 33,259 |
| UR5e | 400 / 268,657 | 50 / 33,719 | 50 / 33,769 |

The split manifest lists source episode IDs. Within each robot, the three splits do not overlap. The stored episode lengths match the source for all 500 episodes per robot.

## Source-to-export spot check

One episode per split per robot was checked. Source episode IDs were 251 (train), 235 (validation), and 294 (test) for each robot. Across all action rows in these six episodes, each stored seven-value vector matched the XenoArm conversion within 1e-7. Inverse conversion matched the clipped source input within 1e-7.

The converter scales position commands to meters and rotation commands to radians. It maps signed gripper commands from -1 (open) and +1 (close) to 0 and 1.

Both decoded camera streams in the six episodes matched the source in length and frame rate. Five corresponding frames per camera and episode matched the expected horizontal flip subject to video re-encoding (mean absolute RGB difference 1.86–2.12 on the 0–255 scale).

The MolmoAct2 loader produced a batch from the one-episode probe export for each robot. This document records dataset and loader checks. Task success is measured later in simulation.
