# Panda MolmoAct2 evaluation on four-object pick-and-place

The final rank-32 LoRA adapter was trained for 67,389 optimizer steps on 400 Panda MimicGen
Pick Place D0 demonstrations, using training seed 777 and batch size four. Evaluation ran
in the XenoArm MuJoCo pp4 environment on seeds 0–49. One success requires all four
objects in their matching bins, with the success predicate held for 10 steps.

**Complete task success: 29/50 (58%).**

| Object | In its matching bin at end |
| --- | ---: |
| Milk | 47/50 |
| Bread | 39/50 |
| Cereal | 37/50 |
| Can | 33/50 |

Panda ACT evaluation starts with identical simulator-state hashes: 38/50. On those 38 starts, ACT completed 0 and MolmoAct2 completed 22.

[Team W&B evaluation](https://forge.coreweave.com/wandb/gist-infonet/xenoarm-proposal/runs/molmoact2-pp4-panda-k400-r32-seed777-evaluation)

## Dual-camera clips

- [Seed 00, failure, 1000 steps](media/seed-0.mp4)
- [Seed 01, failure, 1000 steps](media/seed-1.mp4)
- [Seed 02, failure, 1000 steps](media/seed-2.mp4)
- [Seed 03, success, 621 steps](media/seed-3.mp4)
- [Seed 04, success, 614 steps](media/seed-4.mp4)
- [Seed 07, failure, 1000 steps](media/seed-7.mp4)
- [Seed 14, failure, 1000 steps](media/seed-14.mp4)
