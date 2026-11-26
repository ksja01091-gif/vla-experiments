# Panda can pick-and-place with 100 scripted demonstrations

MolmoAct2 LoRA rank 32, training seed 777, batch size 4, and 4,000 optimizer steps.

| Metric | Result |
| --- | ---: |
| Grasped and lifted | 26 / 50 (52%) |
| Placed in target bin | 4 / 50 (8%) |
| Training wall time | 4628 s |
| Evaluation wall time | 2287 s |

The model was evaluated at the 50 fixed held-out starts (seeds 30000–30049). Each trial ran for up to 1,000 environment steps.

- [Results by starting position](summary.json)
- [Team W&B run with training curves and example videos](https://forge.coreweave.com/wandb/gist-infonet/xenoarm-proposal/runs/panda-can-k100-s777)
- [All 50 dual-camera evaluation videos](videos/README.md)
