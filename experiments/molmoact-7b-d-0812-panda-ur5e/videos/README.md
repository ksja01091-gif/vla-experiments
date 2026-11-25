# MolmoAct 7B Pick Place D0 videos

All clips use the external camera and simulator seed 1007.

| Recording | Panda | UR5e |
| --- | --- | --- |
| 01. Pretrained base model, stopped at the wall-time limit | [141 steps](panda/01_Panda_base_model_timeout_141steps_seed1007.mp4) | [144 steps](ur5e/01_UR5e_base_model_timeout_144steps_seed1007.mp4) |
| 02. One-step simulator check with the trained LoRA adapter | [1 step](panda/02_Panda_trained_LoRA_1step_smoke_test_seed1007.mp4) | [1 step](ur5e/02_UR5e_trained_LoRA_1step_smoke_test_seed1007.mp4) |
| 03. Trained LoRA evaluation, stopped at the wall-time limit | [582 steps](panda/03_Panda_trained_LoRA_timeout_582steps_seed1007.mp4) | [566 steps](ur5e/03_UR5e_trained_LoRA_timeout_566steps_seed1007.mp4) |
| 04. Trained LoRA evaluation, continued to the 800-step limit | [800 steps](panda/04_Panda_trained_LoRA_full_eval_800steps_seed1007.mp4) | [800 steps](ur5e/04_UR5e_trained_LoRA_full_eval_800steps_seed1007.mp4) |

For each robot, recording 04 resumes recording 03 from its last simulator state. Count them as one evaluation. Both trained models reached 800 steps without completing the task. The one-step clips verify that the trained policy runs in the simulator.
