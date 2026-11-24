# Teacher rollout → 2B student: result

All rows use the same official LIBERO Spatial bowl task and the same 50 initial states.
The student trained on successful teacher trajectories from states 5–44 only.
States 0–4 and 45–49 were excluded from the new distillation data.

| Model | All 50 | Distillation states 5–44 | Evaluation-only states 0–4, 45–49 |
|---|---:|---:|---:|
| teacher_7B | 32/50 | 25/40 | 7/10 |
| student_2B_before | 0/50 | 0/40 | 0/10 |
| student_2B_after | 0/50 | 0/40 | 0/10 |

Successful teacher training episodes: 25
New teacher image-action pairs: 2129
Existing human image-action pairs mixed in training: 4025
Student training time: 9.6 minutes
Student GPU peak allocated: 4.71 GiB
Human validation loss: 3.2556 → 3.2290

Interpretation: the 50-state total is descriptive, not an independent generalization estimate.
The 10 evaluation-only states form the more relevant comparison, though this is still a small sample.
A lower validation loss alone does not prove higher simulator success.
The teacher collection on fixed official states cannot create unlimited distinct start scenes.
