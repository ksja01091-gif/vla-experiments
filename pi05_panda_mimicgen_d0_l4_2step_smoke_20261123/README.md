# π0.5 + Panda + MimicGen D0: L4 학습 연결 시험 (2026-11-23)

## 이 실험은 무엇을 했나?

연구실 Google Colab의 **NVIDIA L4**에서 공개 MimicGen Panda 시범을 π0.5 학습 코드에 넣고, 가중치 업데이트가 실제로 끝나는지 확인했다. 이 폴더 이름의 `2step_smoke`는 **학습 업데이트 2번만 한 연결 시험**이라는 뜻이다.

| 항목 | 이번에 실행한 값 |
| --- | --- |
| 공개 데이터 | `OliverHausdoerfer/demo_src_pick_place_task_D0_robot_Panda_gripper_PandaGripper` |
| 사용한 시범 | 번호 0, 1의 **완성된 시범 2판** |
| 읽어 들인 프레임 | 1,392개. 이 중 학습 업데이트에 사용된 샘플은 2개 |
| 기본 모델 | `lerobot/pi05_base` |
| 학습 방법 | 비전 인코더 고정, **행동 전문가만 학습** (`train_expert_only=true`). **LoRA 시험은 아님** |
| 설정 | BF16, batch size 1, 업데이트 2번, seed 7, gradient checkpointing 사용 |
| 결과 | 오류 없이 2번 업데이트하고 `End of training` 출력 |
| 각 업데이트의 loss | 0.548, 0.268 |
| 학습 로그의 최고 GPU 메모리 | 12.85 GB (`mem_gb`) |
| 체크포인트 | 저장하지 않음 (`save_checkpoint=false`) |

공개 데이터 전체에는 Panda 시범 500판이 있다. 노트북에서 제안한 **학습 100판 / 검증 20판 / 별도 평가 20판** 분할은 이번 2회 업데이트에서 사용하지 않았다. 따라서 이 결과로 검증 성능이나 집어서 놓기 성공률을 말할 수 없다. 두 loss 값만으로 학습 효과를 판단할 수도 없다.

## 파일을 보는 순서

1. [`result.json`](result.json): 이번 시험의 설정과 결과를 짧게 정리한 파일.
2. [`panda_smoke.log`](panda_smoke.log): `lerobot-train`이 실제로 출력한 학습 기록. 마지막의 `step:1`, `step:2`, `End of training`을 보면 된다.
3. [`run_panda_l4_smoke.py`](run_panda_l4_smoke.py): 똑같은 2회 업데이트를 다시 실행하는 명령.
4. [`Pi05_MimicGen_Panda_D0_data_audit_and_L4_smoke.ipynb`](Pi05_MimicGen_Panda_D0_data_audit_and_L4_smoke.ipynb): 데이터 구조 확인부터 실행까지 진행한 Colab 코드. GitHub용 복사본은 셀의 저장된 출력을 제거했으며 실제 실행 출력은 로그에 남겼다.

## 다시 실행하려면

GPU가 연결된 Colab에서 Hugging Face의 PaliGemma 접근 권한과 인증을 준비한 뒤:

```bash
pip install "lerobot[training,pi]==0.7.0"
python run_panda_l4_smoke.py
```

실행 로그는 `/content/panda_smoke.log`에 기록된다. 이 시험 코드는 모델 체크포인트를 저장하거나 기본 모델을 덮어쓰지 않는다.
