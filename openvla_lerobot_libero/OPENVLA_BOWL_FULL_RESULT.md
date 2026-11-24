# OpenVLA 그릇 옮기기 본 실험 결과

공개 LIBERO 데이터에서 선택·전처리한 실험용 분할로 실행한 결과입니다.
시범 20개 전체의 1,928장면을 한 번씩 학습했고, 별도 시범 5개의 462장면 전체를 검증했습니다.

설정: 1 epoch, batch 4, 학습률 0.0001, LoRA rank 8 (q_proj/v_proj), 482회 업데이트.
학습 루프 시간(중간·마지막 검증 포함): 3.88분. GPU 최대 allocated: 14.92 GiB.

| 전체 검증 지표 | 학습 전 | 학습 후 |
|---|---:|---:|
| loss | 11.8272 | 3.2220 |
| 행동 토큰 정확도 | 9.03% | 35.50% |
| 정규화 행동 L1 | 0.4605 | 0.2408 |

| 모델 | 성공/시도 | 성공률 | 평균 episode step |
|---|---:|---:|---:|
| 기본 OpenVLA | 0/5 | 0.0% | 220.0 |
| 이번 LoRA 학습 후 | 0/5 | 0.0% | 220.0 |
| 공개 Spatial 미세조정 모델 (참고) | 4/5 | 80.0% | 112.0 |

검증 오차 감소와 시뮬레이터 성공률은 별개의 결과입니다. 이번 설정에서 실제로 확인한 결과만 위에 기록했습니다.
세 모델의 초기 상태·초기 이미지 일치를 확인했고 기술적 오류는 없었습니다. 초기 상태 0~4, seed 7, 환경 seed 0, 준비 10회, 최대 220회 행동입니다.
기본 모델과 LoRA는 같은 학습용 행동 정규화 통계를 사용했습니다. 공개 Spatial 모델의 결과를 기본 모델 결과로 섞지 않았습니다.

## 실패의 관찰 가능한 양상

- 기본 OpenVLA, 대표 초기 상태 0: 집게와 목표 그릇 중심의 최단거리 37.6cm, 그릇 최대 높이 증가 0.000mm.
- 이번 LoRA 학습 후, 대표 초기 상태 0: 집게와 목표 그릇 중심의 최단거리 10.1cm, 그릇 최대 높이 증가 7.434mm.
재생한 제어 명령의 로봇 궤적과 성공 판정은 원래 기록과 일치했습니다. 이 수치는 물체 움직임의 관찰이며, 신경망의 어떤 부분이 근본 원인인지는 확정하지 않았습니다.
대표 장면에서는 학습 후 집게가 그릇 근처에서 접시 쪽으로 움직여도 그릇이 테이블에 남는 모습이 보입니다.

## 저장과 확인

- 선택 checkpoint: /home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z/checkpoints/step_000482
- 새 Python 프로세스의 가중치 재로딩·동일 입력 loss 대조·유효한 (1,7) 행동 출력 확인.
- 영상 15개를 직접 열어 20 FPS, 첫·마지막 프레임 및 실제 프레임 수를 확인.
- 기존 Python 소스와 openvla310·lerobot312·eval310의 패키지 목록을 대조했습니다.
- W&B는 인증 정보가 없어 offline 기록을 준비했으며 업로드하지 않았습니다. WANDB_SYNC_COMMANDS.txt를 로그인 후 직접 실행하면 됩니다.

## 다시 실행

```bash
conda activate lerobot312
python /home/jovyan/projects/lerobot-openvla/examples/run_openvla_bowl_full.py
```
새 출력 폴더에 새로운 실험을 실행하는 명령입니다. 기존 결과는 덮어쓰지 않습니다.

```bash
python /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl_experiment.py check --run /home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z
```
위 명령은 이번 저장 checkpoint를 새 프로세스에서 다시 확인합니다.

## 핵심 파일과 개념

- /home/jovyan/projects/lerobot-openvla/src/lerobot/policies/openvla/full_experiment_openvla.py: 배치 구성, forward, LoRA 저장·재로딩.
- /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl_experiment.py: 전체 학습, 전체 검증, 같은 조건의 시뮬레이션.
- 전체 결과: /home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z
- 수치와 원본 로그: /home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z/experiment_summary.json, training_metrics.csv, validation_metrics.csv, simulation_summary.csv.
- 결과 그래프: /home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z/experiment_summary.png
- 최종 영상 장면: /home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z/rollout_endpoints.png

epoch는 학습 장면 전체를 한 번 사용하는 단위입니다. loss는 시범 행동 예측 오차이고 성공률은 실제 제어 시도의 목표 달성 비율입니다.

## 남은 일과 해석 범위

W&B 온라인 연결에는 사용자 로그인이 필요합니다. 실력 개선을 주장하려면 다음 설정 실험과 더 많은 초기 상태 평가가 필요합니다.
이번은 한 작업·작은 공개 데이터 분할·1 epoch의 최소 베이스라인입니다. 전체 LIBERO 논문 수치 재현이나 통계적으로 확정된 일반화 성능이라고 해석하지 않습니다.
학습 사진 처리는 기존 HF processor를 유지했습니다. 공식 RLDS 전체 학습 레시피의 재현과는 구분합니다.
