# 그릇 옮기기 본 실험 결과

공개 LIBERO 데이터에서 선택·전처리한 실험용 분할로 실행했습니다.
전체 시범 20개의 1,928장면을 1 epoch 학습하고, 다른 시범 5개의 462장면 전체를 평가했습니다.

- 학습률: 0.0001, batch size: 4, LoRA rank: 8, 업데이트: 482회.
- 학습 루프 시간(중간·최종 검증 및 저장 포함): 3.88분.
- 최대 GPU allocated 메모리: 14.92 GiB.
- 전체 validation loss: 11.8272 → 최종 3.2220.
- 선택한 checkpoint: /home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z/checkpoints/step_000482 (validation loss 3.2220).

| 모델 | 성공/시도 | 성공률 | 평균 step | 기술적 오류 |
|---|---:|---:|---:|---:|
| Base OpenVLA | 0/5 | 0.0% | 220.0 | 0 |
| LoRA trained | 0/5 | 0.0% | 220.0 | 0 |
| Public Spatial ref. | 4/5 | 80.0% | 112.0 | 0 |

기본 OpenVLA와 LoRA는 학습용 20개 시범에서 계산한 동일한 행동 정규화 값을 사용했습니다. 공개 Spatial 미세조정 모델은 별도 참고 기준입니다.
초기 상태 0~4, seed 7, 환경 seed 0, 준비 동작 10회, 최대 220회 행동, 기존 공식 전처리 및 제어 명령 변환을 유지했습니다.
초기 상태·초기 이미지 일치 확인: True. 학습 자료와 평가 초기 상태가 서로 겹치지 않는다는 보장은 없습니다.
loss 감소와 시뮬레이터 성공률은 서로 다른 결과입니다. 5회 결과만으로 일반적인 성능 향상이나 통계적 유의성을 주장하지 않습니다.
토큰 정확도와 L1은 teacher forcing으로 계산하며, L1은 정규화 행동 단위입니다. 미터 단위나 성공률이 아닙니다.
사진 전처리는 기존 연결 시험의 HF processor를 유지했으며, 공식 RLDS 전체 학습 레시피를 그대로 재현했다는 뜻은 아닙니다.

학습 설정·validation 지표·checkpoint·환경 보존 검사는 experiment_summary.json에, 업데이트별 기록은 CSV와 JSONL에 남겼습니다.
원본 데이터·checkpoint·optimizer 상태·영상은 로컬 서버에 보존합니다. W&B는 인증이 없어 offline 기록이며 아직 업로드하지 않았습니다.

## 다시 실행하는 명령

원격 Linux의 lerobot312 환경에서 examples/openvla_bowl_experiment.py의 calibrate, train, check, simulate, audit 단계를 사용합니다.
기존 결과 폴더를 덮어쓰지 않도록 새 실험 폴더를 만들고 환경 기준 기록부터 남겨야 합니다. 이번 실험을 다시 평가할 때는 저장된 checkpoint와 동일한 run 경로를 사용하세요.

## 이번에 기억할 두 개념

epoch 하나는 학습용 장면 전체를 한 번 사용하는 것입니다. 시범 20개는 사진 20장이 아닙니다.
validation loss는 시범 행동 예측 오차이고, 시뮬레이터 성공률은 실제 제어 반복 끝에 목표를 달성한 비율입니다.