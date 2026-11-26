# Panda ACT · Pick and Place

변환한 MimicGen 시범 400판을 사용해 ACT로 Panda의 집어서 놓기 동작을 학습하는 실험입니다. XenoArm pp4에서는 우유·빵·시리얼·캔을 각각의 목표 칸에 놓아야 성공입니다.

[실험 결과표](RESULTS.md) · [시드별 결과와 학습 시간](results.json)

| 항목 | 설정 |
| --- | --- |
| GPU | NVIDIA TITAN RTX 24 GiB 한 장 |
| 학습 | ACT, 시범 400판, 시드 777, 3,000 optimizer step, batch 4 |
| 평가 | XenoArm pp4 시작 시드 0–49, 최대 1,000 동작, 5동작마다 재예측 |
| 모델 입력 | 외부·손목 카메라와 8차원 로봇 상태 |

기존 MolmoAct2 평가와 같은 시뮬레이터 리셋 시드를 사용합니다. 학습 시범의 원본 물건 좌표를 확보하지 못했으므로, 50곳을 좌표 기준으로 검증된 미경험 배치로 해석하지 않습니다.

## 실행 파일

- [run.sh](run.sh)는 데이터와 GPU를 점검한 뒤, 학습 시간을 측정하고 기존 XenoArm 평가 코드로 50개 장면을 실행합니다.
- [serve_act.py](serve_act.py)는 ACT의 행동값을 XenoArm 평가 코드에 전달합니다.
- [summarize.py](summarize.py)는 성공 횟수와 학습 시간, 시드별 결과를 검증한 뒤 JSON을 작성합니다.
- [publish_wandb.py](publish_wandb.py)는 검증된 결과를 팀 W&B에 게시하고 저장된 값을 확인합니다.

실행 폴더는 `dataset/`, 평가 버전의 `source/`, 가상환경 `train-env/`와 `sim-env/`, 위 스크립트의 `code/`로 구성됩니다. 학습 실행 시간은 학습 명령 시작부터 마지막 체크포인트 저장까지입니다. 데이터 복사, 환경 설치, 시뮬레이션 평가는 이 시간에서 제외합니다.
