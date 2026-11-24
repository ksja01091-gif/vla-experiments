# OpenVLA를 실제로 조금 연습시킨 첫 기록

이번에는 학습 연결 시험을 실제로 실행했습니다. 아직 전체 자료를 이용한 본 실험은 아닙니다.

## 지금 무엇을 했나요?

사진 + 작업 지시 → 시범 행동을 토큰으로 변환한 정답과 비교 → loss 계산 → LoRA 값 수정 → 저장 → 새 Python 실행에서 다시 불러오기까지 확인했습니다.

- 시작 모델: openvla/openvla-7b (기본 모델). 기존 LIBERO Spatial 미세조정 모델과 구분합니다.
- LeRobot 데이터: 연습용 20개 시범에서 중간 장면 1장씩, 총 20장.
- 실제 업데이트: 20번, 한 번에 사진 1장. 전체 1,928장면을 모두 학습한 것이 아닙니다.
- 확인용 자료: 다른 5개 시범에서 중간 장면 1장씩, 총 5장. 이 장면들은 optimizer 업데이트에 넣지 않았습니다.
- 연습 장면의 전후 비교: 같은 장면 2장만 고정해서 비교했습니다.
- 변경한 부분: 언어 모델의 q_proj, v_proj에 LoRA rank 8 추가. 약 419만 개의 학습용 숫자(전체 모델 대비 약 0.056%)만 업데이트했습니다.
- 기존 예측 클래스와 지연 import 코드는 유지하고 새 학습 클래스를 추가했습니다.
- LeRobot 데이터와 정책을 사용하는 전용 PyTorch 학습 스크립트입니다. 기본 lerobot-train 명령의 모델 선택 factory까지 연결한 것은 아닙니다.

## 실제 결과

| 같은 장면으로 비교 | 연습 전 loss | 20번 연습 후 loss |
|---|---:|---:|
| 연습 장면 2장 | 12.4001 | 7.0854 |
| 확인 장면 5장 | 10.8770 | 8.8774 |

loss는 행동 토큰 예측의 오차를 나타냅니다. 낮을수록 이 채점 기준에서는 정답에 가까운 예측입니다.
위 숫자는 로봇의 성공률이 아닙니다. 모든 462개 확인 장면을 측정한 것도 아닙니다.

- LoRA 텐서 128개 묶음 모두 변화 확인. 가장 큰 값 변화는 약 0.002319.
- 손실과 기울기가 유효한 숫자인지 확인했습니다.
- optimizer에는 requires_grad=True인 LoRA 파라미터만 전달했습니다.
- 학습하지 않는 파라미터는 샘플 값을 뽑아 전후 동일한지 확인했습니다. 전체 모델 파일을 바이트 단위로 대조했다는 뜻은 아닙니다.
- 저장된 LoRA 파라미터 전체를 다시 불러왔을 때 정확히 일치했습니다.
- 새 Python 프로세스에서도 다시 로딩하고 같은 장면의 loss 7.9467635155가 재현됐습니다. 이 값은 확인 5장의 평균이 아니라 그중 한 장의 값입니다.
- 학습 후 select_action() 출력도 (1, 7), 모든 값이 유효한 숫자인지 확인했습니다.
- GPU 최대 할당 메모리 약 14.31 GiB. 이는 프로세스가 측정한 allocated 값으로 nvidia-smi 전체 사용량과 다를 수 있습니다.
- lerobot312 패키지 목록과 기존 정책·지연 import 코드가 실행 중 바뀌지 않았습니다. 패키지 설치나 버전 변경은 하지 않았습니다.

## 코드에서 두 군데만 이해하기

### 1. 실제로 값을 수정하는 두 줄 — PyTorch 기능

```python
loss.backward()
optimizer.step()
```

loss는 모델이 시범의 답과 얼마나 다른지 나타내는 숫자입니다.
backward()는 그 오차를 줄이려면 각 학습용 값을 어느 방향으로 바꿀지 계산합니다.
step()은 계산된 방향으로 LoRA 값을 실제로 수정합니다.
이 프로젝트에서는 train_openvla_lora_connection.py의 반복문 안에 있습니다. 사진을 한 장씩 보며 이 과정을 20번 반복했습니다.

### 2. 행동 부분만 채점하기 — Python 자르기 문법 + PyTorch 손실 규칙

```python
labels[:, :-8] = -100
```

:는 묶음 안의 모든 항목을, :-8은 뒤의 8개를 제외한 앞부분을 고릅니다.
-100은 이 손실 함수에서 채점에서 제외하는 값입니다.
마지막 행동 토큰 7개와 끝 표시 1개만 남기고, 지시문 부분은 채점하지 않습니다.
이 프로젝트에서는 training_openvla.py의 prepare_training_batch()에 있습니다.

## 사진과 시범 행동은 어떻게 넣었나요?

LeRobot에서 읽은 같은 장면의 사진·지시·행동을 사용했습니다.
행동은 연습용 20개 시범에서만 계산한 q01/q99로 팔의 6개 성분을 정규화하고, 집게 값은 별도로 유지합니다. 항상 같은 값인 성분은 0으로 둡니다.
공식 ActionTokenizer와 PurePromptBuilder를 원본 파일에서 직접 읽어 사용했습니다. 불필요한 prismatic 전체 의존성을 불러오지 않도록 해당 파일만 로딩했습니다.
숫자 7개가 실제로 토큰 7개로 변환되고 끝 표시까지 정답에 포함되는지 assertion으로 확인했습니다.
사진은 Hugging Face의 기본 image processor로 처리했습니다. 이번 연결 시험에서는 공식 RLDS 전체 사진 처리 과정이나 데이터 증강을 그대로 재현한 것은 아닙니다.
LoRA 대상도 공식 전체 all-linear 대신 언어 모델의 q_proj/v_proj로 제한했습니다. 먼저 작은 범위에서 학습 흐름을 확인하기 위한 선택입니다.

이 loss는 teacher forcing으로 계산합니다. 즉 다음 행동 토큰을 예측할 때 앞의 행동 토큰은 시범 정답을 넣어 계산합니다. 모델이 스스로 행동을 끝까지 생성하거나 시뮬레이터를 움직인 성능과는 다릅니다.

## 다시 실행하는 명령

원격 Linux에서 실행합니다. 학습 명령은 새 결과 폴더를 만듭니다. 기존 학습을 이어가는 resume 기능은 이번에 구현하지 않았습니다.

```bash
conda activate lerobot312
python /home/jovyan/projects/lerobot-openvla/examples/train_openvla_lora_connection.py --steps 20
```

이번에 저장한 결과를 새 프로세스에서 불러와 확인하려면:

```bash
python /home/jovyan/projects/lerobot-openvla/examples/check_openvla_trained_adapter.py --run /home/jovyan/projects/lerobot-openvla/outputs/openvla_training/20261102T070018_373350Z
```

## 핵심 파일

- /home/jovyan/projects/lerobot-openvla/src/lerobot/policies/openvla/training_openvla.py: TrainableOpenVLAPolicy, prepare_training_batch, forward, get_optim_params, save_training_adapter.
- /home/jovyan/projects/lerobot-openvla/examples/train_openvla_lora_connection.py: main, measure. 실제 optimizer 반복 실행과 기록.
- /home/jovyan/projects/lerobot-openvla/examples/check_openvla_trained_adapter.py: main. 새 실행에서 저장한 학습 결과를 읽어 확인.
- 결과 폴더: /home/jovyan/projects/lerobot-openvla/outputs/openvla_training/20261102T070018_373350Z
- adapter/: 실제 LoRA 가중치, 기본 모델 위치, 행동 정규화 값. 이 폴더만으로 전체 7B 모델이 되는 것은 아닙니다. 기본 모델도 필요합니다.
- run_config.json: 실제 입력 장면 번호, 학습 설정과 제한.
- result.json: 실제 전후 측정과 검증 결과.
- metrics.jsonl, metrics.csv: 업데이트별 손실·토큰 정확도·정규화 행동 L1 오차.
- fresh_process_check.json: 독립 재로딩 및 행동 출력 확인.
- training_connection.png: 실제 기록을 그린 그래프. 각 업데이트가 서로 다른 장면이므로 왼쪽 선이 매번 내려가야 하는 것은 아닙니다.

## 오류와 경고

학습을 중단시킨 Python 오류나 메모리 부족은 발생하지 않았습니다.
기존 Pydantic Field 관련 경고는 남아 있지만 실행은 통과했습니다.
저장 중 PEFT에서 config 파일을 확인할 수 없어 어휘가 바뀌지 않았다고 가정한다는 경고가 있었습니다.
실제 PEFT 저장 코드를 확인하면 오프라인 Hub 설정 조회 단계의 경고입니다. 이번에 어휘나 embedding 크기를 변경하지 않았고, 저장 후 가중치 전체 일치와 새 프로세스의 loss 재현으로 확인했습니다.

## 남은 일

전체 연습 장면 1,928개를 사용하는 데이터 반복 방식, 정식 사진 전처리 선택, 학습 횟수와 검증 주기, 학습 중단 기준을 정한 뒤 본 실험을 진행해야 합니다.
학습 전후의 같은 기본 모델과 같은 시뮬레이터 조건을 비교해야 실제 그릇 옮기기 개선 여부를 알 수 있습니다.
시뮬레이터 평가, W&B 연결, 전체 자료 학습, 완전한 학습 재개 기능은 이번에 실행하거나 구현했다고 주장하지 않습니다.
