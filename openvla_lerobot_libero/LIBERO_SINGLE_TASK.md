# 공식 조건을 맞춘 LIBERO 작업 1회 실행

이 단계에서는 이미 학습된 공식 LIBERO Spatial 모델로 작업 하나를 끝까지 실행합니다.
학습, W&B, 전체 벤치마크 성공률 측정은 하지 않습니다.

## 이전 데모와 이번 실행의 차이

| 항목 | 이전 이동 데모 | 이번 단일 작업 실행 |
|---|---|---|
| 시작 장면 | seeded reset | 공식 제공 초기 상태 복원 |
| 모델 입력 사진 | PIL 리사이즈 | 공식 JPEG 처리/Lanczos3/중앙 90% 면적 crop |
| 행동 한도 | 30회 | 공식 Spatial 한도 220회 |
| 준비 동작 | 10회 | 공식 설정과 같은 10회 |
| 종료 판단 | 30회 도달 또는 done | 성공 판정 또는 공식 한도 도달 |
| 확인 목적 | 움직이는가 | 이 작업을 이 시작 상태에서 성공하는가 |

## 숫자 세 가지를 구분하기

- task-id 0: 어떤 문제를 풀지 선택합니다. 검은 그릇을 접시에 놓는 첫 작업입니다.
- init-state-index 0: 제공된 시작 상태 중 첫 번째를 선택합니다. 작업 번호와 다른 값입니다.
- step 1,2,...: 이번 시도에서 로봇에게 행동을 전달한 횟수입니다. 학습 횟수가 아닙니다.

시작 상태는 로봇과 물체의 상태를 복원할 수 있는 숫자 목록입니다.
같은 상태를 사용해야 모델을 바꿨을 때 비슷한 출발 조건에서 비교할 수 있습니다.
이번에는 결과가 잘 나오는 상태를 고르지 않고 처음부터 0번으로 고정했습니다.

## 파일과 실행 흐름

1. examples/openvla_libero_single_task.py: 공식 한도 읽기, 정책 실행, 성공/실패 결과 및 영상 저장.
2. examples/libero_eval_worker.py: 시뮬레이터 생성, 공식 시작 상태 복원, 10회 준비 동작, env.step(), 성공 확인.
3. examples/openvla_eval_preprocessing.py: 공식 이미지 함수 본문을 읽어 동일한 이미지 연산을 실행하고 픽셀 대조 검사.
4. 기존 modeling_openvla.py: LeRobot 정책의 select_action()으로 실제 행동 예측. 이전 파일을 수정하지 않습니다.

이미지 전처리는 별도의 TensorFlow CPU 환경에서 실행합니다.
파생 환경: /home/jovyan/projects/venvs/openvla_eval310
이 환경은 기존 openvla310 패키지를 읽되 추가 설치는 자기 폴더에만 합니다.
모델은 기존 lerobot312에서 실행합니다.

## 확인한 전처리 조건

- 카메라 256x256, 화면 방향 180도 회전.
- 공식 JPEG encode/decode와 Lanczos3 리사이즈.
- 중앙 90% 면적 crop 후 224x224 입력 생성.
- 무작위/검은 이미지 및 실제 첫 관측 사진을 공식 전체 전처리 함수와 비교.
- 모델에 전달하기 위한 tensor/PIL 왕복에서도 픽셀 값이 바뀌지 않는지 검사.
- 원래 카메라 사진은 frames/, 실제 모델 입력은 policy_frames/에 따로 저장.

## 읽어볼 코드 두 가지

### 1. 같은 출발선 복원

```python
obs = env.set_init_state(state)
```

Python 기본 문법: 오른쪽 함수의 반환값을 obs 변수에 저장합니다.
LIBERO 기능: set_init_state는 제공된 상태로 시뮬레이터를 복원하고 관측값을 반환합니다.
역할: 매번 임의로 배치하는 대신, 선택한 공식 시작 상태 0번에서 출발합니다.

### 2. 실제 성공했을 때 종료

```python
if current["task_success"]:
    break
```

Python 기본 문법: 딕셔너리의 task_success 값을 꺼내 참이면 반복문을 끝냅니다.
LIBERO 기능: task_success는 env.check_success()가 현재 상태의 목표 조건을 검사한 결과입니다.
역할: 영상을 보고 사람이 대충 성공이라고 정하는 대신 환경 판정을 기록합니다.
성공하지 않으면 220회까지 진행하고 horizon_exhausted로 기록합니다.

## 재실행

```bash
conda activate lerobot312
cd /home/jovyan/projects/lerobot-openvla
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -u examples/openvla_libero_single_task.py --task-id 0 --init-state-index 0
```

결과는 outputs/libero_single_task/<UTC 실행 식별자>/에 저장됩니다.
result.json은 성공 여부와 실제 행동 횟수, initialization.json은 시작 상태와 전처리 검증 기록입니다.
steps.jsonl에는 매 단계 예측/전달 행동과 관측을 기록합니다.
episode.mp4와 episode.gif는 전체 실행, episode_overview.png는 진행 장면 6개입니다.
preprocessing_comparison.png에서는 원래 화면과 모델이 받은 화면을 비교합니다.

## 결과 해석

한 작업·한 시작 상태의 시도입니다. 성공해도 전체 성공률이 100%라는 뜻은 아닙니다.
실패해도 전체 모델이 실패한다는 뜻은 아닙니다.
공식 전처리, 초기 상태, 한도, 집게 변환을 맞췄지만 GPU attention 구현과 패키지 버전까지 논문의 실험과 완전히 같은 것은 아닙니다.
모델 attention backend와 환경 버전, 원본 참조 코드 해시를 result.json에 남깁니다.

## 이번 실행의 실제 결과

{
  "success": true,
  "policy_steps": 85,
  "warmup_steps": 10,
  "max_policy_steps": 220,
  "frames": 86,
  "simulated_seconds": 4.249999999999699,
  "init_state_index": 0,
  "available_initial_states": 50,
  "max_pixel_error": 0,
  "attention_backend": "sdpa",
  "existing_files_unchanged": 8,
  "original_environments_unchanged": true,
  "video_decoding_passed": true,
  "single_trial_only": true
}
