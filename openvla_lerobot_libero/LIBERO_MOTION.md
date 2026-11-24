# OpenVLA → LeRobot → LIBERO/MuJoCo: 짧은 이동 연결 실험

이 단계의 목적은 모델이 예측한 행동으로 시뮬레이터 로봇이 실제로 움직이는지 확인하는 것입니다.
학습, W&B, 공식 벤치마크 성공률 평가는 수행하지 않습니다.

## 실행 구조

lerobot312 Python: 현재 사진 + 작업 지시 → OpenVLAPolicy.select_action() → 행동 텐서 (1,7).
openvla310 Python: 행동값 수신 → env.step() → 새 사진과 손끝/관절 위치 반환.
두 프로세스는 이 서버 내부의 표준 입력/출력 JSON 메시지로 연결됩니다.
기존 환경을 합치거나 패키지 버전을 바꾸지 않습니다.

## 재실행

```bash
conda activate lerobot312
cd /home/jovyan/projects/lerobot-openvla
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u examples/openvla_libero_motion.py --steps 30
```

처음 10개 no-op 행동은 물체와 로봇이 안정되도록 하는 준비 단계입니다.
그 이후 모델이 새 사진을 보고 한 번씩 예측하며 최대 30번 움직입니다.
환경 done 신호가 오면 일찍 중단합니다. 이 신호만으로 일반적인 성공률을 주장하지 않습니다.

## 새 파일

- examples/libero_motion_worker.py: main(), capture(), env.step(). 기존 openvla310에서 실행.
- examples/openvla_libero_motion.py: main(), controller_action(), save_media(). lerobot312에서 실행.
- outputs/libero_motion/<UTC 실행 식별자>/frames/: 매 단계의 실제 시뮬레이터 이미지.
- 같은 폴더의 steps.jsonl: 모델 원본 행동, 집게 변환 후 행동, 최종 명령, 이동 전후 상태.
- result.json: 이동 확인, 환경 버전, 초기화·전처리·실행 범위 기록.
- motion.gif / motion.mp4: 매 단계 화면. 보기 쉽게 초당 5프레임으로 재생.
- before_after.png: 시작과 마지막 화면 비교.
- simulator.log: 시뮬레이터 경고·오류 로그.
- outputs/libero_motion_audit/: 기존 핵심 파일 해시와 공식 집게 변환 대조 검사 기록.

## 예시로 이해하는 실행 흐름

1. 시뮬레이터가 사진 A와 작업 지시를 보냅니다.
2. 모델은 사진 A를 보고 다음 행동값 7개를 계산합니다.
3. 집게 값의 표현을 공식 OpenVLA 방식으로 변환합니다.
   예: 원본 0.99 → 2*0.99-1=0.98 → sign=1 → 부호 반전=-1.
4. 시뮬레이터가 env.step()으로 한 제어 주기를 진행합니다.
5. 바뀐 사진 B와 실제 손끝·관절 위치를 다시 보냅니다.
6. 다음에는 처음 사진 A가 아니라 새 사진 B로 예측합니다.

이것이 폐루프(closed loop)입니다. 행동한 결과를 다시 보고 다음 행동을 정합니다.
행동 앞 6개 값은 로봇 제어기의 설정을 통해 위치·회전 명령으로 해석되므로,
숫자 0.01을 곧바로 이동 거리 1cm라고 읽으면 안 됩니다.
실제 이동 거리는 eef_position 차이를 통해 별도로 확인합니다.

## 이번에 읽을 코드 두 가지

### 1. 시뮬레이터의 한 단계

```python
obs, reward, done, _ = env.step(action.tolist())
```

Python 기본 문법: 함수 결과 네 개를 각각 변수로 받습니다. _는 이번에 사용하지 않을 값입니다.
MuJoCo/LIBERO 기능: step()은 명령을 적용해 시뮬레이션을 진행하고 새 관측값을 돌려줍니다.
obs는 새 사진과 로봇 상태, reward는 보상 신호, done은 종료 신호입니다.
이 프로젝트에서는 obs로 새 사진과 실제 손끝 위치를 저장합니다.

### 2. 새 사진을 반복해서 보기

```python
for step in range(1, args.steps+1):
```

Python 기본 문법: args.steps가 30이면 step에 1~30을 차례로 넣어 들여쓴 코드를 반복합니다.
range의 끝값은 포함하지 않으므로 +1을 합니다.
반복문 안에서는 current의 최신 사진으로 batch를 만들고 select_action()을 다시 호출합니다.
이전 행동 결과가 다음 입력이 되므로 매번 같은 사진으로 예측하는 코드와 다릅니다.

## 해석 범위

공식 연구진이 학습한 LIBERO Spatial 체크포인트를 사용합니다. 직접 학습한 모델이 아닙니다.
seeded reset과 10 no-op 준비 단계를 사용하며 공식 benchmark initial states를 적용하지 않습니다.
카메라 화면은 공식 방향과 맞추기 위해 180도 회전합니다.
정책 입력은 이전 연결 검사와 동일한 PIL bilinear 224x224 리사이즈입니다.
공식 center-crop 전처리와 완전한 episode 평가는 다음 단계입니다.
이 실험의 성공 기준은 유효한 행동 전달과 실제 로봇 위치/관절 변화 및 영상 저장입니다.

## 이번 실행 결과

{
  "steps": 30,
  "frames": 31,
  "max_eef_displacement_m": 0.29491597627612964,
  "simulated_seconds": 1.5000000000000009,
  "control_frequency": 20.0,
  "clipped_steps": 0,
  "initial_eef": [
    -0.20846466056582366,
    2.489065006250401e-12,
    1.17327947572964
  ],
  "final_eef": [
    -0.030534795263646327,
    0.13559602755857758,
    0.9811073825228995
  ],
  "first_raw_action": [
    0.40620845556259155,
    0.42670169472694397,
    -0.0026785731315612793,
    0.05219537764787674,
    0.015084031969308853,
    0.00012604848598130047,
    0.9960784316062927
  ],
  "first_controller_action": [
    0.40620845556259155,
    0.42670169472694397,
    -0.0026785731315612793,
    0.05219537764787674,
    0.015084031969308853,
    0.00012604848598130047,
    -1.0
  ],
  "last_raw_action": [
    -0.002985029947012663,
    0.21567228436470032,
    -0.8385189175605774,
    4.831982005271129e-05,
    -0.05711764469742775,
    0.00012604848598130047,
    0.9960784316062927
  ],
  "done_signal": false,
  "existing_files_unchanged": 6,
  "video_decode_passed": true,
  "closed_loop_frame_chain_passed": true
}
