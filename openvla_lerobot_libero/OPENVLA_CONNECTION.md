# OpenVLA → LeRobot: 행동 예측 연결

실행 대상은 원격 Linux 서버 /home/jovyan/projects/lerobot-openvla 입니다.
환경: /home/jovyan/anaconda3/envs/lerobot312 (LeRobot 0.4.0).

## 다시 실행

```bash
conda activate lerobot312
cd /home/jovyan/projects/lerobot-openvla
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u examples/openvla_select_action.py
```

모델 없이 클래스 import와 지연 import만 검사:

```bash
python examples/openvla_select_action.py --check-imports
```

## 파일과 함수

- src/lerobot/policies/openvla/configuration_openvla.py: OpenVLAConfig (기존 설정 유지)
- src/lerobot/policies/openvla/modeling_openvla.py: OpenVLAPolicy.__init__, select_action, reset, predict_action_chunk
- examples/openvla_select_action.py: main (이미지 읽기, 입력 생성, 예측, 출력 검증, 결과 저장)
- outputs/openvla_select_action/<UTC 실행 식별자>/result.json: 행동값, 형태, 유한성, 모델 캐시 경로, 이미지 SHA256, 패키지 버전
- 같은 폴더의 packages.txt: 실행 당시 pip freeze. 완전한 환경 잠금 파일은 아닙니다.
- outputs/openvla_connection_audit/: 기존 지연 import 내용과 변경 전후 Git 상태 기록

## 입력과 출력 계약

입력 이미지: observation.images.image, float32 텐서 (1, 3, 224, 224), RGB 값 0~1.
작업 지시: task 키에 문자열 한 개를 담은 리스트.
출력: 실제 모델이 예측한 행동을 libero_spatial 통계로 역정규화한 float32 텐서 (1, 7).
NaN 또는 infinity가 있거나 형태가 다르면 실패합니다.

이미지는 기존 /home/jovyan/projects/openvla/libero_test.png를 사용합니다.
이미지 방향은 다시 뒤집지 않습니다. RGB 변환 후 PIL bilinear로 224x224 리사이즈합니다.
이 전처리는 연결 검사 전용입니다. 공식 평가의 center crop/resize와 동일하지 않습니다.
기본 작업 지시는 이미지 생성 때 선택한 LIBERO Spatial 첫 작업입니다.
다른 이미지에는 --image와 --task를 함께 지정해 지시와 장면을 맞추세요.

## 검증 범위

이 정책은 LeRobot PreTrainedPolicy의 하위 클래스이며 직접 import/생성합니다.
LeRobot의 공용 factory 및 lerobot-train/lerobot-eval CLI 등록은 이번 범위에 포함하지 않습니다.
forward와 get_optim_params는 NotImplementedError로 학습 실행을 명시적으로 차단합니다.
학습, 시뮬레이터 실행, 성공률 평가, W&B 연결은 수행하지 않습니다.
행동값이 출력됐다는 사실은 LIBERO 작업 성공을 뜻하지 않습니다.
공식 연구진이 LIBERO Spatial로 미세조정한 체크포인트를 사용하며, 직접 학습한 결과가 아닙니다.

## import 오류 이력과 이번 점검

이전 오류 1: diffusers 0.35.2가 peft>=0.17.0을 요구했지만 peft는 0.11.1.
이전 해결: diffusers 0.29.2로 조정. 이번 작업 중 패키지 버전은 변경하지 않습니다.
이전 오류 2: Pi0/Pi0.5의 eager import가 transformers.models.paligemma를 요청.
이전 해결: groot/pi0/pi05 패키지의 모델 import를 __getattr__로 지연.
Pi0.5 processor가 모델을 간접 import하는 경로도 지연 처리되어 있습니다.
이전 오류 3: 설정 파일에 터미널 명령이 섞여 IndentationError 발생.
이전 해결: 설정 파일을 올바른 Python 코드로 교체. 현재 파일의 import를 다시 확인합니다.

이번 실행 스크립트는 sys.modules를 검사해 GR00T/Pi0/Pi0.5 모델 및 Pi0.5 processor가
OpenVLA import 때문에 로딩되지 않았는지 확인합니다.
다른 정책의 실제 실행 호환성 전체를 검증한 것은 아닙니다.

## 읽어볼 코드 두 가지

1. batch = {config.image_key: ..., "task": [args.task]}
   Python 기본 문법인 딕셔너리로 이미지와 작업 지시를 이름별로 묶습니다.
   프로젝트에서는 LeRobot 정책에 전달할 한 번의 입력입니다.
2. torch.isfinite(action).all()
   PyTorch 기능입니다. 각 원소가 유효한 유한 숫자인지 검사한 다음 모두 참인지 확인합니다.
   프로젝트에서는 잘못된 행동 출력을 성공으로 기록하지 않도록 합니다.
