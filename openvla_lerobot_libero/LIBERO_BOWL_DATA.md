# 그릇 옮기기: 학습 데이터 준비 기록

이번에는 로봇을 가르치기 전, 사용할 교재를 준비했습니다. 학습은 실행하지 않았습니다.

## 무엇을 골랐나요?

작업 지시: pick up the black bowl between the plate and the ramekin and place it on the plate

뜻: 접시와 라메킨(작은 그릇) 사이의 검은 그릇을 집어서 접시 위에 놓기.

| 구분 | 시범 전체 개수 | 사진·행동 쌍 개수 | 쓰임 |
|---|---:|---:|---|
| train | 20 | 1,928 | 앞으로 모델이 따라 연습할 교재 |
| validation | 5 | 462 | 앞으로 학습에 넣지 않고 예측을 확인할 문제 |

시범 하나는 시작부터 끝까지 이어진 움직임입니다. 사진 한 장이 아닙니다.
예를 들어 시범 하나가 100단계라면, 사진과 행동의 짝도 100개입니다.
이번 원본에는 Spatial 전체 432개 시범, 이 작업에는 45개 시범이 있었습니다.
중복 여부를 검사한 뒤 이 작업의 시범 25개만 선택했습니다. 나머지는 이번 학습용으로 내보내지 않았습니다.

## 데이터의 출처와 버전

- Hugging Face: openvla/modified_libero_rlds
- 고정 revision: 6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551
- 하위 데이터: libero_spatial_no_noops/1.0.0
- 원본 18개 파일: 약 1.91 GB. 원본 전체는 보존합니다.
- LeRobot 저장 형식: v3.0, 손실 없는 이미지 형태(use_videos=False).
- 선택 방법: 시범 내용의 SHA256 ID를 정렬하고 seed=7로 섞은 뒤 20/5로 나눴습니다.
- 선택에 모델의 성공/실패 결과를 사용하지 않았습니다.
- 같은 시범 ID가 양쪽에 들어가지 않습니다. 다만 기존 모델이 과거에 이 자료를 보았는지, 시뮬레이터 평가의 초기 상태와 겹치는지는 이 분할로 증명할 수 없습니다.

## 코드에서 두 군데만 읽어보세요

### 1. 시범을 통째로 나누기 — Python 문법

실제 추출 파일의 코드:

```python
train, validation = selected[:20], selected[20:25]
```

selected는 고른 시범 25개를 담은 목록입니다.
[:20]은 앞 20개, [20:25]는 그다음 5개를 꺼내는 목록 자르기 문법입니다.
이 프로젝트에서는 한 시범의 앞뒤 장면이 연습용과 확인용으로 흩어지는 것을 막습니다.

### 2. 사진과 행동을 짝지어 저장하기 — LeRobot 기능

실제 변환 파일의 핵심 호출은 다음과 같습니다. 다른 상태 필드도 함께 전달합니다.

```python
dataset.add_frame({
    IMAGE_KEY: demo["images"][step],
    "action": demo["actions"][step],
    "observation.state": demo["states"][step],
    "observation.joint_state": demo["joints"][step],
    "source_step_index": np.array([step], dtype=np.int64),
    "task": task,
})
```

{키: 값}은 Python의 사전 문법입니다. 예를 들어 action이라는 이름에 행동 숫자 7개를 붙입니다.
add_frame은 LeRobot이 제공하는 저장 함수입니다. 같은 step의 사진과 행동을 함께 저장합니다.
바깥 반복문은 시범을 하나씩, 안쪽 반복문은 그 시범의 장면을 순서대로 읽습니다.
save_episode()는 시범 하나의 끝을, finalize()는 전체 파일 저장의 끝을 처리합니다.

## 사진과 행동을 어떻게 처리했나요?

원본 사진 256×256의 픽셀은 바꾸지 않았습니다. 평가용 사진의 회전이나 자르기를 여기서 다시 적용하지 않았습니다.
현재 모델이 쓰는 외부 카메라 사진만 내보냈습니다. 손목 카메라 사진은 원본 RLDS에 남아 있습니다.
행동 숫자 앞 6개는 보존하고, 마지막 집게 값은 공식 OpenVLA libero_dataset_transform 함수 그대로 변환했습니다.
즉 마지막 값은 1-clip(raw,0,1)입니다. 시뮬레이터에 보내는 집게 명령과 학습 데이터의 표기를 혼동하지 않아야 합니다.
통계 파일 action_statistics_train_only.json은 연습용 20개만 사용했습니다. 이 단계에서는 행동 토큰화나 정규화 학습 처리를 하지 않았습니다.
20 Hz는 이미 멈춤 구간이 제거된 장면의 인덱스용 설정입니다. 원래 실제 시간 간격을 복구했다고 해석하지 않습니다.

## 다시 실행하기

원격 Linux에서 아래를 실행합니다. 이미 완성된 데이터가 있으면 다시 읽어 대조합니다. 학습하지 않습니다.

```bash
conda activate lerobot312
python /home/jovyan/projects/lerobot-openvla/examples/prepare_libero_bowl_data.py
```

다른 폴더에 처음부터 다시 생성하려면 같은 명령에 --root /home/jovyan/projects/datasets/libero_bowl_25/새로운폴더이름 을 붙입니다.
중간에 멈춘 폴더를 발견하면 삭제하거나 덮어쓰지 않고 중단합니다. 새로운 폴더 이름으로 재실행하세요.

## 파일 위치

프로젝트: /home/jovyan/projects/lerobot-openvla

- examples/extract_libero_bowl_episodes.py: main, official_action_transform. 원본에서 대상 시범을 찾고 통째로 분할.
- examples/prepare_libero_bowl_data.py: export_split, verify_split, make_preview. LeRobot 저장·재읽기 검증·시범 미리보기.
- 데이터 루트: /home/jovyan/projects/datasets/libero_bowl_25/20261101T120613_454595Z
- train/ 및 validation/: 실제 LeRobot 데이터.
- split_manifest.json: 어떤 원본 시범을 어디에 넣었는지 기록.
- verification_report.json: 실제 재읽기 검증 결과. 이 파일이 있어야 완료 상태입니다.
- preview/training_demo_contact_sheet.png 및 training_demo.gif: 시범 영상. 모델이 실행한 영상이 아닙니다.
- preview/example.json: 사진 한 장과 대응하는 실제 행동 숫자.
- outputs/libero_data_prep/20261101T120613_454595Z/: 다운로드 출처·체크섬·실행 로그·환경 보존 검사.

## 환경과 오류 처리

기존 openvla310, lerobot312, openvla_eval310 환경은 보존했습니다. 원본 RLDS 읽기용 openvla_data310 가상환경만 별도로 만들었습니다.
TensorFlow CPU 2.15.1, TFDS 4.9.3, TensorFlow Metadata 1.15.0을 사용하며 설치 전 의존성 검토와 설치 후 pip check를 수행했습니다.
처음 원본 사이트 접근 중 Temporary failure in name resolution 오류가 발생했습니다. 이후 호스트 이름 조회와 접속이 정상으로 돌아와 재시도했습니다. 시스템 DNS나 인증 설정은 바꾸지 않았습니다.

## 다음에 할 일

OpenVLA 정책의 학습용 forward는 아직 구현되어 있지 않습니다. 데이터 저장 성공만으로 학습 가능하다고 판단하지 않습니다.
다음 작업은 학습을 시작할 체크포인트를 정하고, 이 데이터에 맞는 행동 정규화·토큰화와 LoRA 학습 경로를 연결하는 것입니다.
학습 전후에 같은 조건으로 시뮬레이터 성공률을 비교해야 개선 여부를 알 수 있습니다.
현재 Spatial 미세조정 모델의 기존 4/5 결과는 참고 결과이며, 새 학습의 사전학습 base 모델 기준 점수로 사용하지 않습니다.

저장 속도 보완: 시범 압축파일은 시범마다 한 번만 읽도록 개선했습니다. 첫 시도의 중간 파일은 train.partial_before_npz_cache 폴더에 보존했습니다.
