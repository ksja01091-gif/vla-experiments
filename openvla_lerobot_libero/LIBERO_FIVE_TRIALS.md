# 같은 작업을 시작 장면 5개에서 시켜보기

이번 실험은 "검은 그릇을 접시 위에 놓아라"라는 같은 문제를
공식 시작 상태 0,1,2,3,4번에서 새로 한 번씩 실행합니다.
이전 한 번의 성공 기록은 이번 5회 통계에 재사용하지 않습니다.

## 무엇을 확인하는가

한 번 성공한 모델이 출발 장면이 조금 바뀌어도 작업을 수행하는지 봅니다.
모델은 바꾸지 않고 사진 처리, 준비 동작 10회, 최대 행동 220회도 동일하게 유지합니다.
이번에는 학습하거나 모델을 개선하지 않습니다. 같은 모델을 시험하는 단계입니다.

## 결과 읽는 법

- success: 제한 안에 환경의 목표 조건을 만족했습니다.
- task_failure: 프로그램은 끝까지 정상 실행됐지만 220회 안에 목표 조건을 만족하지 못했습니다.
- technical_error: 프로그램이나 환경 문제로 유효한 시도를 완료하지 못했습니다.
- policy_steps: 로봇에게 행동을 전달한 횟수입니다. 학습 횟수가 아닙니다.

5회가 모두 유효하게 완료되면 성공 횟수를 5로 나눕니다.
예를 들어 5번 중 3번이면 이번 다섯 장면에서 관측한 성공률은 60%입니다.
첫 작업의 고정된 시작 장면 5개에 대한 결과이므로 전체 LIBERO 성능으로 해석하지 않습니다.
기술 오류가 있으면 오류를 별도로 표시하고 이 5회 성공률을 확정하지 않습니다.

## 실행한 코드에서 볼 부분 두 개

```python
INITIAL_STATES = [0, 1, 2, 3, 4]
```

Python 기본 문법인 리스트입니다. 시험할 시작 장면 번호를 미리 적어둔 것입니다.
결과를 보고 성공한 장면으로 바꾸지 않고 이 다섯 장면을 사용합니다.

```python
for state in INITIAL_STATES:
```

리스트의 번호를 하나씩 꺼내 state에 넣고 같은 시험 코드를 실행하는 반복문입니다.
로봇이 한 번 움직이는 step 반복문과는 다릅니다.
바깥쪽의 이 반복문은 시험을 다섯 번, 안쪽 정책 반복문은 각 시험에서 최대 220번의 행동을 진행합니다.

## 파일

examples/openvla_libero_five_trials.py: 다섯 번 실행하고 결과를 모읍니다.
기존 openvla_libero_single_task.py, libero_eval_worker.py, openvla_eval_preprocessing.py는 수정하지 않습니다.

outputs/libero_five_trials/<UTC 실행 식별자>/
- manifest.json: 실행 전에 정한 장면과 공통 조건.
- summary.json: 각 시도의 결과와 전체 집계.
- results.csv: 표 형식의 결과.
- results_table.png: 결과 표 이미지.
- start_and_finish.png: 다섯 시작 장면과 각 마지막 장면 비교.
- state_0.log ~ state_4.log: 개별 시도의 로그.
- 각 환경의 packages_before/after.txt: 기존 패키지 목록 보존 확인.

개별 시도의 사진·영상·행동 기록은 summary.json의 result_path가 가리키는
outputs/libero_single_task/<각 실행 식별자>/ 폴더에 있습니다.

## 다시 실행

```bash
conda activate lerobot312
cd /home/jovyan/projects/lerobot-openvla
python -u examples/openvla_libero_five_trials.py
```

다시 실행하면 기존 결과를 덮어쓰지 않고 새로운 폴더에서 다섯 번을 새로 수행합니다.

## 이번 실행 결과

| 시작 장면 | 결과 | 행동 횟수 |
|---|---|---|
| 0 | 성공 | 85 |
| 1 | 작업 실패 | 220 |
| 2 | 성공 | 79 |
| 3 | 성공 | 102 |
| 4 | 성공 | 74 |

이번 다섯 장면에서 4/5 성공, 관측 성공률 80%. 기술 오류 0회.

실제 결과 폴더: /home/jovyan/projects/lerobot-openvla/outputs/libero_five_trials/20261101T104237_824617Z
