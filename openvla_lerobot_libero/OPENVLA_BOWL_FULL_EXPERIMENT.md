# OpenVLA 그릇 옮기기 본 실험

이번 실험은 공개 LIBERO 데이터에서 우리가 선택·전처리한 실험용 분할을 사용합니다. 시범 20개는 처음부터 끝까지의 시범 20개이며 사진 20장이 아닙니다.

## 실험 목표

기본 openvla/openvla-7b를 추가 학습한 뒤, 같은 LIBERO 초기 상태에서 행동 성공률이 바뀌는지 확인합니다.
공개 openvla-7b-finetuned-libero-spatial은 별도 참고 기준으로 다시 실행합니다.

## 실제 학습 흐름

LeRobotDataset이 사진·지시·시범 행동을 읽습니다. 기존 prepare_training_batch가 사진을 HF processor에, 지시를 PurePromptBuilder와 tokenizer에 전달합니다.
시범 행동은 학습용 시범 20개에서 구한 q01/q99로 정규화하고 공식 ActionTokenizer로 행동 토큰 7개로 바꿉니다. 집게는 별도로 유지하며, 항상 같은 값인 성분은 0으로 둡니다.
정답은 행동 토큰 7개와 끝 표시 1개입니다. 지시문 부분의 labels는 -100으로 채점에서 제외합니다.
forward의 output.loss가 토큰 분류의 cross entropy를 계산합니다. backward는 기울기를 계산하고 AdamW.step이 LoRA 파라미터를 수정합니다.

학습 코드는 새 FullExperimentOpenVLAPolicy에 추가했고 기존 inference·20회 연결 시험 코드와 지연 import 코드는 보존합니다.
LoRA는 언어 모델 q_proj, v_proj만 대상으로 합니다. 기본 가중치는 bf16, LoRA 학습용 가중치는 float32로 유지합니다. LoRA rank 8, alpha 8, dropout 0, 학습률 0.0001, batch size 4, 1 epoch입니다.
전체 1,928장면을 한 번씩 사용해 482회 업데이트합니다. 중간 241회와 마지막 482회에 checkpoint를 저장합니다.

## 읽어볼 실제 코드 두 곳

### 전체 장면을 순서대로 묶어서 처리하기 — Python

```python
for offset in range(0,len(order),args.batch_size):
    indices=order[offset:offset+args.batch_size]
```

range는 위치를 batch size인 4칸씩 옮깁니다. 목록 자르기는 그 위치부터 장면 4개를 꺼냅니다. order에는 1,928장면 번호가 섞여 있고 모두 한 번씩 들어 있습니다.
따라서 예전 중간 장면 20장만 사용하는 연결 시험과 다르게 모든 장면을 학습합니다.

### 검증에서는 기울기를 계산하지 않기 — PyTorch

```python
@torch.no_grad()
def evaluate(policy, data, batch_size, destination, optimizer_step):
```

@torch.no_grad()는 바로 아래 함수에서 기울기 계산을 끄는 PyTorch 기능입니다. evaluate에서는 optimizer.step도 호출하지 않습니다.
이 프로젝트에서는 검증용 462장면으로 채점만 하고 가중치를 수정하지 않도록 사용합니다.

## 기록하는 지표와 뜻

- training loss: 각 학습 배치에서 행동 토큰과 끝 표시를 예측한 오차.
- validation loss: 검증용 462장면 전체에서 평균한 같은 오차.
- action token accuracy: 행동 토큰 7개 중 정확히 맞춘 비율. 로봇 성공률이 아닙니다.
- normalized action L1: 예측 토큰을 행동으로 해석한 뒤 정규화된 시범 행동과 비교한 평균 절댓값 차이. 실제 미터 단위가 아닙니다.
- predicted_action_token_fraction: 예측한 토큰이 행동 토큰 범위 안에 들어온 비율.
- GPU allocated/reserved/peak 메모리, 배치 시간, 경과 시간, 장면 번호, checkpoint 위치.

이 지표들은 teacher forcing으로 계산합니다. 앞 행동 토큰을 정답으로 주고 다음 토큰을 예측하는 방식이므로 자유롭게 행동을 생성한 성공률과 구분해야 합니다.
validation loss가 가장 낮은 학습 checkpoint를 비교에 사용합니다. 원본 기본 모델은 checkpoint 선택 후보에 넣지 않습니다. 최종 checkpoint도 별도로 보존합니다.

## 시뮬레이터 비교 조건

세 모델에 초기 상태 0,1,2,3,4를 동일하게 적용합니다. 환경 seed 0, 전역 seed 7, 준비 동작 10회, 최대 220회 정책 행동입니다.
기존 libero_eval_worker.py와 공식 이미지 전처리 함수를 그대로 재사용합니다. 매 초기 상태의 파일 checksum과 시작 사진 checksum을 모델 간 대조합니다.
기본 모델과 LoRA 모델은 모두 학습용 20개 시범에서 계산한 같은 행동 정규화 통계를 사용합니다. 기본 모델의 가중치는 수정하지 않습니다.
공개 Spatial 미세조정 모델은 원래 libero_spatial 통계를 사용하고 참고 기준으로 표시합니다.

성공은 LIBERO의 task success 판정으로 측정합니다. 기술적 오류는 정책의 과제 실패와 구분합니다.
실패한 경우 horizon_exhausted와 영상·마지막 사진을 기록합니다. 이 기록만으로 세부 물리 원인을 확정하지 않고 관찰한 내용과 추정을 구분합니다.
5개 초기 상태의 작은 비교이며 전체 LIBERO 벤치마크 결과나 새로운 초기 상태 일반화 증거라고 해석하지 않습니다. 시범 분할과 평가 초기 상태가 겹치지 않는지도 보장할 수 없습니다.

사진 학습 전처리는 기존 HF processor를 유지했습니다. 공식 RLDS 전체 학습 레시피나 논문의 수치를 그대로 재현하는 실험은 아닙니다.

## 실행 명령

원격 Linux에서 새로운 실험을 처음부터 실행하면 새 출력 폴더가 만들어집니다. 시작 시 속도·메모리를 측정하고 설정을 출력합니다.

```bash
conda activate lerobot312
python /home/jovyan/projects/lerobot-openvla/examples/run_openvla_bowl_full.py
```

이번 결과 폴더의 checkpoint만 새 프로세스에서 다시 확인하려면:

```bash
python /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl_experiment.py check --run /home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z
```

개별 단계는 openvla_bowl_experiment.py의 calibrate, train, check, simulate, audit입니다. 같은 결과 폴더에 시뮬레이션을 덮어써 실행하지 않습니다.
run_openvla_bowl_full.py --run 기존폴더 는 완료한 학습·평가를 재사용하고 남은 단계를 진행합니다. 불완전하게 중단된 학습을 자동으로 resume하는 기능은 없습니다. trainer_state.pt는 복구 연구를 위해 저장합니다.

## 핵심 파일

- /home/jovyan/projects/lerobot-openvla/src/lerobot/policies/openvla/full_experiment_openvla.py: FullExperimentOpenVLAPolicy, collate_samples, forward, save_adapter, load_saved_adapter.
- /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl_experiment.py: calibrate, train_full, evaluate, check_checkpoint, simulate_trial, simulate, audit_after.
- /home/jovyan/projects/lerobot-openvla/examples/run_openvla_bowl_full.py: 전체 실험 순차 실행.
- /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl_report.py: CSV·PNG·한국어 결과 보고서.
- /home/jovyan/projects/lerobot-openvla/examples/openvla_bowl_wandb_summary.py: 최종 비교 지표의 W&B offline 기록.
- /home/jovyan/projects/lerobot-openvla/outputs/full_bowl_experiment/20261102T091448_811738Z: 이번 실험 결과 폴더.

## 로컬과 W&B

이번 환경에서는 WANDB_API_KEY 환경변수와 기본 netrc 인증 정보가 발견되지 않아 offline으로 기록합니다. 업로드한 것은 없습니다.
W&B offline 기록에는 학습·검증 지표, 메모리, 시간, 설정, 모델별 성공률·평균 step을 남깁니다.
로컬에는 원본 데이터, LoRA 가중치, optimizer 상태, 소스 기록, 단계별 JSONL/CSV, 모든 영상·사진을 보존합니다. 모델·원본 데이터·영상은 자동 업로드하지 않습니다.
로그인 후 wandb sync 명령으로 offline 기록을 올릴 수 있습니다. API key는 대화창에 붙이지 말고 원격 터미널의 wandb login에서 직접 입력하세요.

## 발생한 문제

초기 nvidia-smi 조회가 응답하지 않아 그 조회만 중단했습니다. 독립 재조회는 정상으로 돌아왔고 실제 CUDA 속도·메모리 측정이 통과한 뒤 학습을 시작했습니다. 원인은 일시적 지연인지 드라이버 문제인지 확정하지 못했습니다.
환경 기록 보조 스크립트의 줄바꿈 SyntaxError는 해당 보조 파일만 수정하고 문법 검사·세 환경 패키지 목록 기록을 다시 실행해 해결했습니다. 기존 프로젝트 코드는 수정하지 않았습니다.
원래 설치된 Pydantic 및 PEFT의 config 확인 경고는 버전 변경 없이 처리했고, 실제 가중치 저장·새 프로세스 재로딩과 동일 입력 loss 대조로 검증합니다.

실제 수치는 결과 폴더의 RESULTS_KO.md와 experiment_summary.json을 확인하세요. pipeline_complete.json이 생겨야 전체 파이프라인 완료입니다.
