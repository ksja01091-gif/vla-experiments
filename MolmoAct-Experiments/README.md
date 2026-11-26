# MolmoAct Experiments

| 실험 폴더 | 한 일 | 먼저 볼 파일 |
| --- | --- | --- |
| [molmoact-7b-d-0812-panda-ur5e](molmoact-7b-d-0812-panda-ur5e/README.md) | MolmoAct 7B를 LoRA로 학습하고 Panda와 UR5e에서 집어서 놓기를 시험했습니다. | [Panda·UR5e 평가 영상](molmoact-7b-d-0812-panda-ur5e/videos/README.md) |
| [molmoact2-libero-panda-ur5e-full-data](molmoact2-libero-panda-ur5e-full-data/README.md) | 공개 MimicGen 시범을 변환해 MolmoAct2로 네 물건 옮기기를 학습했습니다. | [시범 변환 검증](molmoact2-libero-panda-ur5e-full-data/DATASET_VERIFICATION.md), [Panda 50곳 평가 결과](molmoact2-libero-panda-ur5e-full-data/panda_mimicgen400_lora_r32_seed777_eval50/RESULTS.md) |
| [molmoact2-panda-can-scripted-100-200-400](molmoact2-panda-can-scripted-100-200-400/README.md) | 직접 만든 Panda 캔 시범을 100·200·400판씩 사용해 집기와 놓기 성공률을 비교합니다. | [400판 변환 검증](molmoact2-panda-can-scripted-100-200-400/converted/k400/conversion_verified.json), [공통 평가 위치 50곳](molmoact2-panda-can-scripted-100-200-400/raw/evaluation_layouts.json) |

- `README.md`는 실험 목적과 실행 방법을 설명합니다.
- `RESULTS.md`는 읽기 쉬운 결과표이고, `summary.json`이나 `results_summary.json`은 평가 수치를 저장합니다.
- `conversion_verified.json`은 데이터 변환을 확인한 기록입니다.
- `videos/`와 `media/`에는 시뮬레이션 영상이 있습니다.
