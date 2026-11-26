# Panda 캔 옮기기 · 100판 LoRA 모델 평가 영상

Panda 캔 시범 100판으로 학습한 MolmoAct2 모델을 시작 배치 50곳(평가 시드 30000~30049)에서 평가했습니다.

- 집어 올리기 성공 26/50 (52%)
- 목표 통에 놓기 성공 4/50 (8%)
- 화면 왼쪽은 외부 카메라, 오른쪽은 손목 카메라입니다. 영상 하단에 시드와 로봇 동작 step이 표시됩니다.

[결과표](../RESULTS.md) · [시드별 평가 기록](../summary.json)

| 시드 | 캔 집어 올리기 | 목표 통에 놓기 | 평가 step | 영상 |
| ---: | --- | --- | ---: | --- |
| 30000 | 성공 (step 91) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30000_failure_1000steps.mp4) |
| 30001 | 성공 (step 427) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30001_failure_1000steps.mp4) |
| 30002 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30002_failure_1000steps.mp4) |
| 30003 | 성공 (step 793) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30003_failure_1000steps.mp4) |
| 30004 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30004_failure_1000steps.mp4) |
| 30005 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30005_failure_1000steps.mp4) |
| 30006 | 성공 (step 583) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30006_failure_1000steps.mp4) |
| 30007 | 성공 (step 239) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30007_failure_1000steps.mp4) |
| 30008 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30008_failure_1000steps.mp4) |
| 30009 | 성공 (step 474) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30009_failure_1000steps.mp4) |
| 30010 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30010_failure_1000steps.mp4) |
| 30011 | 성공 (step 311) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30011_failure_1000steps.mp4) |
| 30012 | 성공 (step 357) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30012_failure_1000steps.mp4) |
| 30013 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30013_failure_1000steps.mp4) |
| 30014 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30014_failure_1000steps.mp4) |
| 30015 | 성공 (step 78) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30015_failure_1000steps.mp4) |
| 30016 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30016_failure_1000steps.mp4) |
| 30017 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30017_failure_1000steps.mp4) |
| 30018 | 성공 (step 448) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30018_failure_1000steps.mp4) |
| 30019 | 성공 (step 122) | 성공 (step 665) | 665 | [영상 보기](Panda_Can_PickPlace_seed30019_success_665steps.mp4) |
| 30020 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30020_failure_1000steps.mp4) |
| 30021 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30021_failure_1000steps.mp4) |
| 30022 | 성공 (step 302) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30022_failure_1000steps.mp4) |
| 30023 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30023_failure_1000steps.mp4) |
| 30024 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30024_failure_1000steps.mp4) |
| 30025 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30025_failure_1000steps.mp4) |
| 30026 | 성공 (step 153) | 성공 (step 477) | 477 | [영상 보기](Panda_Can_PickPlace_seed30026_success_477steps.mp4) |
| 30027 | 성공 (step 293) | 성공 (step 859) | 859 | [영상 보기](Panda_Can_PickPlace_seed30027_success_859steps.mp4) |
| 30028 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30028_failure_1000steps.mp4) |
| 30029 | 성공 (step 497) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30029_failure_1000steps.mp4) |
| 30030 | 성공 (step 854) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30030_failure_1000steps.mp4) |
| 30031 | 성공 (step 120) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30031_failure_1000steps.mp4) |
| 30032 | 성공 (step 144) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30032_failure_1000steps.mp4) |
| 30033 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30033_failure_1000steps.mp4) |
| 30034 | 성공 (step 147) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30034_failure_1000steps.mp4) |
| 30035 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30035_failure_1000steps.mp4) |
| 30036 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30036_failure_1000steps.mp4) |
| 30037 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30037_failure_1000steps.mp4) |
| 30038 | 성공 (step 107) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30038_failure_1000steps.mp4) |
| 30039 | 성공 (step 852) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30039_failure_1000steps.mp4) |
| 30040 | 성공 (step 254) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30040_failure_1000steps.mp4) |
| 30041 | 성공 (step 613) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30041_failure_1000steps.mp4) |
| 30042 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30042_failure_1000steps.mp4) |
| 30043 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30043_failure_1000steps.mp4) |
| 30044 | 성공 (step 197) | 성공 (step 905) | 905 | [영상 보기](Panda_Can_PickPlace_seed30044_success_905steps.mp4) |
| 30045 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30045_failure_1000steps.mp4) |
| 30046 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30046_failure_1000steps.mp4) |
| 30047 | 성공 (step 156) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30047_failure_1000steps.mp4) |
| 30048 | 성공 (step 406) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30048_failure_1000steps.mp4) |
| 30049 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30049_failure_1000steps.mp4) |
