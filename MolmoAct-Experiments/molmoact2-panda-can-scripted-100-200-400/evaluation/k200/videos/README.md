# Panda 캔 옮기기 · 200판 LoRA 모델 평가 영상

Panda 캔 시범 200판으로 학습한 MolmoAct2 모델을 100판 모델과 같은 50곳(평가 시드 30000~30049)에서 평가했습니다.

- 집어 올리기 성공 34/50 (68%)
- 목표 통에 놓기 성공 21/50 (42%)
- 화면 왼쪽은 외부 카메라, 오른쪽은 손목 카메라입니다. 영상 하단에 시드와 로봇 동작 step이 표시됩니다.

[100판과 200판 비교 결과](../RESULTS.md) · [시드별 평가 기록](../summary.json)

| 시드 | 캔 집어 올리기 | 목표 통에 놓기 | 평가 step | 영상 |
| ---: | --- | --- | ---: | --- |
| 30000 | 성공 (step 166) | 성공 (step 381) | 381 | [영상 보기](Panda_Can_PickPlace_seed30000_success_381steps.mp4) |
| 30001 | 성공 (step 306) | 성공 (step 782) | 782 | [영상 보기](Panda_Can_PickPlace_seed30001_success_782steps.mp4) |
| 30002 | 성공 (step 135) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30002_failure_1000steps.mp4) |
| 30003 | 성공 (step 460) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30003_failure_1000steps.mp4) |
| 30004 | 성공 (step 713) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30004_failure_1000steps.mp4) |
| 30005 | 성공 (step 161) | 성공 (step 392) | 392 | [영상 보기](Panda_Can_PickPlace_seed30005_success_392steps.mp4) |
| 30006 | 성공 (step 195) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30006_failure_1000steps.mp4) |
| 30007 | 성공 (step 180) | 성공 (step 401) | 401 | [영상 보기](Panda_Can_PickPlace_seed30007_success_401steps.mp4) |
| 30008 | 성공 (step 173) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30008_failure_1000steps.mp4) |
| 30009 | 성공 (step 121) | 성공 (step 342) | 342 | [영상 보기](Panda_Can_PickPlace_seed30009_success_342steps.mp4) |
| 30010 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30010_failure_1000steps.mp4) |
| 30011 | 성공 (step 161) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30011_failure_1000steps.mp4) |
| 30012 | 성공 (step 215) | 성공 (step 525) | 525 | [영상 보기](Panda_Can_PickPlace_seed30012_success_525steps.mp4) |
| 30013 | 성공 (step 277) | 성공 (step 805) | 805 | [영상 보기](Panda_Can_PickPlace_seed30013_success_805steps.mp4) |
| 30014 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30014_failure_1000steps.mp4) |
| 30015 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30015_failure_1000steps.mp4) |
| 30016 | 성공 (step 272) | 성공 (step 580) | 580 | [영상 보기](Panda_Can_PickPlace_seed30016_success_580steps.mp4) |
| 30017 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30017_failure_1000steps.mp4) |
| 30018 | 성공 (step 141) | 성공 (step 402) | 402 | [영상 보기](Panda_Can_PickPlace_seed30018_success_402steps.mp4) |
| 30019 | 성공 (step 110) | 성공 (step 803) | 803 | [영상 보기](Panda_Can_PickPlace_seed30019_success_803steps.mp4) |
| 30020 | 성공 (step 175) | 성공 (step 426) | 426 | [영상 보기](Panda_Can_PickPlace_seed30020_success_426steps.mp4) |
| 30021 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30021_failure_1000steps.mp4) |
| 30022 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30022_failure_1000steps.mp4) |
| 30023 | 성공 (step 265) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30023_failure_1000steps.mp4) |
| 30024 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30024_failure_1000steps.mp4) |
| 30025 | 성공 (step 161) | 성공 (step 397) | 397 | [영상 보기](Panda_Can_PickPlace_seed30025_success_397steps.mp4) |
| 30026 | 성공 (step 205) | 성공 (step 451) | 451 | [영상 보기](Panda_Can_PickPlace_seed30026_success_451steps.mp4) |
| 30027 | 성공 (step 221) | 성공 (step 426) | 426 | [영상 보기](Panda_Can_PickPlace_seed30027_success_426steps.mp4) |
| 30028 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30028_failure_1000steps.mp4) |
| 30029 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30029_failure_1000steps.mp4) |
| 30030 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30030_failure_1000steps.mp4) |
| 30031 | 성공 (step 165) | 성공 (step 947) | 947 | [영상 보기](Panda_Can_PickPlace_seed30031_success_947steps.mp4) |
| 30032 | 성공 (step 162) | 성공 (step 543) | 543 | [영상 보기](Panda_Can_PickPlace_seed30032_success_543steps.mp4) |
| 30033 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30033_failure_1000steps.mp4) |
| 30034 | 성공 (step 177) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30034_failure_1000steps.mp4) |
| 30035 | 성공 (step 877) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30035_failure_1000steps.mp4) |
| 30036 | 성공 (step 727) | 성공 (step 937) | 937 | [영상 보기](Panda_Can_PickPlace_seed30036_success_937steps.mp4) |
| 30037 | 성공 (step 370) | 성공 (step 705) | 705 | [영상 보기](Panda_Can_PickPlace_seed30037_success_705steps.mp4) |
| 30038 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30038_failure_1000steps.mp4) |
| 30039 | 성공 (step 156) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30039_failure_1000steps.mp4) |
| 30040 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30040_failure_1000steps.mp4) |
| 30041 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30041_failure_1000steps.mp4) |
| 30042 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30042_failure_1000steps.mp4) |
| 30043 | 성공 (step 293) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30043_failure_1000steps.mp4) |
| 30044 | 성공 (step 285) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30044_failure_1000steps.mp4) |
| 30045 | 성공 (step 231) | 성공 (step 733) | 733 | [영상 보기](Panda_Can_PickPlace_seed30045_success_733steps.mp4) |
| 30046 | 실패 | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30046_failure_1000steps.mp4) |
| 30047 | 성공 (step 177) | 성공 (step 592) | 592 | [영상 보기](Panda_Can_PickPlace_seed30047_success_592steps.mp4) |
| 30048 | 성공 (step 632) | 실패 | 1000 | [영상 보기](Panda_Can_PickPlace_seed30048_failure_1000steps.mp4) |
| 30049 | 성공 (step 145) | 성공 (step 485) | 485 | [영상 보기](Panda_Can_PickPlace_seed30049_success_485steps.mp4) |
