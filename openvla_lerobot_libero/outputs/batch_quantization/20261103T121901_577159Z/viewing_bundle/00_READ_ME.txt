같은 5 epoch 모델: batch 속도 비교 + 8비트 평가

1. Batch size 비교
   각각 같은 학습 장면 128개. 16개는 준비 용도. 본 모델에 변경을 저장하지 않음.
   batch 4: 10.21 frames/s, peak 14.92 GiB
   batch 8: 11.53 frames/s, peak 15.69 GiB
   처리량 비율: 1.130x
   속도 실험이며 batch 8 최종 성능은 확인하지 않음.

2. 모델 실행 비교
   같은 5 epoch LoRA. 언어모델 주요 선형층만 INT8. 사진 처리/연결층/lm_head는 BF16, LoRA FP32.
   BF16: peak 14.44 GiB, mean action 283.5 ms
   INT8: peak 8.51 GiB, mean action 799.1 ms
   BF16 success 3/5; INT8 success 3/5
   같은 초기 상태 5개, 최대 220 step. 독립적 일반화 검증은 아님.
   예측 시간은 사진 전처리 포함. 시뮬레이터 시간 제외.
   메모리는 PyTorch peak allocated. reserved/GPU 전체 사용량과 다름.
   원래 15.08 GB 파일은 보존. 이번은 로딩 시 양자화, 파일 압축 결과로 해석하지 않음.

3. 보존 확인
   기존 소스/체크포인트 해시 동일, openvla310/lerobot312/eval310 pip freeze 동일.
   bitsandbytes 0.45.5는 이 실험 deps 폴더에만 설치.

REMOTE FOLDER: /home/jovyan/projects/lerobot-openvla/outputs/batch_quantization/20261103T121901_577159Z
CODE: experiment.py
  line 33 batch_benchmark: same 5-epoch start, disposable training speed test.
  line 68 make_inference: load same model as BF16 or quantized language layers + unchanged LoRA.
  line 125 infer: 32-image output/timing/memory check, then INT8 simulator trials.
  line 182 audit: original source/checkpoint hashes and three environments.
  Quantization uses the ORIGINAL saved epoch-5 adapter, not the disposable batch-8 updates.
RE-RUN BATCH ONLY (new output folder): /home/jovyan/anaconda3/envs/lerobot312/bin/python /home/jovyan/projects/lerobot-openvla/outputs/batch_quantization/20261103T121901_577159Z/reproduce.py --batch-only
RE-RUN FULL (new output folder): /home/jovyan/anaconda3/envs/lerobot312/bin/python /home/jovyan/projects/lerobot-openvla/outputs/batch_quantization/20261103T121901_577159Z/reproduce.py
reproduce.py: syntax and --help checked; original experiment stages actually executed. Full fresh-output wrapper has not been rerun.

4. Errors and fixes
Transformers dispatched quantized model through forbidden .to(): enabled Accelerate force_hooks only during this model load.
PEFT read removed memory_efficient_backward metadata: added unused legacy flag False to this process only.
Both handled in experiment.py; package files and original sources unchanged.
ERROR RECORDS: infer_mode_int8.log; infer_mode_int8_retry1.log; successful retry2 in infer_mode_int8_retry2.log.

5. Concepts
Batch size: frames grouped before one optimizer update; affects training throughput and update count.
Quantization: store/calculate main weights with fewer bits; parameter count stays the same.
Here memory saving and prediction latency are different outcomes. Simulation success is a third outcome.