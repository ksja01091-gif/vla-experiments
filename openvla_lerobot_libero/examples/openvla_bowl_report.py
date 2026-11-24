"""Export full-bowl experiment JSON/CSV/PNG and a short Korean report.
Run using existing openvla310 matplotlib, after GPU experiments finish.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def csv_write(path,records):
    fields=list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(records)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True);args=parser.parse_args();run=args.run
    train=read(run/"training_result.json");config=read(run/"training_config.json");updates=rows(run/"training_metrics.jsonl")
    validation=[train["baseline_validation"],*rows(run/"validation_history.jsonl")]
    csv_write(run/"training_metrics.csv",[{k:v for k,v in row.items() if k!="indices"} for row in updates])
    csv_write(run/"validation_metrics.csv",validation)
    cohorts={tag:read(run/f"simulation_{tag}.json") for tag in ["base","lora","reference"]}
    cohort_rows=[{k:v for k,v in report.items() if k!="trials"} for report in cohorts.values()]
    csv_write(run/"simulation_summary.csv",cohort_rows)
    csv_write(run/"simulation_trials.csv",[r for report in cohorts.values() for r in report["trials"]])
    motion_records=[read(p) for p in (run/"simulation").glob("*/state_*/replay_diagnostics.json")]
    if motion_records:csv_write(run/"motion_diagnostics.csv",motion_records)
    paired=True
    for state in config.get("simulation_states",[0,1,2,3,4]):
        trials=[next(r for r in cohort["trials"] if r["init_state_index"]==state) for cohort in cohorts.values()]
        if all(r["trial_executed"] for r in trials):
            paired &= len({r["selected_init_state_sha256"] for r in trials})==1
            paired &= len({r["initial_image_sha256"] for r in trials})==1
        else:paired=False
    coverage=read(run/"training_coverage.json")
    assert len(updates)==train["optimizer_steps"]
    assert all(sorted(row["indices"])==list(range(1928)) for row in coverage)
    assert all(row["frames"]==462 for row in validation)
    reloaded=read(run/"checkpoint_reload_check.json");preservation=read(run/"preservation_report.json")
    all_media=[]
    for tag,cohort in cohorts.items():
        for trial in cohort["trials"]:
            if trial["trial_executed"]:
                for key in ["video","overview"]:
                    path=Path(trial[key]);assert path.is_file() and path.stat().st_size>0
                all_media.append(trial["video"])
    completion_evidence={
        "training":bool(train.get("training_completed")) and train["frames_seen"]==1928,
        "reload":bool(reloaded.get("fresh_python_process_reload")) and bool(reloaded.get("all_actions_finite")),
        "simulator":all(c["attempts"]==len(c["trials"])==5 and all(t["trial_executed"] for t in c["trials"]) for c in cohorts.values()),
        "preservation":bool(preservation["existing_sources_unchanged"]) and all(e["packages_unchanged"] for e in preservation["environments"].values())
    }
    summary={"run":str(run),"completion_evidence":completion_evidence,"simulator_evaluation_completed":completion_evidence["simulator"],"training":train,"paired_initial_states_and_initial_images":bool(paired),
             "cohorts":{k:{n:v for n,v in cohort.items() if n!="trials"} for k,cohort in cohorts.items()},
             "reload_check":reloaded,"preservation":preservation,"media_files":all_media,"motion_diagnostics":motion_records,
             "full_training_coverage_verified":True,"full_validation_coverage_verified":True,
             "optimizer_iteration_seconds_sum":sum(row["iteration_seconds"] for row in updates),
             "selected_validation_loss_change":train["selected_validation"]["loss"]-train["baseline_validation"]["loss"],
             "base_vs_lora_success_rate_change":cohorts["lora"]["success_rate"]-cohorts["base"]["success_rate"],
             "wandb":read(run/"wandb_status.json")}
    (run/"experiment_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    x=np.asarray([r["optimizer_step"] for r in updates]);y=np.asarray([r["loss"] for r in updates])
    axes[0,0].plot(x,y,alpha=.25,color="#2a7f85",label="Training batch loss")
    if len(y)>=25:axes[0,0].plot(x[24:],np.convolve(y,np.ones(25)/25,mode="valid"),color="#2a7f85",label="25-batch mean")
    axes[0,0].set(title="All 1,928 training frames / 1 epoch",xlabel="Optimizer update",ylabel="Action-token loss");axes[0,0].legend(fontsize=8)
    axes[0,1].plot([r["optimizer_step"] for r in validation],[r["loss"] for r in validation],marker="o",color="#9e4f37")
    for r in validation:axes[0,1].annotate(f"{r['loss']:.3f}",(r["optimizer_step"],r["loss"]),xytext=(0,8),textcoords="offset points",ha="center")
    axes[0,1].set(title="Full validation: same 462 frames",xlabel="Optimizer update",ylabel="Mean token loss")
    names=["Base OpenVLA","LoRA trained","Public Spatial ref."]
    rates=[cohorts[tag]["success_rate"]*100 for tag in cohorts]
    bars=axes[1,0].bar(names,rates,color=["#9faab4","#2a7f85","#c6a15a"])
    for bar,tag in zip(bars,cohorts):
        c=cohorts[tag];axes[1,0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,f"{c['successes']}/{c['attempts']}",ha="center")
    axes[1,0].set(title="LIBERO: same task and 5 initial states",ylabel="Success rate (%)",ylim=(0,115))
    steps=[cohorts[tag]["mean_policy_steps"] or 0 for tag in cohorts]
    bars=axes[1,1].bar(names,steps,color=["#9faab4","#2a7f85","#c6a15a"])
    axes[1,1].bar_label(bars,fmt="%.1f")
    axes[1,1].set(title="Mean episode length (max 220)",ylabel="Policy actions",ylim=(0,250))
    for ax in axes.flat:ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("OpenVLA bowl baseline: full-data training and paired simulator evaluation",fontweight="bold")
    fig.savefig(run/"experiment_summary.png",dpi=160)
    sheet=Image.new("RGB",(3*256,5*288),"white");draw=ImageDraw.Draw(sheet)
    for col,(tag,cohort) in enumerate(cohorts.items()):
        for row,trial in enumerate(cohort["trials"]):
            x0,y0=col*256,row*288
            label=f"{tag} / state {trial['init_state_index']} / {'SUCCESS' if trial['success'] else trial['termination_reason']}"
            draw.text((x0+4,y0+4),label,fill="black")
            frames=sorted((run/"simulation"/tag/f"state_{trial['init_state_index']:02d}"/"frames").glob("*.png"))
            if frames:
                with Image.open(frames[-1]) as image:sheet.paste(image.convert("RGB"),(x0,y0+28))
    sheet.save(run/"rollout_endpoints.png")
    report=["# 그릇 옮기기 본 실험 결과","", "공개 LIBERO 데이터에서 선택·전처리한 실험용 분할로 실행했습니다.",
            "전체 시범 20개의 1,928장면을 1 epoch 학습하고, 다른 시범 5개의 462장면 전체를 평가했습니다.","",
            f"- 학습률: {config['learning_rate']}, batch size: {config['batch_size']}, LoRA rank: {config['lora_rank']}, 업데이트: {train['optimizer_steps']}회.",
            f"- 학습 루프 시간(중간·최종 검증 및 저장 포함): {train['training_loop_wall_seconds']/60:.2f}분.",
            f"- 최대 GPU allocated 메모리: {train['gpu_peak_allocated_gib']:.2f} GiB.",
            f"- 전체 validation loss: {train['baseline_validation']['loss']:.4f} → 최종 {train['final_validation']['loss']:.4f}.",
            f"- 선택한 checkpoint: {train['selected_checkpoint']} (validation loss {train['selected_validation']['loss']:.4f}).","",
            "| 모델 | 성공/시도 | 성공률 | 평균 step | 기술적 오류 |","|---|---:|---:|---:|---:|"]
    for name,(tag,cohort) in zip(names,cohorts.items()):
        report.append(f"| {name} | {cohort['successes']}/{cohort['attempts']} | {cohort['success_rate']*100:.1f}% | {cohort['mean_policy_steps']} | {cohort['technical_errors']} |")
    report += ["", "기본 OpenVLA와 LoRA는 학습용 20개 시범에서 계산한 동일한 행동 정규화 값을 사용했습니다. 공개 Spatial 미세조정 모델은 별도 참고 기준입니다.",
               "초기 상태 0~4, seed 7, 환경 seed 0, 준비 동작 10회, 최대 220회 행동, 기존 공식 전처리 및 제어 명령 변환을 유지했습니다.",
               f"초기 상태·초기 이미지 일치 확인: {paired}. 학습 자료와 평가 초기 상태가 서로 겹치지 않는다는 보장은 없습니다.",
               "loss 감소와 시뮬레이터 성공률은 서로 다른 결과입니다. 5회 결과만으로 일반적인 성능 향상이나 통계적 유의성을 주장하지 않습니다.",
               "토큰 정확도와 L1은 teacher forcing으로 계산하며, L1은 정규화 행동 단위입니다. 미터 단위나 성공률이 아닙니다.",
               "사진 전처리는 기존 연결 시험의 HF processor를 유지했으며, 공식 RLDS 전체 학습 레시피를 그대로 재현했다는 뜻은 아닙니다.","",
               "학습 설정·validation 지표·checkpoint·환경 보존 검사는 experiment_summary.json에, 업데이트별 기록은 CSV와 JSONL에 남겼습니다.",
               "원본 데이터·checkpoint·optimizer 상태·영상은 로컬 서버에 보존합니다. W&B는 인증이 없어 offline 기록이며 아직 업로드하지 않았습니다.","",
               "## 다시 실행하는 명령","", "원격 Linux의 lerobot312 환경에서 examples/openvla_bowl_experiment.py의 calibrate, train, check, simulate, audit 단계를 사용합니다.",
               "기존 결과 폴더를 덮어쓰지 않도록 새 실험 폴더를 만들고 환경 기준 기록부터 남겨야 합니다. 이번 실험을 다시 평가할 때는 저장된 checkpoint와 동일한 run 경로를 사용하세요.","",
               "## 이번에 기억할 두 개념","", "epoch 하나는 학습용 장면 전체를 한 번 사용하는 것입니다. 시범 20개는 사진 20장이 아닙니다.",
               "validation loss는 시범 행동 예측 오차이고, 시뮬레이터 성공률은 실제 제어 반복 끝에 목표를 달성한 비율입니다."]
    (run/"RESULTS_KO.md").write_text(chr(10).join(report))
    print("REPORT_COMPLETE",run,flush=True)
    print(json.dumps({"before_validation":train["baseline_validation"],"after_validation":train["final_validation"],"models":cohort_rows,"paired":paired},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
