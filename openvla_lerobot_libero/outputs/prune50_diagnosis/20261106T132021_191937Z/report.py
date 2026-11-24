"""Evidence-based contact/lift diagnosis; no new success-rate benchmark."""
from pathlib import Path
import json,csv,hashlib,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False))
CFG=read(R/'diagnostic_config.json');SOURCE=Path(CFG['source_run'])
result=read(R/'replay_results.json');assert result['complete'] and result['all_replays_match_original_trajectories']
same=read(R/'same_image_comparison.json');assert same['complete'] and same['own_recorded_actions_reproduced']
summary=[]
for row in result['rows']:
    commands=[json.loads(l) for l in (SOURCE/'simulation'/row['tag']/f"state_{row['state']:02d}"/'steps.jsonl').read_text().splitlines()]
    signs=[r['controller_action'][-1] for r in commands]
    switches=sum(a!=b for a,b in zip(signs,signs[1:]))
    window=[r['controller_action'][-1] for r in commands if 60<=r['step']<=100]
    window_switches=sum(a!=b for a,b in zip(window,window[1:]))
    if row['official_success']:
        category='접시 위 도달 성공(손 완전 해제는 별도)'
    elif row['stable_grasp_and_lift_observed']:
        category='안정적으로 든 뒤 운반 또는 배치 단계 실패'
    elif row['physics_contact_measurements']['both_contact_samples']:
        category='양쪽 접촉은 있었지만 안정적인 집기·들기 실패'
    else:category='대상 그릇을 양쪽 손가락으로 잡지 못함'
    summary.append({'model':row['tag'],'state':row['state'],'failure_stage':category,
        'first_move_5mm_step':row['first_motion_5mm'],'first_grasp_step':row['first_grasp_both_pads'],
        'first_lift_5cm_step':row['first_lift_5cm'],'first_grasp_and_lift_step':row['first_grasp_and_lift_5cm'],
        'stable_grasp_and_lift':row['stable_grasp_and_lift_observed'],'max_lift_cm':100*row['max_target_lift_m'],
        'final_plate_xy_cm':100*row['final_target_plate_xy_m'],
        'max_both_pad_contact_ms':1000*row['physics_contact_measurements']['max_consecutive_grasp_samples']*.002,
        'official_success':row['official_success'],'success_with_grasp_contact':row['official_success_while_grasp_contact'],
        'first_close_command_step':row['first_close_command'],'first_open_after_grasp_command':row['first_open_after_grasp_command'],
        'gripper_command_switches':switches,'gripper_switches_per_100_actions':100*switches/len(commands),
        'gripper_switches_in_actions_60_to_100':window_switches,
        'replay_exact':row['replay_exact']})
cohort={}
for tag in CFG['tags']:
    rows=[r for r in result['rows'] if r['tag']==tag]
    cohort[tag]={'trials':len(rows),'any_both_pad_contact':sum(r['physics_contact_measurements']['both_contact_samples']>0 for r in rows),
        'stable_grasp_and_lift':sum(r['stable_grasp_and_lift_observed'] for r in rows),
        'lifted_5cm':sum(r['first_lift_5cm'] is not None for r in rows),'official_success':sum(r['official_success'] for r in rows),
        'first_motion_5mm_steps':[r['first_motion_5mm'] for r in rows],
        'max_lift_cm':[100*r['max_target_lift_m'] for r in rows]}
with (R/'stage_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
same_rows=same['rows'];unique={r['image_sha256'] for r in same_rows}
same_evidence={'tested_observation_records':len(same_rows),'unique_images':len(unique),
    'gripper_command_disagreement_records':sum(r['gripper_command_disagrees'] for r in same_rows),
    'initial_same_image':same_rows[0],
    'same_dense_pregrasp_frame_68':next(r for r in same_rows if r['source_tag']=='dense_sft' and r['frame']==68),
    'same_dense_pregrasp_frame_60':next(r for r in same_rows if r['source_tag']=='dense_sft' and r['frame']==60)}
audit={};changed=[]
for name,digest in read(R/'protected_before.json').items():
    if not Path(name).exists() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:changed.append(name)
env={}
for name,exe in read(R/'interpreters.json').items():
    snap=subprocess.run([exe,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (R/(name+'_after.txt')).write_text(snap)
    env[name]=sorted(snap.splitlines())==sorted((R/(name+'_before.txt')).read_text().splitlines())
assert not changed and all(env.values())
audit={'protected_files':len(read(R/'protected_before.json')),'changed_files':changed,'environments_unchanged':env}
write(R/'preservation_report.json',audit)
report={'complete':True,'no_training':True,'no_optimizer_updates':True,'all_recorded_trajectories_reproduced':True,
    'cohort':cohort,'per_trial':summary,'same_image_evidence':same_evidence,'preservation':audit,
    'definitions':{'grasp':'Both robosuite fingerpad geometry groups contact the target object.',
        'lift':'Target body center at least5cm above its post-warmup height; chosen diagnostic threshold, not LIBERO success definition.',
        'stable_grasp_and_lift':'Both contact plus5cm lift for >=3 consecutive20Hz control snapshots.',
        'official_success':'Original LIBERO goal predicate: target bowl on plate; does not explicitly require release.',
        'physics_contact_sampling':'Every MuJoCo .step() (2ms), plus control snapshots; read-only instrumentation.'},
    'not_proven':['No exact causal identification of which pruned head/neuron caused the failures.',
        'No proof that more epochs, different LoRA targets or weaker pruning would fix the behavior.',
        'Same-image counterfactuals compare commands but are not successful rescue rollouts.',
        'Does not rule out grasping or moving a distractor object; primary sensors track designated target.',
        'This is diagnosis of the existing five fixed states, not a new generalization benchmark.']}
write(R/'diagnosis.json',report)
fig,axes=plt.subplots(2,2,figsize=(12,7))
colors={'dense_sft':'#377bbc','p50':'#df873d'}
for tag in CFG['tags']:
    trace=[json.loads(l) for l in (R/tag/'state_00/trace.jsonl').read_text().splitlines()]
    steps=[r['step'] for r in trace];c=colors[tag];label='Original + SFT1' if tag=='dense_sft' else '50% pruned + SFT1'
    axes[0,0].plot(steps,[100*r['target_lift_m'] for r in trace],label=label,color=c)
    axes[0,1].plot(steps,[100*r['target_plate_xy_m'] for r in trace],label=label,color=c)
    axes[1,0].step(steps,[int(r['grasp_both_pads']) for r in trace],where='post',label=label,color=c)
    axes[1,1].plot(steps,[100*r['eef_target_xy_m'] for r in trace],label=label,color=c)
axes[0,0].axhline(5,color='gray',ls='--',lw=1);axes[0,0].set_title('State0: target bowl lift (cm)')
axes[0,1].axhline(3,color='gray',ls='--',lw=1);axes[0,1].set_title('Bowl-to-plate horizontal distance (cm)')
axes[1,0].set_title('Both fingerpads contact the TARGET bowl');axes[1,0].set_yticks([0,1],['No','Yes'])
axes[1,1].set_title('End-effector-to-bowl horizontal distance (cm)')
for ax in axes.flat:ax.set_xlabel('Recorded policy action step');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Exact replay of the same previously evaluated initial state | No training',fontsize=13)
fig.tight_layout(rect=[0,.02,1,.95]);fig.savefig(R/'state0_contact_lift_comparison.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for tag,offset in [('dense_sft',-.17),('p50',.17)]:
    rows=[x for x in summary if x['model']==tag]
    axes[0].bar(np.arange(5)+offset,[x['max_lift_cm'] for x in rows],width=.32,label=tag,color=colors[tag])
    axes[1].bar(np.arange(5)+offset,[int(x['stable_grasp_and_lift']) for x in rows],width=.32,label=tag,color=colors[tag])
axes[0].axhline(5,color='gray',ls='--',lw=1);axes[0].set_title('Maximum TARGET bowl lift (cm)')
axes[1].set_title('Stable target grasp + 5cm lift observed');axes[1].set_yticks([0,1],['No','Yes'])
for ax in axes:ax.set_xticks(range(5),[str(i) for i in range(5)]);ax.set_xlabel('Previously evaluated state');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(R/'all_states_grasp_lift.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,3.2))
for tag in CFG['tags']:
    commands=[json.loads(l) for l in (SOURCE/'simulation'/tag/'state_00/steps.jsonl').read_text().splitlines()]
    ax.step([r['step'] for r in commands],[r['controller_action'][-1] for r in commands],where='post',label=tag,color=colors[tag])
ax.set_yticks([-1,1],['OPEN','CLOSE']);ax.set_xlim(1,110);ax.set_xlabel('State0 recorded policy action');ax.set_title('Gripper command sequence (same initial state, diverging observations)')
ax.legend();ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(R/'state0_gripper_commands.png',dpi=160);plt.close(fig)
font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),15)
board=Image.new('RGB',(1024,640),'white');draw=ImageDraw.Draw(board)
for rowindex,tag in enumerate(CFG['tags']):
    trial=next(x for x in result['rows'] if x['tag']==tag and x['state']==0)
    if tag=='dense_sft':points=[(0,'Start'),(trial['first_grasp_both_pads'],'Both-pad grasp'),(trial['first_lift_5cm'],'Lift >5cm'),(trial['first_official_success'],'On plate (still held)')]
    else:
        trace=[json.loads(l) for l in (R/tag/'state_00/trace.jsonl').read_text().splitlines()]
        closest=min(trace,key=lambda x:x['eef_target_xy_m'])['step'];peak=max(trace,key=lambda x:x['target_lift_m'])['step']
        points=[(0,'Start'),(closest,'Closest horizontal approach'),(peak,'Largest target lift'),(220,'No task success')]
    trace={r['step']:r for r in [json.loads(l) for l in (R/tag/'state_00/trace.jsonl').read_text().splitlines()]}
    draw.text((8,rowindex*320+2),'ORIGINAL + SFT1' if tag=='dense_sft' else '50% PRUNED + SFT1',font=font,fill=colors[tag])
    for col,(step,label) in enumerate(points):
        with Image.open(SOURCE/'simulation'/tag/'state_00/frames'/f'{step:04d}.png') as source:image=source.convert('RGB')
        board.paste(image,(col*256,rowindex*320+28))
        text=f"step {step}: {label}\nlift {trace[step]['target_lift_m']*100:.1f}cm; grasp {trace[step]['grasp_both_pads']}"
        draw.multiline_text((col*256+4,rowindex*320+286),text,font=font,fill='black',spacing=3)
board.save(R/'state0_keyframes.png')
original=cohort['dense_sft'];compressed=cohort['p50']
lines=['이번 진단은 여기부터 읽어줘','',
    '추가 학습은 하지 않았어. 저장된 행동을 그대로 재생하고, 모델은 고정한 채 같은 사진의 예측도 비교했어.',
    '원래 로봇 위치·관절·성공 판정과 일치했고, 처음/주기적 사진도 픽셀 차이0으로 재현했어.','',
    '단계                         원본                  50% 모델',
    f"양쪽 손가락의 대상 접촉      {original['any_both_pad_contact']}/5                   {compressed['any_both_pad_contact']}/5",
    f"안정적으로 잡고 5cm 들기     {original['stable_grasp_and_lift']}/5                   {compressed['stable_grasp_and_lift']}/5",
    f"접시 위 도달 판정            {original['official_success']}/5                   {compressed['official_success']}/5",'',
    '상태별 구분은 stage_comparison.csv에서 확인할 수 있어.','']
for row in summary:lines.append(f"{row['model']} 상태{row['state']}: {row['failure_stage']}; 최대 높이 +{row['max_lift_cm']:.2f}cm; 양쪽접촉 시작 {row['first_grasp_step']}")
lines+=['','같은 사진에서도 달랐나?',
    f"{len(same_rows)}개 입력 기록({len(unique)}개 고유 사진)을 두 모델에 똑같이 넣었어. 각 모델 자신의 원래 기록 행동도 재현했어.",
    '처음 사진에서 원본은 물체 쪽으로 움직이는 명령,50%모델은 거의 제자리 명령을 냈어.',
    '집기 직전의 같은 사진에서도 이동·회전 명령이 달랐어. 손 닫기 명령이 같은 장면도 있었어.',
    '따라서 사진이 달라진 탓만은 아니며, 그리퍼 열고 닫기만의 문제로도 볼 수 없어.',
    '상태0의 시작점에서5mm움직이기까지: 원본2행동,50%모델25행동. 상태1은원본2행동,50%모델64행동이었어.',
    '상태0의손여닫기전환은원본5회/104행동,50%모델41회/220행동이야. 길이가달라비율도CSV에함께남겼어.',
    '같은길이의60~100행동구간만보면원본0회,50%모델15회전환이야. 단,두궤적의사진은이미달라진구간이야.',
    '50%상태0은그릇중심높이가약6.2cm올랐지만양손가락동시접촉이없었어. 높이상승만으로안정적인집기라고판정하면안돼.','',
    '확인된 사실과 추정의 경계',
    '이 기록은 어디서 실패하는지를 보여줘. 어느 가중치/attention head가 원인인지까지 증명하지는 못했어.',
    '집을 위치·방향·이동량을 맞추는 행동과 손여닫기 시점이 충분히 회복되지 않았을 가능성이 있어.',
    '다만 이 둘의 인과 기여도를 나눈 구조변경/명령교체 실험은 하지 않았고, 추가 학습하면 해결될지도 아직 몰라.',
    '원본 상태0은 접시 위 성공 시에도 양쪽 손가락 접촉이 남아 있었어. 완전히 놓기까지 성공했다고 말하면 안 돼.',
    '원본 상태3에서는 그릇을 든 뒤 열기 명령이 나오고 손가락 접촉이 사라진 것도 확인됐어.',
    '공식 성공 기준은 그릇이 접시 위에 있는지야. 안정 집기 기준(5cm,3연속step)은 이번 원인 진단용으로 정한 기준이야.','',
    '읽어볼 코드',
    'replay.py의 one(): 저장 행동 재생 → 손가락 접촉·물체 높이 측정 → 원래 궤적과 대조.',
    'same_image_probe.py: 두 모델에 같은 사진과 지시를 입력. 모든 가중치를 고정하고 추론만 실행.',
    'report.py: 단계별 표와 실제 장면·그래프 작성.',
    'state0_keyframes.png: 집기·들기·접시 위 도달을 사진으로 비교.',
    'state0_contact_lift_comparison.png: 상태0의 높이·접촉·거리 그래프.',
    'all_states_grasp_lift.png: 다섯 상태 전체 들기 비교.',
    'diagnosis.json: 측정값과 한계,보존 검사 전체 기록.',
    '', '핵심 개념2개: 손가락 접촉은 잡기의 단서이고, 실제로 그릇을 들고 유지했는지는 별도 확인해야 해.',
    '같은 입력 비교는 사진 차이의 영향을 줄여 모델의 행동 차이를 확인하는 방법이야.']
(R/'START_HERE_KO.txt').write_text('\n'.join(lines),encoding='utf-8')
print('GRASP_DIAGNOSIS_COMPLETE',json.dumps(cohort),flush=True)
