from pathlib import Path
import argparse,importlib.util,json
RUN=Path(__file__).parent
PREVIOUS=Path("/home/jovyan/projects/lerobot-openvla/outputs/bowl40_experiment/20261103T051914_711739Z")
p=argparse.ArgumentParser();p.add_argument("tag",choices=["epoch1","epoch5"]);args=p.parse_args()
spec=importlib.util.spec_from_file_location("grasp_contact_helper",PREVIOUS/"check_grasp_contacts.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
reports=[]
for state in range(5):
 result=m.analyze(RUN/"simulation"/args.tag/f"state_{state:02d}")
 assert result["success_flags_match_original"] and result["maximum_eef_replay_error_m"]<1e-7
 reports.append(result)
(RUN/f"motion_{args.tag}.json").write_text(json.dumps(reports,indent=2))
print("MOTION_COMPLETE",args.tag,flush=True)
