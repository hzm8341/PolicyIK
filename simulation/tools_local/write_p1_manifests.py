"""Write one YAML manifest per P1 run (outputs/p1/{eval,tune,robust,budget}) to outputs/manifests/p1/."""
import ast, datetime, glob, hashlib, os, warnings
import numpy as np
from policyik_sim.metrics import single_attempt
warnings.filterwarnings("ignore")
def sha(f): return hashlib.sha256(open(f, "rb").read()).hexdigest()
base = open("outputs/manifests/E01.yaml").read().split("controller:")[0]
base = "\n".join(l for l in base.splitlines() if not l.startswith(("experiment_id", "status", "run_start_utc")))
tsha = {f: sha(f) for f in ["data/eval_tasks.npz", "outputs/p1/tune_tasks.npz"]}
os.makedirs("outputs/manifests/p1", exist_ok=True)
n = 0
for d, proto in [("outputs/p1/eval", "single"), ("outputs/p1/tune", "tuning"), ("outputs/p1/robust", "robustness"),
                 ("outputs/p1/budget", "budget_320_steps")]:
    for f in sorted(glob.glob(f"{d}/**/*.npz", recursive=True)):
        if "corrected_tasks" in f:
            continue
        z = np.load(f)
        if "config" not in z.files:
            continue
        cfg = ast.literal_eval(str(z["config"]))
        rid = ("P1_" + os.path.relpath(f, "outputs/p1")[:-4]).replace("/", "__")
        s = single_attempt(z)
        mt = datetime.datetime.utcfromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%dT%H:%M:%SZ")
        txt = f"experiment_id: {rid}\nstatus: done\nrun_end_utc: {mt}\n{base}\ncontroller: {cfg['controller']}\n"
        txt += "".join(f"{k}: {v}\n" for k, v in cfg.items() if k != "controller")
        if cfg.get("checkpoint"):
            txt += f"checkpoint_sha256: {sha(cfg['checkpoint'])}\n"
        if cfg.get("task_file") in tsha:
            txt += f"task_file_sha256: {tsha[cfg['task_file']]}\n"
        ts = cfg.get("transient_seed", -1)
        txt += f"protocol: {proto}{'' if ts < 0 else ' transient_seed=' + str(ts)}\n"
        txt += (f"output_npz: {f}\noutput_npz_sha256: {sha(f)}\nlog_file: {f[:-4]}.log\n"
                f"result_strict_pct: {s['strict']:.3f}\nresult_coarse_pct: {s['coarse']:.3f}\n")
        open(f"outputs/manifests/p1/{rid}.yaml", "w").write(txt)
        n += 1
print("wrote", n, "manifests")
