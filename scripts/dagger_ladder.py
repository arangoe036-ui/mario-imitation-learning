"""§5: does adding recovery data MOVE THE PEAK LATER? The prediction that tests the whole thesis.

The peak sits at ~1,000 steps (0.82 epochs) because further passes bind the policy to a trajectory it cannot
hold. **If the corpus's lack of recoveries is the binding constraint, augmenting it should let the policy keep
learning for longer and move the peak later and higher.** If the peak stays at ~1,000, the diagnosis is
incomplete.

Same mixture as §3c, one seed, trained to each rung and evaluated at n=100. **The augmented arm learned a
marginal in round 1 (Left 0.05 -> 0.55), so this ladder characterises that arm** -- the prediction is still
worth testing, but a peak that does not move is confounded with the mix imbalance and is reported as such.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.generation_sweep as G
import scripts.overnight as O
from scripts.button_mask_eval import rollout
from scripts.compose import warm_session
from scripts.scaleup_eval import resumable, score
from tasdata.bc import rollout_budget as RB
from tasdata.bc.budget import Deadline, TimedOut, time_limit
from tasdata.bc.data import FrameStackDataset
from tasdata.bc.model import BCPolicy, PolicyConfig, pick_device
from tasdata.bc.runlength import RunLengthDataset, joint_size

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/dagger_ladder.json"
RUNGS = [500, 1000, 2000, 5000, 15000]
N_EVAL, TEMP, SEED = 100, 0.7, 0
EXPERT_FLOOR, BATCH, LR = 20_000, 64, 3e-4
WALLS = {"pipe2_630": 630, "pipe3_735": 735, "pipe4_975": 975}

def main():
    dl = Deadline(float(sys.argv[1]) if len(sys.argv) > 1 else 60*60)
    z = np.load(ROOT/"data/dagger_round1_samples.npz", allow_pickle=True)
    new_obs = torch.from_numpy(z["obs"]).float().div_(255.0); new_lab = z["lab"]
    ctx = O.Ctx(); n_cls = joint_size(ctx.vocab.size)
    start = next(p for p in ctx.points if p.kind=="level_start" and p.label=="1-1")
    base = FrameStackDataset(ctx.expert_train, ctx.vocab, stack=4, label_mode="token")
    zz = np.load(ROOT/"data/runlength_index_runs.npz")
    expert = RunLengthDataset(base, {k: zz[k] for k in ("rows","joints","lengths")})
    n_new = len(new_lab); n_exp = min(EXPERT_FLOOR, len(expert))
    byte_of = np.array([ctx.vocab.decode_byte(c//G.N_BUCKETS) for c in range(n_cls)], dtype=np.int64)
    lut = G.class_lengths({k: zz[k] for k in ("rows","joints","lengths")}, n_cls)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("rungs", {})
    out.update({"rungs_planned": RUNGS, "n_eval": N_EVAL, "seed": SEED,
                "mix": {"expert": int(n_exp), "correction": int(n_new)},
                "terminator": RB.describe(),
                "caveat": ("round 1 learned a MARGINAL (Left 0.05 -> 0.55) because 1,480 of 1,507 "
                           "correction samples were retreat macros; a flat ladder here is confounded "
                           "with that mix imbalance")})
    torch.manual_seed(SEED); dev = pick_device("mps")
    cfg = PolicyConfig(n_actions=n_cls, stack=4, frame_size=84, d_model=64, n_layers=1,
                       head_type="categorical", cnn_channels=(32,64,64))
    policy = BCPolicy(cfg).to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)
    g = np.random.default_rng(SEED)
    pick = g.choice(len(expert), size=n_exp, replace=False)
    step = 0; losses=[]
    for rung in RUNGS:
        while step < rung:
            k_new = max(1, min(BATCH-1, int(round(BATCH*n_new/(n_exp+n_new)))))
            ei = g.choice(n_exp, size=BATCH-k_new, replace=False)
            ni = g.choice(n_new, size=k_new, replace=(n_new < k_new))
            xb, yb = [], []
            for t in ei:
                o,_p,y = expert[int(pick[t])]; xb.append(o); yb.append(y)
            ob = torch.stack(xb + [new_obs[int(j)] for j in ni]).to(dev)
            yv = torch.tensor(list(yb)+[int(new_lab[int(j)]) for j in ni], dtype=torch.long).to(dev)
            loss = torch.nn.functional.cross_entropy(policy(ob), yv)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0); opt.step()
            losses.append(float(loss.detach())); step += 1
        ck = ROOT/f"data/bc_scaleup/DAGLAD_s{SEED}_{rung}.pt"
        torch.save({"model_state":{k:v.cpu() for k,v in policy.state_dict().items()},
                    "policy_config":cfg.to_dict(),"corpus":"runs","step":rung,
                    "loss_at_snapshot":float(np.mean(losses[-250:]))}, ck)
        print(f"  trained to {rung}, loss {np.mean(losses[-250:]):.4f}", flush=True)
    # ---- evaluate ----
    def sess_get():
        s = G.session_when_free(O.ROM, O.MOVIE, ctx.frames_needed()); warm_session(s, start.frame); return s
    for rung in RUNGS:
        key = str(rung)
        if key in out["rungs"]: continue
        if not dl.can_afford(120):
            out.setdefault("skipped",[]).append(rung); continue
        blob = torch.load(ROOT/f"data/bc_scaleup/DAGLAD_s{SEED}_{rung}.pt", map_location="cpu", weights_only=False)
        cf = PolicyConfig.from_dict(blob["policy_config"]); pol = BCPolicy(cf)
        pol.load_state_dict(blob["model_state"]); pol.eval()
        tp = ROOT/f"data/traces/daglad_{rung}_{N_EVAL}.json"
        try:
            with time_limit(min(12*60, dl.remaining()-60), f"rung {rung}"):
                s = sess_get()
                try:
                    traces = resumable(tp, N_EVAL, lambda i: rollout(s, pol, cf, start, i, lut, byte_of, None, temp=TEMP))
                finally: s.close()
        except TimedOut as e:
            out.setdefault("skipped",[]).append({"rung":rung,"reason":str(e)}); continue
        rec = score(f"daglad_{rung}", traces)
        xs = [max(f[0] for f in t.frames) for t in traces]
        rec.update({"steps":rung,"loss":blob.get("loss_at_snapshot"),
                    "past_wall":{w:{"rate":float(np.mean([x>v for x in xs]))} for w,v in WALLS.items()}})
        out["rungs"][key] = rec
        OUT.write_text(json.dumps(out, indent=2, default=str))
        pw = rec["past_wall"]
        print(f"  {dl.stamp()} rung {rung:6d} loss {rec['loss']:.3f} x_med {rec['x_median']:4.0f} "
              f"p2 {pw['pipe2_630']['rate']*100:5.1f}% p3 {pw['pipe3_735']['rate']*100:5.1f}% "
              f"Left {rec['button_marginals']['rates']['Left']:.3f}", flush=True)
    rr = [out["rungs"][str(r)] for r in RUNGS if str(r) in out["rungs"]]
    if rr:
        best = max(rr, key=lambda r: r["past_wall"]["pipe3_735"]["rate"])
        out["peak"] = {"steps": best["steps"], "past_pipe3": best["past_wall"]["pipe3_735"]["rate"]*100,
                       "baseline_peak_steps": 1000,
                       "moved_later": bool(best["steps"] > 1000)}
        out["verdict"] = (
            f"**Peak on the augmented set is at {best['steps']} steps** (past pipe 3 "
            f"{best['past_wall']['pipe3_735']['rate']*100:.1f}%), against ~1,000 on the expert-only "
            f"corpus. " + ("**The peak moved later, as the diagnosis predicts.**" if best["steps"]>1000
            else "**The peak did NOT move later** -- but round 1 learned a marginal, so this is "
                 "confounded with the mix imbalance and does not cleanly refute the diagnosis."))
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n"+"="*78); print(out.get("verdict","no rungs")); print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
