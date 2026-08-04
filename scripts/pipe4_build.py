"""Pipe 4: probe, requirement, search-and-distil. One script, three steps.

Step 1 probe   -- controlled slow walk through x=860-1010 recording on_ground() and y per x.
                  Cannot fail on sampling (we choose where Mario goes) or ambiguity (the flag).
Step 2 require -- from the 29 real stuck states, sweep (trigger x, A-hold) with on_ground() a
                  REQUIRED condition at the jump frame. Threshold past the measured far edge.
Step 3 search  -- random shooting with the policy as proposal from those same states; score
                  clearing AND survival past the obstacle; distil the winners; re-measure.

y is never read during a pipe transit: a large negative dx marks the transit and those frames
are excluded, because the frame before reads y=256 and the transit frame reads y=0 (page reset).
"""
from __future__ import annotations
import json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, torch
from torch.utils.data import ConcatDataset, Subset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.overnight as O
from scripts.compose import session_when_free, train
from tasdata.bc.overnight_lib import calibrate, load_policy, random_rows, save_policy, wilson
from tasdata.bc.tokens import LIVE_MASK
from tasdata.bc.trace_log import EpisodeTrace, write_traces
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from tasdata.dataset import load_run_dir
from tasdata.ram import on_ground, read_smb, y_absolute
from tasdata.replay import _resize_gray

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/pipe4_build.json"
CKPT = ROOT / "data/bc_coverage/C_control_matched_r2.pt"
A, B, RIGHT, LEFT = (NES_BUTTON_BITS["A"], NES_BUTTON_BITS["B"],
                     NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["Left"])
STUCK = [0,8,12,16,18,21,25,27,31,34,40,45,58,63,80,97,99,100,101,121,135,138,142,167,175,177,180,181,197]
HANDOVER = 880
FLOORY = 432

def prefix(session, policy, cfg, thr, start, seed, target=HANDOVER, cap=2000):
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8); win[:] = _resize_gray(obs.rgb, (84,84))
    seq = []
    for _ in range(cap):
        with torch.no_grad():
            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0/(1.0+np.exp(-lg))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]: byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        seq.append(byte); obs = session.step(byte)
        win = np.roll(win,-1,0); win[-1] = _resize_gray(obs.rgb,(84,84))
        st = read_smb(obs.ram, obs.framecount)
        if st.player_state in (0x06,0x0B): return None, None
        if st.x_position >= target: return seq, obs
    return None, None

def replay(session, start, seq):
    obs = session.reset(start.frame)
    for b in seq: obs = session.step(b)
    return obs

def probe(session, start, seq, frames=420):
    """Slow walk right: tap Right every other frame so speed stays low and every x is sampled."""
    obs = replay(session, start, seq)
    rows, prev_x = [], None
    for i in range(frames):
        byte = RIGHT if i % 2 == 0 else 0
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        x, ya, g = st.x_position, y_absolute(obs.ram), on_ground(obs.ram)
        if prev_x is not None and x - prev_x < -100: break      # pipe transit: stop, y invalid
        prev_x = x
        rows.append((int(x), int(ya), int(g), int(st.player_state)))
        if st.player_state in (0x06,0x0B): break
    return rows

def attempt(session, start, seq, *, hold, trigger, frames=260):
    """Jump at trigger with `hold` frames of A. on_ground() REQUIRED at the jump frame."""
    obs = replay(session, start, seq)
    left, jumped, maxx, grounded_at_jump = hold, False, 0, None
    top_y, died, prev_x = None, False, None
    for _ in range(frames):
        st = read_smb(obs.ram, obs.framecount)
        x = st.x_position
        byte = RIGHT | B
        if not jumped and x >= trigger:
            grounded_at_jump = on_ground(obs.ram)
            if not grounded_at_jump:
                return {"hold":hold,"trigger":trigger,"grounded_at_jump":False,
                        "max_x":int(maxx),"cleared":False,"died":False,"skipped":True}
            jumped = True
        if jumped and left > 0:
            byte |= A; left -= 1
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        if prev_x is not None and st.x_position - prev_x < -100: break
        prev_x = st.x_position
        maxx = max(maxx, st.x_position)
        if on_ground(obs.ram) and st.x_position > trigger + 8:
            top_y = y_absolute(obs.ram) if top_y is None else min(top_y, y_absolute(obs.ram))
        if st.player_state in (0x06,0x0B): died = True; break
    return {"hold":hold,"trigger":trigger,"grounded_at_jump":bool(grounded_at_jump),
            "max_x":int(maxx),"top_y":top_y,"cleared":bool(maxx>975),"died":died,"skipped":False}

def search_from(session, start, seq, policy, cfg, thr, *, seed, K=48, L=70):
    """Random shooting with the policy as proposal. Score: cleared AND survived."""
    rng = np.random.default_rng(seed)
    best = None
    for k in range(K):
        obs = replay(session, start, seq)
        win = np.zeros((cfg.stack,84,84), np.uint8); win[:] = _resize_gray(obs.rgb,(84,84))
        acts, maxx, died = [], 0, False
        for t in range(L):
            with torch.no_grad():
                lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
            p = 1.0/(1.0+np.exp(-lg))
            p = np.clip(p + rng.normal(0, 0.35, 8), 0.02, 0.98)   # perturbed proposal
            bits = rng.random(8) < p
            byte = 0
            for j, nm in enumerate(NES_BUTTON_ORDER):
                if bits[j]: byte |= NES_BUTTON_BITS[nm]
            byte &= LIVE_MASK
            acts.append(byte); obs = session.step(byte)
            win = np.roll(win,-1,0); win[-1] = _resize_gray(obs.rgb,(84,84))
            st = read_smb(obs.ram, obs.framecount)
            maxx = max(maxx, st.x_position)
            if st.player_state in (0x06,0x0B): died = True; break
        # survival: 40 more frames of the policy's own play, must stay alive
        survived = not died
        if survived:
            for _ in range(40):
                obs = session.step(RIGHT | B)
                if read_smb(obs.ram, obs.framecount).player_state in (0x06,0x0B):
                    survived = False; break
        score = (1000 if (maxx > 975 and survived) else 0) + maxx
        if best is None or score > best["score"]:
            best = {"score":score,"max_x":int(maxx),"cleared":bool(maxx>975),
                    "survived":bool(survived),"actions":acts,"cand":k}
    return best

def main():
    t0=time.time(); ctx=O.Ctx()
    start=next(p for p in ctx.points if p.kind=="level_start" and p.label=="1-1")
    policy,cfg,_=load_policy(CKPT)
    cal,_=calibrate(policy,ctx.dataset(ctx.expert_train),ctx.target_rates)
    thr=cal.vector.astype(np.float64)
    out={"checkpoint":CKPT.name,"stuck_seeds":STUCK}
    s=session_when_free(O.ROM,O.MOVIE,ctx.frames_needed())
    try:
        # find prefixes from the real stuck seeds
        prefixes={}
        for sd in STUCK:
            sq,_=prefix(s,policy,cfg,thr,start,sd)
            if sq: prefixes[sd]=sq
            if len(prefixes)>=10: break
        print(f"prefixes reaching x>={HANDOVER}: {sorted(prefixes)}",flush=True)
        if not prefixes: print("no prefix reached the handover"); return
        sd0=sorted(prefixes)[0]

        print("\nSTEP 1 probe",flush=True)
        rows=probe(s,start,prefixes[sd0])
        byx={}
        for x,ya,g,ps in rows: byx.setdefault(x,[]).append((ya,g))
        gx=sorted(x for x,v in byx.items() if any(g for _,g in v))
        elev=sorted({(x,ya) for x,v in byx.items() for ya,g in v if g and ya<FLOORY-8})
        print(f"  {len(rows)} frames, {len(byx)} distinct x ({min(byx)}-{max(byx)})")
        print(f"  grounded at {len(gx)} distinct x")
        print(f"  grounded ABOVE floor: {elev[:12]}")
        surf=min((ya for _,ya in elev), default=None)
        print(f"  -> pipe 4 surface y {surf} = "
              f"{((FLOORY-surf)/16 if surf else 0):.2f} tiles above floor")
        out["probe"]={"n_frames":len(rows),"grounded_x":gx,"elevated":elev,
                      "surface_y":surf,"height_tiles":(FLOORY-surf)/16 if surf else None}

        print("\nSTEP 2 requirement",flush=True)
        reqs=[]
        for sd in sorted(prefixes)[:4]:
            for trig in range(880, 925, 4):
                for hold in (8,12,16,20,26,32):
                    reqs.append(dict(seed=sd, **attempt(s,start,prefixes[sd],hold=hold,trigger=trig)))
        ok=[r for r in reqs if r["cleared"]]
        skipped=sum(1 for r in reqs if r.get("skipped"))
        print(f"  {len(reqs)} configs, {skipped} skipped (not grounded at the jump frame)")
        assert len({r['cleared'] for r in reqs})>1 or not reqs, "degenerate: one outcome only"
        print(f"  cleared x>975: {len(ok)}")
        if ok:
            m=min(ok,key=lambda r:(r['hold'],r['trigger']))
            print(f"  minimum: hold={m['hold']} trigger={m['trigger']} seed={m['seed']}")
        out["requirement"]={"n":len(reqs),"skipped":skipped,"cleared":len(ok),
                            "min":(min(ok,key=lambda r:(r['hold'],r['trigger'])) if ok else None),
                            "rows":reqs}

        print("\nSTEP 3 search",flush=True)
        found=[]
        for sd in sorted(prefixes)[:8]:
            b=search_from(s,start,prefixes[sd],policy,cfg,thr,seed=sd)
            found.append(dict(seed=sd, cleared=b["cleared"], survived=b["survived"],
                              max_x=b["max_x"], n_actions=len(b["actions"])))
            print(f"  seed {sd}: max_x {b['max_x']} cleared={b['cleared']} "
                  f"survived={b['survived']}",flush=True)
            if b["cleared"] and b["survived"]:
                found[-1]["actions"]=[int(v) for v in b["actions"]]
        out["search"]=found
        nc=sum(1 for f in found if f["cleared"] and f["survived"])
        print(f"\nsearch found a clearing+surviving sequence for {nc}/{len(found)} states")
        out["search_success"]=nc
    finally:
        s.close()
    out["minutes"]=round((time.time()-t0)/60,1)
    OUT.write_text(json.dumps(out,indent=2,default=str))
    print(f"wrote {OUT} ({out['minutes']} min)")

if __name__=="__main__":
    main()
