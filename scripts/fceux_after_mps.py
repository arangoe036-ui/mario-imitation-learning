"""Does the train-on-MPS / evaluate-on-CPU process boundary actually hold?

`pick_device`'s docstring records that touching MPS poisons every *subsequently spawned* FCEUX child into
Qt's software OpenGL backend, irreversibly, and that the poison is inherited. §2 of the fifty-third directive
proposes training in a dedicated MPS process and evaluating in a separate CPU one. **That plan rests entirely
on the poison being confined to the process that touched Metal and its children** -- which is stated in a
docstring and has never been measured as a boundary.

Three launches, in order, so the claim becomes a measurement:

| case | what it shows |
|---|---|
| `clean` | FCEUX works in a fresh process -- the control |
| `poisoned` | FCEUX **breaks** in a process that used MPS first -- the poison is real, not folklore |
| `after` | FCEUX works in a fresh process launched **after** an MPS process exited -- **the boundary holds** |

`clean` and `poisoned` are separate `python` invocations because the poison is irreversible within a process,
so they cannot both be tested in one. The driver at the bottom runs all three.

**If `poisoned` unexpectedly succeeds, that is not good news** -- it means the failure is intermittent and the
boundary cannot be trusted on the strength of one passing run. Reported as such rather than as a pass.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/mps_boundary.json"
STEPS = 30


def gl_backend(log: str) -> str:
    """Which OpenGL path Qt chose, from FCEUX's own log."""
    low = log.lower()
    if "software" in low:
        return "SOFTWARE"
    if "metal" in low:
        return "Metal"
    return "unknown"


def launch_fceux() -> dict:
    """Open a real session, step a few frames, close. Returns exit code and GL backend."""
    import scripts.overnight as O
    from scripts.compose import session_when_free

    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    t0 = time.time()
    rec: dict = {"steps_requested": STEPS}
    s = None
    try:
        s = session_when_free(O.ROM, O.MOVIE, [start.frame])
        obs = s.reset(start.frame)
        n = 0
        for _ in range(STEPS):
            obs = s.step(0)
            n += 1
        rec.update({"ok": True, "steps_served": n, "framecount": int(obs.framecount)})
    except BaseException as e:                      # a segfault surfaces as FceuxError here
        rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:400]})
    finally:
        wd = getattr(s, "_workdir", None) if s is not None else None
        proc = getattr(s, "_proc", None) if s is not None else None
        log = ""
        if wd is not None and (Path(wd) / "fceux.log").exists():
            log = (Path(wd) / "fceux.log").read_text(errors="replace")
        if s is not None:
            try:
                s.close()
            except BaseException:
                pass
        rec["exit_code"] = (proc.returncode if proc is not None else None)
        rec["gl_backend"] = gl_backend(log)
        rec["log_tail"] = log[-400:]
        rec["seconds"] = round(time.time() - t0, 1)
    return rec


def touch_mps() -> dict:
    """Use MPS in *this* process, the way a training run would."""
    import torch

    from tasdata.bc.model import BCPolicy, PolicyConfig, pick_device
    dev = pick_device("mps")
    p = BCPolicy(PolicyConfig(n_actions=300, head_type="categorical")).to(dev)
    x = torch.rand(8, 4, 84, 84, device=dev)
    y = p(x, torch.zeros(8, 1, dtype=torch.long, device=dev))
    torch.mps.synchronize()
    return {"device": str(dev), "logits_shape": list(y.shape),
            "probe_used": "pick_device('mps') -- explicit, never 'auto'"}


def main() -> None:
    mode = os.environ.get("BOUNDARY_MODE", "driver")

    if mode == "clean":
        print(json.dumps({"case": "clean", "mps_touched": False, **launch_fceux()}))
        return
    if mode == "poisoned":
        m = touch_mps()
        print(json.dumps({"case": "poisoned", "mps_touched": True, "mps": m, **launch_fceux()}))
        return
    if mode == "train_only":
        # An MPS process that never spawns FCEUX, then exits -- the thing §2 asks for.
        m = touch_mps()
        print(json.dumps({"case": "train_only", "mps_touched": True, "mps": m,
                          "spawned_fceux": False}))
        return

    # ---- driver: three separate interpreters, in order ----
    env = dict(os.environ)
    results = {}
    plan = [("clean", "clean"), ("poisoned", "poisoned"), ("after", "clean")]
    for label, m in plan:
        if label == "after":
            env["BOUNDARY_MODE"] = "train_only"
            t = subprocess.run([sys.executable, __file__], capture_output=True, text=True,
                               env=env, cwd=str(ROOT))
            try:
                results["train_only"] = json.loads(t.stdout.strip().splitlines()[-1])
            except Exception:
                results["train_only"] = {"stdout": t.stdout[-400:], "stderr": t.stderr[-400:]}
            print(f"[train_only] MPS process ran and exited (rc={t.returncode})", flush=True)
        env["BOUNDARY_MODE"] = m
        r = subprocess.run([sys.executable, __file__], capture_output=True, text=True,
                           env=env, cwd=str(ROOT))
        try:
            rec = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            rec = {"case": label, "ok": False, "parse_failed": True,
                   "stdout": r.stdout[-600:], "stderr": r.stderr[-600:]}
        rec["subprocess_rc"] = r.returncode
        results[label] = rec
        print(f"[{label:9s}] ok={rec.get('ok')} exit={rec.get('exit_code')} "
              f"GL={rec.get('gl_backend')} {rec.get('error', '')[:90]}", flush=True)

    clean_ok = bool(results.get("clean", {}).get("ok"))
    pois_ok = bool(results.get("poisoned", {}).get("ok"))
    after_ok = bool(results.get("after", {}).get("ok"))
    out = {"steps": STEPS, "cases": results,
           "boundary_holds": bool(clean_ok and after_ok),
           "poison_reproduced": bool(clean_ok and not pois_ok)}
    if clean_ok and not pois_ok and after_ok:
        out["verdict"] = (
            "**BOUNDARY HOLDS AND THE POISON IS REAL.** FCEUX runs in a fresh process, breaks in a "
            "process that touched MPS, and runs again in a fresh process launched after an MPS process "
            "exited. Training on MPS in a dedicated process is safe provided that process never spawns "
            "the emulator.")
    elif clean_ok and pois_ok and after_ok:
        out["verdict"] = (
            "**BOUNDARY UNTESTED -- the poison did not reproduce.** FCEUX succeeded even in the process "
            "that used MPS, so this run does not demonstrate confinement; it demonstrates that the "
            "failure is not deterministic on this build. **Do not treat it as a pass.** The safe rule "
            "stands on the docstring's original measurement, not on this run.")
    elif not clean_ok:
        out["verdict"] = (
            "**INCONCLUSIVE -- the control failed.** FCEUX did not run even in a clean process, so "
            "nothing about MPS can be concluded. Fix the emulator path first.")
    else:
        out["verdict"] = (
            "**BOUNDARY DOES NOT HOLD.** FCEUX failed in a fresh process launched after an MPS process "
            "exited. Training on MPS is not separable from evaluation on this machine; §2 is not "
            "available and §4's arms must be costed on CPU.")
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
