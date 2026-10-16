"""Evaluate the C_tau-corrected allocation objective without retraining.

The script reads the frozen six-scale experiment, estimates C_tau on a fresh
Monte Carlo signal/noise stream, chooses every corrected objective using only
saved validation scores, and only then evaluates the selected models on a
fresh independent test stream.  Existing raw-J/centered/oracle test scores
are reused for comparison; no test score is read during selection.
"""
from __future__ import annotations
# LEGACY NOTATION: locked artifacts use ``J``, ``val_J``, and ``raw_J`` for the
# mean-plus-SD stability objective denoted by J_stab in the manuscript.  Keep
# these serialized field names unchanged so existing results remain readable.


import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from noise_models import make_pair, sample_observation
from run_allocation import (
    BoundedCNN, HIGH, LOW, MODELS, SPLITS, intervals, predict, sample_images,
    score_arrays, seed_for,
)

LAMBDAS = (0.0, 0.25, 0.5, 1.0)
OLD_METHODS = ("raw_J", "centered_mean", "fixed_tau_1", "clean_val_oracle")
METHOD_LABELS = {
    "raw_J": "raw J_stab", "centered_mean": "centered mean",
    "fixed_tau_1": "fixed tau=1", "clean_val_oracle": "clean oracle",
    "corr_lambda_0": "J_corr, lambda=0", "corr_lambda_0.25": "J_corr, lambda=0.25",
    "corr_lambda_0.5": "J_corr, lambda=0.5", "corr_lambda_1": "J_corr, lambda=1",
}


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quadrature_order(protocol, tau):
    return max(int(protocol["quad_order"]), 48) if tau <= 0.25 else int(protocol["quad_order"])


def estimate_c(protocol, seed=20261109, images_per_batch=128, batches=32):
    """Estimate per-pixel C_tau from independent image-level MC units.

    PG uses the exact conditional second-moment formula and only Monte Carlo
    integrates over the clean-image distribution.  GG/GP use independent
    Y,Z,T draws and the exact-label quadrature already specified by the
    frozen protocol.  Standard errors are computed from independent image
    means, not individual correlated pixels.
    """
    params={k: protocol[k] for k in ("gamma", "sigma", "ell")}
    accum = {(model, tau): [] for model in protocol["models"] for tau in protocol["taus"]}
    for rep in range(batches):
        for model in protocol["models"]:
            x = sample_images(
                images_per_batch, int(protocol["size"]),
                np.random.default_rng(seed_for(seed, rep, 11, MODELS[model])),
            )
            if model == "pg":
                for tau in protocol["taus"]:
                    tau2 = float(tau) ** 2
                    a = params["sigma"] ** 2 * params["gamma"] / tau2
                    et2 = np.exp(params["gamma"] ** 2 / tau2) * (
                        (x + a) ** 2 + params["gamma"] * x + params["sigma"] ** 2
                        + params["sigma"] ** 4 / tau2
                    )
                    accum[model, tau].extend(np.mean(et2 - x ** 2, axis=(1, 2, 3)).tolist())
            else:
                y = sample_observation(
                    x, model,
                    np.random.default_rng(seed_for(seed, rep, 101, MODELS[model])),
                    **params,
                )
                for tau in protocol["taus"]:
                    rng = np.random.default_rng(seed_for(seed, rep, 201 + int(round(100 * tau)), MODELS[model]))
                    _, t = make_pair(y, model, float(tau), rng,
                                     quad_order=quadrature_order(protocol, float(tau)), **params)
                    accum[model, tau].extend(np.mean((t - x) ** 2, axis=(1, 2, 3)).tolist())
    rows=[]
    for model in protocol["models"]:
        for tau in protocol["taus"]:
            values=np.asarray(accum[model, tau], dtype=float)
            mean=float(values.mean())
            se=float(values.std(ddof=1)/np.sqrt(len(values)))
            rows.append({"model":model,"tau":float(tau),"C_hat":mean,"mc_se":se,
                         "mc_low_95":mean-1.96*se,"mc_high_95":mean+1.96*se,
                         "mc_images":int(len(values)),"mc_batches":int(batches),
                         "method":"PG exact conditional second moment" if model=="pg" else "independent MC exact-label quadrature"})
    return rows


def validation_candidates(root, protocol, c_rows):
    c={(r["model"], float(r["tau"])): r["C_hat"] for r in c_rows}
    rows=[]
    for model in protocol["models"]:
        for rep in range(int(protocol["reps"])):
            for tau in protocol["taus"]:
                path=root/"scores"/f"{model}_r{rep}_tau{float(tau):g}_val.npz"
                with np.load(path) as z: raw=np.asarray(z["raw"],dtype=float)
                mean=float(raw.mean()); sd=float(raw.std(ddof=1)); ct=c[model,float(tau)]
                row={"model":model,"rep":rep,"tau":float(tau),"C_hat":ct,
                     "val_mean_L":mean,"val_sd_L":sd,"val_J":mean+sd,
                     "val_centered":mean-ct}
                for lam in LAMBDAS:
                    row[f"val_corr_{lam:g}"]=mean-ct+lam*sd
                rows.append(row)
    return rows


def choose(validation_rows, protocol):
    selections=[]
    for model in protocol["models"]:
        for rep in range(int(protocol["reps"])):
            local=[r for r in validation_rows if r["model"]==model and int(r["rep"])==rep]
            for lam in LAMBDAS:
                key=f"val_corr_{lam:g}"
                best=min(local,key=lambda r:float(r[key]))
                selections.append({"model":model,"rep":rep,"method":f"corr_lambda_{lam:g}",
                                   "lambda":lam,"tau":float(best["tau"]),
                                   "val_objective":float(best[key]),"boundary":float(best["tau"]) in (min(protocol["taus"]),max(protocol["taus"]))})
    return selections


def evaluate_new_methods(root, protocol, selections):
    params={k: protocol[k] for k in ("gamma", "sigma", "ell")}
    out_rows=[]; score_dir=root/"corrected_scores"; score_dir.mkdir(exist_ok=True)
    for model in protocol["models"]:
        model_id=MODELS[model]
        for rep in range(int(protocol["reps"])):
            xt=sample_images(int(protocol["test_n"]),int(protocol["size"]),np.random.default_rng(seed_for(protocol["seed"],rep,SPLITS["test"])))
            yt=sample_observation(xt,model,np.random.default_rng(seed_for(protocol["seed"],rep,SPLITS["test"]+100,model_id)),**params)
            local=[r for r in selections if r["model"]==model and int(r["rep"])==rep]
            for tau in sorted(set(float(r["tau"]) for r in local)):
                rng=np.random.default_rng(seed_for(protocol["seed"],rep,SPLITS["test"]+200,model_id))
                zt,tt=make_pair(yt,model,tau,rng,quad_order=quadrature_order(protocol,tau),**params)
                net=BoundedCNN()
                cp=root/"checkpoints"/f"{model}_r{rep}_tau{tau:g}.pt"
                if not cp.exists(): raise FileNotFoundError(cp)
                net.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True))
                sc=score_arrays(predict(net,zt),xt,yt,tt,model,params)
                np.savez_compressed(score_dir/f"{model}_r{rep}_tau{tau:g}_test.npz",**sc)
                for method in [r["method"] for r in local if float(r["tau"])==tau]:
                    a=sc["clean"]
                    out_rows.append({"model":model,"rep":rep,"method":method,"tau":tau,
                                     "test_n":int(len(a)),"clean_mean":float(a.mean()),
                                     "clean_sd":float(a.std(ddof=1)),"test_raw_mean":float(sc["raw"].mean()),
                                     "test_raw_sd":float(sc["raw"].std(ddof=1)),
                                     "test_J":float(sc["raw"].mean()+sc["raw"].std(ddof=1))})
    return out_rows


def load_new_results(root, selections):
    """Rebuild per-repeat corrected test rows from saved corrected NPZ files."""
    rows=[]
    for r in selections:
        model, rep, tau = r["model"], int(r["rep"]), float(r["tau"])
        path=root/"corrected_scores"/f"{model}_r{rep}_tau{tau:g}_test.npz"
        with np.load(path) as z:
            clean=np.asarray(z["clean"],dtype=float); raw=np.asarray(z["raw"],dtype=float)
        rows.append({"model":model,"rep":rep,"method":r["method"],"tau":tau,
                     "test_n":len(clean),"clean_mean":float(clean.mean()),
                     "clean_sd":float(clean.std(ddof=1)),"test_raw_mean":float(raw.mean()),
                     "test_raw_sd":float(raw.std(ddof=1)),"test_J":float(raw.mean()+raw.std(ddof=1))})
    return rows


def old_method_results(root, protocol):
    tests=read_rows(root/"test_results.csv"); rows=[]
    for r in tests:
        if r["method"] not in OLD_METHODS: continue
        path=root/"scores"/f"{r['model']}_r{int(r['rep'])}_tau{float(r['tau']):g}_test.npz"
        with np.load(path) as z: a=np.asarray(z["clean"],dtype=float)
        rows.append({"model":r["model"],"rep":int(r["rep"]),"method":r["method"],"tau":float(r["tau"]),
                     "test_n":len(a),"clean_mean":float(a.mean()),"clean_sd":float(a.std(ddof=1)),
                     "test_raw_mean":float(r["test_raw_mean"]),"test_raw_sd":float(r["test_raw_sd"]),"test_J":float(r["test_J"])})
    return rows


def validation_objectives(root, protocol, validation_rows, selections):
    """Return validation objective summaries for old and corrected methods."""
    candidates = read_rows(root / "candidates.csv")
    out = {}
    for model in protocol["models"]:
        for method in OLD_METHODS:
            values = []
            for rep in range(int(protocol["reps"])):
                sel = next(r for r in read_rows(root / "selections.csv")
                           if r["model"] == model and int(r["rep"]) == rep and r["method"] == method)
                cand = next(r for r in candidates if r["model"] == model and int(r["rep"]) == rep
                            and float(r["tau"]) == float(sel["tau"]))
                key = {"raw_J": "val_J", "centered_mean": "val_centered_mean",
                       "fixed_tau_1": "val_J", "clean_val_oracle": "val_clean_mse_diagnostic"}[method]
                values.append(float(cand[key]))
            out[model, method] = (float(np.mean(values)), float(np.std(values, ddof=1)))
        for lam in LAMBDAS:
            method = f"corr_lambda_{lam:g}"
            values = [float(r["val_objective"]) for r in selections
                      if r["model"] == model and r["method"] == method]
            out[model, method] = (float(np.mean(values)), float(np.std(values, ddof=1)))
    return out


def summarize_results(root, protocol, all_results, val_summary):
    rows=[]
    for model in protocol["models"]:
        for method in sorted({r["method"] for r in all_results if r["model"]==model}):
            rr=sorted([r for r in all_results if r["model"]==model and r["method"]==method],key=lambda r:r["rep"])
            arrays=[]
            for r in rr:
                if method.startswith("corr_lambda_"):
                    path=root/"corrected_scores"/f"{model}_r{r['rep']}_tau{r['tau']:g}_test.npz"
                else:
                    path=root/"scores"/f"{model}_r{r['rep']}_tau{r['tau']:g}_test.npz"
                with np.load(path) as z: arrays.append(np.asarray(z["clean"],dtype=float))
            joined=np.concatenate(arrays); ci=intervals(joined,bounded=True)
            ps=np.concatenate([10*np.log10((HIGH-LOW)**2/a) for a in arrays])
            rows.append({"model":model,"method":method,"label":METHOD_LABELS[method],
                         "taus":json.dumps([r["tau"] for r in rr]),
                         "val_objective_mean":val_summary[model,method][0],
                         "val_objective_rep_sd":val_summary[model,method][1],
                         "test_mse":float(joined.mean()),"mse_se":float(joined.std(ddof=1)/np.sqrt(len(joined))),
                         "mse_t_low":float(ci["t_low"]),"mse_t_high":float(ci["t_high"]),
                         "mse_bernstein_low":float(ci["bernstein_low"]),"mse_bernstein_high":float(ci["bernstein_high"]),
                         "mean_image_psnr":float(ps.mean()),"pooled_psnr":float(10*np.log10((HIGH-LOW)**2/joined.mean())),
                         "test_J":float(np.mean([r["test_J"] for r in rr]))})
    for row in rows:
        raw=next(x for x in rows if x["model"]==row["model"] and x["method"]=="raw_J")
        row["psnr_gain_vs_raw_J"]=row["mean_image_psnr"]-raw["mean_image_psnr"]
    write_rows(root/"corrected_results.csv",rows)
    (root/"corrected_results.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")
    return rows


def main(args):
    root=Path(args.run).resolve(); protocol=json.loads((root/"protocol.json").read_text(encoding="utf-8"))
    if args.reuse_existing:
        c_rows=json.loads((root/"corrected_C_tau.json").read_text(encoding="utf-8"))
        val=read_rows(root/"corrected_validation_candidates.csv")
        selections=read_rows(root/"corrected_selections.csv")
        for r in selections:
            r["rep"]=int(r["rep"]); r["tau"]=float(r["tau"]); r["val_objective"]=float(r["val_objective"])
        new_results=load_new_results(root,selections)
    else:
        c_rows=estimate_c(protocol,seed=args.mc_seed,images_per_batch=args.mc_images_per_batch,batches=args.mc_batches)
        write_rows(root/"corrected_C_tau.csv",c_rows); (root/"corrected_C_tau.json").write_text(json.dumps(c_rows,indent=2,ensure_ascii=False),encoding="utf-8")
        val=validation_candidates(root,protocol,c_rows); write_rows(root/"corrected_validation_candidates.csv",val)
        selections=choose(val,protocol); write_rows(root/"corrected_selections.csv",selections)
        new_results=evaluate_new_methods(root,protocol,selections)
    all_results=old_method_results(root,protocol)+new_results
    val_summary=validation_objectives(root,protocol,val,selections)
    results=summarize_results(root,protocol,all_results,val_summary)
    print(json.dumps({"C_rows":len(c_rows),"validation_rows":len(val),"selections":len(selections),"test_rows_new":len(new_results),"summary_rows":len(results),"mc_seed":args.mc_seed},ensure_ascii=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",default="output/experiments/allocation_main_20261008")
    ap.add_argument("--mc-seed",type=int,default=20261109)
    ap.add_argument("--mc-images-per-batch",type=int,default=128)
    ap.add_argument("--mc-batches",type=int,default=32)
    ap.add_argument("--reuse-existing",action="store_true",
                    help="reuse saved C_tau, validation choices and corrected test scores")
    main(ap.parse_args())
