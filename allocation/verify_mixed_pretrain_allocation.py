"""Pre-training allocation for scalar no-intercept OLS, NOT a CNN rule.

The selector receives noisy pilot observations only, constructs pseudo-labels,
and integrates the finite-sample OLS risk under the empirical pair law. It never
fits candidate predictors. All selections are persisted before independent
training repetitions used for evaluation. Existing CNN results are untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noise_models import sample_observation, make_pair, label_from_yz, second_moment_score

TAUS = np.array([.25, .5, 1., 2., 4., 8.])
PARAMS = dict(gamma=.25, sigma=.25, ell=4.)
MODELS = ("pg", "gg", "gp")


def csv_write(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def laplace_moments(z, target, ns, order=320, lo=-22., hi=20.):
    """E[b], E[b^2] for n iid draws from empirical (Z,T) distribution.

    Integrate s=exp(u)/scale. No candidate regressions are fitted here.
    Truncation/order comparisons are numerical diagnostics, not certified bounds.
    """
    z, target = np.ravel(z), np.ravel(target)
    zz, a = z*z, z*target
    nodes, weights = roots_legendre(order)
    u = (hi-lo)/2*nodes + (hi+lo)/2
    weights = weights*(hi-lo)/2
    scale = min(ns)*float(np.mean(zz))
    s = np.exp(u)/scale
    phi, first, second = [], [], []
    for chunk in np.array_split(s, max(1, (len(s)+31)//32)):
        e = np.exp(-chunk[:, None]*zz[None, :])
        phi.extend(np.mean(e, axis=1))
        first.extend(e @ a / len(z))
        second.extend(e @ (a*a) / len(z))
    phi, first, second = map(np.asarray, (phi, first, second))
    result = {}
    for n in ns:
        eb = np.sum(weights*s*n*first*phi**(n-1))
        eb2 = np.sum(weights*s*s*(n*second*phi**(n-1)
                                + n*(n-1)*first*first*phi**(n-2)))
        result[n] = (float(eb), float(eb2))
    return result


def risk_from_moments(eb, eb2, m2, sz):
    return float(m2 - 2*m2*eb + sz*eb2)


def clean_sample(shape, rng):
    # New scalar protocol: not image pixels and not the old CNN protocol.
    return rng.choice(np.array([.5, 1.5]), size=shape)


def exact_test_risk(b, model, tau):
    # Rao-Blackwellized independent test expectation, evaluation only.
    # E X=1, E X^2=1.25 for the declared two-point signal distribution.
    m2 = 1.25
    noise_power = dict(pg=.3125, gg=.375, gp=.5625)[model]
    return m2*(b-1)**2 + (noise_power+tau*tau)*b*b


def fit_for_evaluation(model, tau, n, reps, seed, quad):
    rng = np.random.default_rng(seed)
    x = clean_sample((reps, n), rng)
    y = sample_observation(x, model, rng, **PARAMS)
    z, target = make_pair(y, model, tau, rng, quad_order=quad, **PARAMS)
    b = np.sum(z*target, axis=1)/np.sum(z*z, axis=1)
    losses = exact_test_risk(b, model, tau)
    return b, losses


def mean_ci(values):
    values = np.asarray(values)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)/np.sqrt(len(values)))
    h = float(student_t.ppf(.975, len(values)-1)*se)
    return mean, se, mean-h, mean+h


def self_checks():
    # Closed-form Gaussian transform integrates back to the known exact risk.
    from scipy.integrate import quad
    rows = []
    for n, k, q, t in [(8, 1., 1., 1.215250), (32, 1., 1., .437851), (8, .35, 1., 4.)]:
        a = k+q+q*t
        b = k/a
        h = k*q*(1+t)/a + q*(1+1/t)
        phi = lambda s: (1+2*a*s)**(-.5)
        aa = lambda s: b*a*(1+2*a*s)**(-1.5)
        cc = lambda s: 3*b*b*a*a*(1+2*a*s)**(-2.5)+h*a*(1+2*a*s)**(-1.5)
        eb = quad(lambda s:n*aa(s)*phi(s)**(n-1), 0, np.inf, epsabs=1e-11)[0]
        eb2 = quad(lambda s:s*(n*cc(s)*phi(s)**(n-1)+n*(n-1)*aa(s)**2*phi(s)**(n-2)), 0, np.inf, epsabs=1e-11)[0]
        got = risk_from_moments(eb, eb2, k, a)
        r = k*q*(1+t)/a
        want = r+(r+q*(1+1/t))/(n-2)
        assert abs(got-want) < 1e-8
        rows.append(dict(n=n, k=k, q=q, t=t, integral_risk=got, closed_risk=want))
    # Exhaustive non-Gaussian two-point pair law, including signed ZT.
    import itertools
    z, target, n = np.array([-.7, 1.9]), np.array([1.1, .3]), 8
    bs = []
    for ix in itertools.product(range(2), repeat=n):
        ix = np.array(ix)
        bs.append(np.sum(z[ix]*target[ix])/np.sum(z[ix]**2))
    eb, eb2 = laplace_moments(z, target, [n], order=640)[n]
    assert abs(eb-np.mean(bs)) < 1e-8
    assert abs(eb2-np.mean(np.square(bs))) < 1e-8
    return rows


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    protocol = vars(args).copy()
    protocol.update(signal="iid scalar X in {0.5,1.5}, equal probabilities",
        noise_parameters=PARAMS, taus=TAUS.tolist(), ns=[8,32],
        selector="Laplace finite-n risk plug-in from noisy pilot pairs; no candidate fit",
        input="Z_tau", model="no-intercept, unregularized scalar OLS",
        test="analytic expectation over an independent test draw; MC over training",
        ci="approximate Student t over independent training repetitions, conditional on pilot",
        gg="finite quadrature; 24/48 order comparisons; no certified uniform bias bound",
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__),Path(__file__).with_name('noise_models.py')]})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (out/'self_checks.json').write_text(json.dumps(self_checks(),indent=2),encoding='utf-8')
    candidates, selections, audit = [], [], []
    for mi, model in enumerate(MODELS):
        for rep in range(args.pilots):
            rng = np.random.default_rng(args.seed+10000*mi+100*rep)
            y = sample_observation(clean_sample(args.pilot_size,rng),model,rng,**PARAMS)
            # Only y is passed into every selection calculation below.
            m2 = float(np.mean(second_moment_score(y,model,**PARAMS)))
            if m2 <= 0:
                raise ValueError('Nonpositive estimated signal energy; do not silently clip')
            current = []
            for ti,tau in enumerate(TAUS):
                prng = np.random.default_rng(args.seed+10000*mi+100*rep+ti+1)
                z,target = make_pair(y,model,tau,prng,quad_order=args.quad,**PARAMS)
                moments = laplace_moments(z,target,[8,32])
                refined = laplace_moments(z,target,[8,32],order=640,lo=-26,hi=24)
                if model == 'gg':
                    higher = label_from_yz(y,z,model,tau,quad_order=2*args.quad,**PARAMS)
                    moments_high = laplace_moments(z,higher,[8,32])
                sz = float(np.mean(y*y)+tau*tau)
                for n in [8,32]:
                    risk = risk_from_moments(*moments[n],m2,sz)
                    row = dict(model=model,pilot_rep=rep,n=n,tau=float(tau),
                        risk_plugin=risk,m2_plugin=m2,sz_plugin=sz,
                        laplace_refinement_delta=risk_from_moments(*refined[n],m2,sz)-risk,
                        gg_order_delta=(risk_from_moments(*moments_high[n],m2,sz)-risk) if model=='gg' else 0.)
                    current.append(row)
                    candidates.append(row)
            for n in [8,32]:
                rows = [r for r in current if r['n']==n]
                best = min(rows,key=lambda r:r['risk_plugin'])
                selections.append(dict(model=model,pilot_rep=rep,n=n,tau=best['tau'],
                    selected_risk_plugin=best['risk_plugin'], boundary=best['tau'] in [TAUS[0],TAUS[-1]]))
            print('pilot done',model,rep,flush=True)
    csv_write(out/'pilot_candidates.csv',candidates)
    # This immutable file is written BEFORE any candidate evaluation/training.
    lock = dict(selections=selections, decision='frozen; no test-based reselection')
    lock_bytes = json.dumps(lock,indent=2).encode()
    (out/'selection_lock.json').write_bytes(lock_bytes)
    lock_hash = hashlib.sha256(lock_bytes).hexdigest()
    summary, curves = [], []
    for mi,model in enumerate(MODELS):
        for n in [8,32]:
            # Independent diagnostic curves, never used to update the lock.
            for ti,tau in enumerate(TAUS):
                b,loss = fit_for_evaluation(model,tau,n,args.reps,
                    args.seed+1_000_000+10000*mi+100*n+ti,args.quad)
                mean,se,low,high=mean_ci(loss)
                curves.append(dict(model=model,n=n,tau=float(tau),risk=mean,mc_se=se,ci_low=low,ci_high=high))
                if model=='gg':
                    bh,lh=fit_for_evaluation(model,tau,n,args.reps,
                        args.seed+1_000_000+10000*mi+100*n+ti,2*args.quad)
                    dm,ds,dl,du=mean_ci(lh-loss)
                    audit.append(dict(model=model,n=n,tau=float(tau),risk_order_delta=dm,delta_mc_se=ds))
                print('curve done',model,n,tau,flush=True)
            for rep in range(args.pilots):
                selected=next(r for r in selections if r['model']==model and r['n']==n and r['pilot_rep']==rep)
                seed=args.seed+2_000_000+10000*mi+100*n+rep
                bs,ls=fit_for_evaluation(model,selected['tau'],n,args.reps,seed,args.quad)
                bb,lb=fit_for_evaluation(model,1.,n,args.reps,seed,args.quad)
                mean,se,low,high=mean_ci(ls)
                bm,bse,bl,bu=mean_ci(lb)
                dm,ds,dl,du=mean_ci(ls-lb)
                # Fixed reference dynamic range 1.0, NOT the earlier CNN range 1.75.
                summary.append(dict(model=model,n=n,pilot_rep=rep,tau=selected['tau'],
                    mse=mean,mc_se=se,ci_low=low,ci_high=high,
                    fixed_mse=bm,fixed_ci_low=bl,fixed_ci_high=bu,
                    mse_delta=dm,delta_ci_low=dl,delta_ci_high=du,
                    pooled_psnr=float(-10*np.log10(mean)),fixed_pooled_psnr=float(-10*np.log10(bm)),
                    psnr_gain=float(10*np.log10(bm/mean))))
    assert hashlib.sha256((out/'selection_lock.json').read_bytes()).hexdigest()==lock_hash
    csv_write(out/'confirmation.csv',summary)
    csv_write(out/'independent_risk_curves.csv',curves)
    csv_write(out/'gg_risk_order_check.csv',audit)
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for mi,model in enumerate(MODELS):
        for ni,n in enumerate([8,32]):
            ax=axs[ni,mi]
            cr=[r for r in curves if r['model']==model and r['n']==n]
            ax.errorbar(TAUS,[r['risk'] for r in cr],yerr=[1.96*r['mc_se'] for r in cr],label='Independent OLS risk',marker='o')
            for rep in range(args.pilots):
                pr=[r for r in candidates if r['model']==model and r['n']==n and r['pilot_rep']==rep]
                ax.plot(TAUS,[r['risk_plugin'] for r in pr],alpha=.55,ls='--',label='Noisy-pilot integral' if rep==0 else None)
            ax.set_xscale('log',base=2); ax.set_title(f'{model.upper()}, n={n}')
            ax.set_xlabel('tau'); ax.set_ylabel('Risk'); ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'risk_curves.png',dpi=180); plt.close(fig)
    (out/'complete.json').write_text(json.dumps(dict(lock_sha256=lock_hash,
        no_cnn_retraining=True,rows=len(summary),max_laplace_change=max(abs(r['laplace_refinement_delta']) for r in candidates)),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='output/experiments/mixed_pretrain_ols_20261011')
    p.add_argument('--seed',type=int,default=2026101117)
    p.add_argument('--pilot-size',type=int,default=8192)
    p.add_argument('--pilots',type=int,default=3)
    p.add_argument('--reps',type=int,default=2000)
    p.add_argument('--quad',type=int,default=24)
    main(p.parse_args())
