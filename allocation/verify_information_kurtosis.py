"""Check Gaussian precision allocation and unconditional recovery-error kurtosis.

No CNN is trained. The Monte Carlo part fits genuine independent OLS training
pairs and integrates the Gaussian test sample analytically conditional on
each fitted coefficient. Fourth-moment standard errors require n>8 here.
"""
import json
from pathlib import Path
import numpy as np


def quantities(n, k, q, t):
    a = k + q + q*t
    r = k - k*k/a
    h = r + q*(1+1/t)
    risk = r + h/(n-2)
    fourth = 3*(r*r + 2*r*h/(n-2) + 3*h*h/((n-2)*(n-4)))
    kappa = 3 + 6*(n-1)/(n-4)*(h/((n-2)*risk))**2
    assert np.isclose(fourth/risk**2, kappa)
    return a, r, h, risk, fourth, kappa


def main():
    seed, repetitions, batch_size = 20261019, 100000, 2000
    rng = np.random.default_rng(seed)
    rows = []
    for n, k, q in [(16, 1., .25), (16, 1., 1.), (16, 1., 4.),
                    (64, 1., 1.)]:
        s = np.sqrt(n-1)
        rho_star = max(0., (s*k-q)/((s+1)*k))
        t_eval = (1-rho_star)/rho_star if rho_star > 0 else 32.
        a, r, h, risk, fourth, kappa = quantities(n,k,q,t_eval)
        risks, fourths = [], []
        for _ in range(repetitions//batch_size):
            shape = (batch_size,n)
            x = rng.normal(0,np.sqrt(k),shape)
            noise = rng.normal(0,np.sqrt(q),shape)
            m = rng.normal(0,np.sqrt(q),shape)
            z = x+noise+np.sqrt(t_eval)*m
            target = x+noise-m/np.sqrt(t_eval)
            bhat = (z*target).sum(1)/(z*z).sum(1)
            w = r+a*(bhat-k/a)**2
            risks.extend(w)
            fourths.extend(3*w*w)
        risks, fourths = np.asarray(risks), np.asarray(fourths)
        risk_se = risks.std(ddof=1)/np.sqrt(repetitions)
        fourth_se = fourths.std(ddof=1)/np.sqrt(repetitions)
        grid = np.geomspace(.001,1e4,1000)
        kurtoses = np.array([quantities(n,k,q,t)[-1] for t in grid])
        assert np.all(np.diff(kurtoses) < 0)
        rows.append(dict(n=n,k=k,q=q,snr=k/q,rho_star=rho_star,
                         t_evaluation=t_eval,risk=risk,mc_risk=float(risks.mean()),
                         mc_risk_se=float(risk_se),fourth=fourth,
                         mc_fourth=float(fourths.mean()),mc_fourth_se=float(fourth_se),
                         kappa=kappa,mc_kappa=float(fourths.mean()/risks.mean()**2),
                         risk_zscore=float((risks.mean()-risk)/risk_se),
                         fourth_zscore=float((fourths.mean()-fourth)/fourth_se)))
    path = Path("output/experiments/information_kurtosis_audit.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(seed=seed,repetitions=repetitions,
        scope="Gaussian scalar OLS only; no new CNN experiment",
        test_integration="Exact conditional E[e^2]=w and E[e^4]=3*w^2",
        rows=rows),indent=2),encoding="utf-8")
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
