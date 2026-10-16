# Notation used in the paper and code

The notation cleanup in the manuscript does not change the experiments. Some
serialized field names are retained so that the released tables remain
compatible with the files produced during the reported runs.

| Paper notation | Meaning | Fields retained in code and result files |
|---|---|---|
| `R` | Restoration risk with respect to the clean signal | `clean_mse`, `val_clean_mse`, `clean_mean` |
| centered `Q` | Observable score `mean(f^2 - 2 f T)` | `val_q_mean`, `val_centered_mean`, `centered_mean`; earlier aliases `centered_R`, `q_r` |
| `J_stab` | Noisy training loss mean plus one standard deviation | `raw_J`, `val_raw_J`, `val_J`, `test_J` |
| `Q_MC,K` | Centered score matched to a deployed `K`-sample ensemble | `val_q_mc_mean`; earlier alias `centered_R_mc` |
| `K` | Number of deployment recorruptions | Some configuration names contain `MC_J` or `deployment_j` |
| `alpha` | GR2R split parameter | `alpha`; converted to `tau` where specified in the paper |

Renaming locked CSV and JSON keys would break compatibility without changing
the tensors, losses, fitted parameters, candidate selections, or reported
numbers. No retraining is required for this notation change.
