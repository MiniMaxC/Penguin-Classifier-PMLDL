# Verification results

Verified locally on **16 September 2026**, using Python 3.11, Docker Desktop's
Linux engine, and the dependencies pinned in this repository.

## Automated tests and clean setup

- **19 tests passed** in the working repository.
- A fresh local Git clone was created with its own new virtual environment.
  Installing `requirements.txt` and running `pipeline.py --once` succeeded
  without copying models, processed data, MLflow state, or a DVC cache.
- **19 tests passed again** in that independently installed clone.
- `pip check` reported no broken requirements.
- Tests verified missing-value and outlier handling, reproducible/disjoint
  splits, training-only scaling, saved-model and MLflow prediction equivalence,
  API input validation, app success/error messages, scheduler overrun/retry
  behavior, and a real DVC run where failed training prevented deployment.
- Starting another pipeline while the scheduler held its lock was correctly
  refused with exit code 1.

The dependency check exposed a DVC/pathspec incompatibility during setup.
`pathspec==0.12.1` is explicitly pinned to prevent it in fresh installations.
The runner also explicitly uses UTF-8 output for Windows terminals.

## Data and model

| Result | Value |
|---|---:|
| Raw records | 344 |
| Records removed for missing required values | 2 |
| Invalid measurement records removed | 0 |
| Duplicate records removed | 0 |
| Training outliers detected and removed | 0 |
| Training records | 273 |
| Test records | 69 |
| Test accuracy | 1.0 |
| Test macro-F1 | 1.0 |

The test split contains 30 Adelie, 14 Chinstrap, and 25 Gentoo records.
These scores describe this small, fixed holdout; they are not a claim about
arbitrary measurements or all penguin populations. Outlier removal was also
verified with synthetic extreme values in tests, without changing the raw CSV.

## Deployment and browser checks

- Both Compose services, `api` and `app`, reached **healthy** status.
- Real HTTP predictions correctly classified a sample of each of the three
  species; an invalid request returned **HTTP 422**.
- In the browser, the four-field Streamlit form returned **Adelie** for its
  defaults and displayed the deployed model's run ID.
- The MLflow UI displayed the `penguin-species` experiment, the completed run,
  accuracy and macro-F1, training parameters, and its logged model.
- The complete initial pipeline finished in **79.14 seconds**. The independent
  clean-clone pipeline finished in **73.20 seconds**, excluding package setup.

## Real five-minute scheduling check

Executed in the main working repository:

```powershell
.\.venv\Scripts\python.exe pipeline.py --interval-seconds 300 --max-runs 2
```

Both runs executed **prepare → train → deploy**, built/recreated both containers,
passed health checks, and verified that the API served the newly trained model.
Times below are UTC; add three hours for Moscow time.

| Run | Started (UTC) | Finished (UTC) | Duration | Exit code |
|---|---|---|---:|---:|
| 1 | 00:41:09.473 | 00:41:31.989 | 22.52 s | 0 |
| 2 | 00:46:09.475 | 00:46:32.034 | 22.56 s | 0 |

The starts were **300.002 seconds apart**. The model run IDs changed:

1. `dda8667764c249708417d1fb32796adf`
2. `1c2701e77df342d0a6ffafe8b35e5964`

After the second run, `/health` and the example `/predict` response both reported
`1c2701e77df342d0a6ffafe8b35e5964`. The bounded scheduler exited successfully
after its second run; it is not left retraining indefinitely.

Full local evidence remains in `.runtime/pipeline.log`, `.runtime/runs.jsonl`,
and `.runtime/deployment.json`. Those generated files are intentionally ignored
by Git. Start the scheduler again using the README command when demonstrating
automation to the TAs.
