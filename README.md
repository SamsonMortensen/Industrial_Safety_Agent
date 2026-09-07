# Industrial Safety & Compliance Auditor

A local, retrieval-grounded safety auditor for intermodal yard operations. It evaluates operational observations against federal safety authorities, verifies citations, and sends unsupported or ambiguous cases to review.

The project combines structured safety rules, local language models, authentic incident narratives, deterministic controls, and a camera-neutral observation interface. Operational data and model inference remain local.

## Start with the notebook

I want someone new to this project to be able to open it, try an incident, and see why it returned that result. [main.ipynb](main.ipynb) walks through the actual backend, with explanations beside the code. The deeper scripts are still available below.

The default notebook runs on the CPU. It does not need a GPU, Ollama, a camera, a model download, or an account. It does not train or change learning memory. You need internet access for the initial repository and package downloads. After setup, the walkthrough runs locally without model requests.

### 1. Check Git and Python

Install [Git](https://git-scm.com/downloads) and [Python](https://www.python.org/downloads/) if needed, then open a new PowerShell window. Run:

```powershell
git --version
python --version
```

Both commands should print a version. Use Python 3.11 or newer; automated checks cover Python 3.11 on Linux and 3.14 on Windows. If Windows does not recognize `python`, try `py --version`. If that works, use `py` for the environment-creation command below. Otherwise, finish installing Python and reopen PowerShell before continuing.

### 2. Install and open the notebook

Choose a folder where you keep projects. Run these commands one line at a time in PowerShell, not inside a notebook cell. Stop if a command reports an error. The first two commands create and enter the repository folder.

```powershell
git clone https://github.com/SamsonMortensen/Industrial_Safety_Agent.git
cd Industrial_Safety_Agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name industrial-safety --display-name "Industrial Safety (.venv)"
.\.venv\Scripts\python.exe -m notebook main.ipynb --ip=127.0.0.1
```

Already have an unchanged public clone? Open PowerShell in that repository directory and skip the clone and `cd` commands. If the folder contains additional development files, use a separate fresh clone to reproduce the public walkthrough. Creating `.venv` does not remove or separate those files.

The package installation may take a few minutes. `Requirement already satisfied` is normal. `pip check` should report `No broken requirements found`. The kernel command connects this notebook to the same Python environment that received the packages; no environment activation is needed.

On Linux/macOS, create the environment with `python3 -m venv .venv`, then replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` in the remaining commands.

### 3. Run the cells and try an incident

The notebook opens in your browser, served by your own computer. If it does not open automatically, use the local URL printed in the terminal. Keep that terminal open while working, and do not share its access token.

Select the **Industrial Safety (.venv)** kernel. Click the first code cell and press **Shift+Enter**, then continue down the notebook. **Run > Run All Cells** also works. The first cell should print `Setup ready`; later cells show example decisions, retrieved text, an editable incident, a small live benchmark, and clearly labeled earlier results. An asterisk beside a cell means it is running.

You do not need to run every script in this README to try the project. The notebook is the beginner path. Optional local models and training are separate workflows under [Advanced scripts and validation](#advanced-scripts-and-validation).

Jupyter can save your edits and outputs in the notebook. Avoid personal information, and clear your incident text and outputs before sharing a copy. To finish, shut down the kernel, then press **Ctrl+C** in the server terminal and confirm if asked.

### If something stops

- **`git` or `python` is not recognized:** finish installing it, reopen PowerShell, and repeat the version checks.
- **The destination folder already exists:** enter the existing public clone, or choose a different parent folder for a fresh clone. Do not delete your development folder.
- **`requirements.txt` or `main.ipynb` is not found:** check that the terminal is in the repository folder.
- **A notebook reports a missing module:** install `requirements.txt` with the exact project Python command above, repeat the kernel registration, and restart the notebook with **Industrial Safety (.venv)** selected. Bare `pip` may install into a different environment.
- **Local tests ask for `torch` or `streamlit`:** the published notebook and public tests do not require them. Check whether the folder also contains private vision code, and use a fresh public clone for this walkthrough.
- **A cell refers to a name that is not defined:** restart the kernel and run the cells from the top.
- **You only installed `requirements-showcase.txt`:** that installs the CLI-only dependencies. Install `requirements.txt` before opening the notebook or running the full public validation checks.

## Prefer the terminal?

I want people to be able to see what this does without setting up a training machine. The walkthrough runs the actual backend with a smaller workload. It does not replay canned decisions.

It works through six different observations, shows what triggered each investigation, builds the queries, retrieves authorities, and returns a decision or sends the case to review. It also checks the citation guard, demonstrates the learning approval rules, and runs a 16-case retrieval benchmark across 16 hazard families.

You need Git and Python 3.11 or newer. No GPU, Ollama server, model download, or account is needed for the default run. From PowerShell:

```powershell
git clone https://github.com/SamsonMortensen/Industrial_Safety_Agent.git
cd Industrial_Safety_Agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-showcase.txt
.\.venv\Scripts\python.exe code/showcase.py
```

Already cloned? Start from the repository directory and skip the first two commands. On Linux/macOS, use `python3 -m venv .venv` and `.venv/bin/python` for the remaining commands.

The default run uses CPU-based lexical retrieval and the existing applicability and policy checks. Live results are labeled separately from the saved full-audit, hybrid-retrieval, incident-coverage, and QLoRA results. Those larger runs are not repeated by the walkthrough. The camera examples are structured observations, not live image recognition.

Try your own incident or keep the session open:

```powershell
.\.venv\Scripts\python.exe code/showcase.py --text "A wire-rope sling with broken outer wires remained in an active lift"
.\.venv\Scripts\python.exe code/showcase.py --interactive
```

For structured input, detailed evidence, or a larger sample:

```powershell
.\.venv\Scripts\python.exe code/showcase.py --input data/example_camera_observation.json --json
.\.venv\Scripts\python.exe code/showcase.py --benchmark-cases 64
.\.venv\Scripts\python.exe code/showcase.py --out benchmark_runs/showcase/my_first_run.json
```

The tour handles up to six observations and 64 benchmark cases per run. Use `--benchmark-cases 0` to skip the benchmark. It does not write learning memory, train models, or run anything in the background. Saving a report is optional; exports stay under the ignored showcase directory and cannot overwrite an existing file. To save another run, choose a new filename such as `my_second_run.json`. The interactive prompt is always offline and ends with `quit`.

If Ollama and a local model are already installed, you can allow one model request for a case the policy cannot resolve:

```powershell
.\.venv\Scripts\python.exe code/showcase.py --reasoner --model qwen3.5:9b --timeout 30
```

That request uses lexical retrieval, the existing citation guard, a limited context, and a 220-token generation limit. It does not build dense embeddings or download a model. The timeout is a network read timeout, not a guarantee of total runtime. A missing or unavailable model leaves the case in review. Smaller installed models can be selected with `--model`, but their output quality is not established by the saved QLoRA results.

This is a research system. A retrieved citation is not proof of a violation, and a clear result applies only to the condition checked. The full training and evaluation tools are still available below.

## What it does

- Evaluates text logs, incident narratives, camera observations, and sensor-derived events.
- Retrieves relevant OSHA, FRA, and statutory authorities with lexical and semantic search.
- Separates hazard detection, citation existence, citation grounding, and citation correctness.
- Applies deterministic controls when the legal scope and measurable facts are explicit.
- Requires every violation citation to appear in the retrieved candidate set.
- Returns `REVIEW` when the evidence or governing authority is uncertain.
- Stores only labeled or reviewer-approved outcomes for controlled future learning.

## Current capabilities

The current backend supports:

- walking-working surfaces and housekeeping;
- materials handling, powered industrial trucks, cranes, dockboards, and slings;
- electrical work practices and energized-part approach conditions;
- fall protection, ladders, stairways, and guardrails;
- freight train-employee duty limits under 49 U.S.C. 21103;
- selected FRA recordkeeping and passenger-service rules;
- stateful checks for cumulative duty time and recurring equipment defects;
- autonomous observation triggers based on change, anomaly, novelty, uncertainty, relationships, and human review flags.

## Measured results

### Full 1,000-event audit

The locked synthetic yard log contains 1,000 events, including 144 violations across 16 hazard families and a matched compliant counterpart for every planted violation.

| Metric | Grounded v1 | Raw v2 | Policy-controlled v2 |
| --- | ---: | ---: | ---: |
| True positives | 103 | 82 | 98 |
| False positives | 4 | 1 | 0 |
| False negatives | 41 | 62 | 46 |
| True negatives | 852 | 855 | 856 |
| Precision | 0.9626 | 0.9880 | **1.0000** |
| Recall | **0.7153** | 0.5694 | 0.6806 |
| F1 | **0.8207** | 0.7225 | 0.8099 |
| Accuracy | **0.9550** | 0.9370 | 0.9540 |
| Correct citation among true positives | 0.5728 | 0.8171 | **0.8571** |
| Status and citation jointly correct | 0.9110 | 0.9220 | **0.9400** |

Grounded v1 and v2 are separate model runs. Raw v2 and policy-controlled v2 are two evaluations of the same saved v2 outputs. The policy-controlled result used deterministic scope rules for 90 cases and retained the model result for the remaining 910.

Saved evidence:

- `json/full_audit_results.json`
- `json/full_audit_policy_controlled.json`
- `json/audit_results.json`

### Autonomous authority retrieval

This benchmark asks whether the governing authority can be retrieved from observable event fields. Labels are used only after retrieval for scoring.

| Retrieval method | Recall@1 | Recall@4 | Recall@8 | Recall@16 |
| --- | ---: | ---: | ---: | ---: |
| Multi-query hybrid fusion | **0.4028** | **0.7986** | **0.9861** | **1.0000** |
| Hybrid with applicability reranking | 0.3125 | 0.7708 | 0.9236 | **1.0000** |

Both methods retrieved the expected authority within the top 16 for all 144 violations in the saved run. Raw fusion ranked the expected authority earlier more often. These figures are computed from the per-case ranks in `json/autonomous_retrieval_benchmark.json`; they are not a field-performance claim.

### Fresh QLoRA result

A 4-bit QLoRA supervised fine-tune of `Qwen/Qwen2.5-3B-Instruct` used 70 synthetic training cases and a text-disjoint 56-case holdout. The trainer did not open the holdout.

| Matched holdout result | Base model | QLoRA adapter |
| --- | ---: | ---: |
| Cases | 56 | 56 |
| Strict verdict accuracy | 0.0000 | **0.8929** |
| F1 | 0.0000 | **0.8929** |
| Violation citation correctness | 0.0357 | **0.5714** |
| Verdict and citation both correct | 0.0000 | **0.7321** |
| Parsed in the required format | 0 of 56 | **56 of 56** |

The adapter is suitable for controlled narration and structured output experiments. Its citation correctness is not sufficient for autonomous legal decisions. Exact configuration, hashes, and per-case outputs are saved in:

- `json/finetune_training_report.json`
- `json/finetune_evaluation_results.json`
- `json/scaled_benchmark_hybrid_results.json`

### Authentic incident-text coverage

The OSHA Severe Injury Report pipeline profiled 105,996 records dated January 1, 2015 through November 30, 2025. Structured employer and location identifiers are removed before local normalization. Free-text narratives remain in ignored local storage and are not committed.

| Authentic OSHA stress test | Result |
| --- | ---: |
| Records sampled | 500 |
| Triggered for investigation | 500 |
| Non-empty top-16 candidate set | 500 |
| Distinct top-ranked authorities | 24 |
| Cases sent to review by the narrow policy | 498 |
| Policy citations supported by retrieval | 1 of 2 |
| Unsupported citations allowed | **0** |

Every sampled row is a known reported incident and is intentionally human-flagged. These figures measure ingestion, retrieval coverage, and citation gating. They do not measure pre-incident detection or legal accuracy.

Saved evidence:

- `json/osha_sir_profile.json`
- `json/authentic_incident_retrieval_hybrid.json`
- `data/incident_text_datasets.json`

## Safety architecture

```text
Observation -> Trigger -> Query plan -> Hybrid retrieval -> Applicability check
            -> Grounded policy or local reasoner -> Decision or REVIEW
```

1. `code/autonomous_investigator.py` accepts camera-neutral structured observations.
2. `code/fetch_regulations.py` and `code/build_section_index.py` build the legal corpus and section verifier.
3. `code/benchmark_autonomous_retrieval.py` evaluates authority retrieval independently from generation.
4. `code/hybrid_policy.py` handles high-confidence cases with explicit scope and facts.
5. `code/audit_agent.py` and `code/audit_agent_v2.py` run the grounded model audits.
6. `code/continuous_learner.py` stores trusted precedents and quarantines unverified outcomes.
7. `code/stateful_tracker.py` evaluates selected cross-event conditions.

## Continuous improvement

Continuous learning is controlled rather than self-modifying:

- only labeled or reviewer-approved decisions can enter trusted memory;
- an event cannot retrieve itself or a near-duplicate as a precedent;
- only earlier events are eligible as precedents;
- candidate training data remains separate from locked evaluation data;
- model weights are updated offline and promoted only after evaluation;
- the running model never trains directly on its own unreviewed predictions.

`code/temporal_eval.py` provides a frozen future-slice evaluation for measuring whether learned memory transfers to unseen events.

## Vision integration

The backend already accepts structured observations from a future camera or sensor pipeline. The vision layer remains a separate training and validation project.

- `data/vision_datasets.json` records dataset licenses and intended uses.
- `code/vision_manifest.py` creates group-level train, validation, and test splits.
- `data/example_camera_observation.json` demonstrates the observation contract.

Public dataset labels are treated as observations, not legal conclusions. Camera calibration, tracking, occlusion, privacy controls, local field labels, and external validation remain required before deployment.

## Regulatory and benchmark scope

The retrievable corpus includes selected authorities from:

- 29 CFR 1910 Subpart D, walking-working surfaces;
- 29 CFR 1910 Subpart N, materials handling and storage;
- 29 CFR 1910 Subpart S, electrical safety;
- 49 CFR 228, railroad hours-of-service recordkeeping and applicable passenger rules;
- 49 U.S.C. 21103, freight train-employee hours-of-service limits.

The current synthetic benchmark covers 16 hazard families:

| Area | Hazard families | Governing sections |
| --- | --- | --- |
| Walking-working surfaces | housekeeping, ladders, stairways, dockboards, fall protection, guardrails | 1910.22, .23, .25, .26, .28, .29 |
| Materials handling | aisle obstruction, rim wheels, lift trucks, cranes, slings | 1910.176, .177, .178, .179, .184 |
| Electrical | exposed live parts, approach distance, protective equipment | 1910.303, .333, .335 |
| Hours of service | freight train-employee duty limit, duty records | 49 U.S.C. 21103, 49 CFR 228.11 |

## Advanced scripts and validation

The notebook setup already installs `requirements.txt`, which also covers the public analysis and validation commands below. The CLI-only walkthrough needs only `requirements-showcase.txt`. Model training is optional and has separate requirements; the notebook does not start it.

### Prerequisites

- Git and Python 3.11 or compatible
- [Ollama](https://ollama.com) for live embedding and language-model runs; not required for saved-result validation or lexical-only examples

Commands below use Windows PowerShell from the repository root. The explicit Python paths avoid environment-activation and interpreter mismatches. On Linux/macOS, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

```powershell
git clone https://github.com/SamsonMortensen/Industrial_Safety_Agent.git
cd Industrial_Safety_Agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe code/check_setup.py
```

If the repository is already cloned, start from its directory and skip the first two commands. Virtual environments and trained adapters are not included in a clone.

### Validate the repository

Run these after installing `requirements.txt` with the project Python. These checks use the included reference files, require no GPU or model downloads, and do not rerun the 500-case benchmark. Notebook validation runs every code cell in a fresh kernel using the same Python interpreter. It blocks outbound socket and HTTP requests in the walkthrough and does not save the executed notebook.

```powershell
.\.venv\Scripts\python.exe -m compileall -q code tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe code/validate_notebook.py
.\.venv\Scripts\python.exe code/validate_saved_evaluation.py
```

Expected: tests finish with `OK`, notebook validation confirms that all code cells ran, and saved-evaluation validation confirms matching metrics and the holdout hash. The saved-evaluation validator does not verify model weights. These are public-repository checks, not validation of additional private development files.

### Try an offline observation

```powershell
.\.venv\Scripts\python.exe code/run_autonomous_investigation.py data/example_camera_observation.json --sample-rate 0 --lexical-only
.\.venv\Scripts\python.exe code/benchmark_autonomous_retrieval.py --lexical-only --out benchmark_runs/autonomous_lexical.json
```

The observation command prints a structured decision with retrieved candidates. Lexical-only results are a smoke test, not a reproduction of the published hybrid scores. New example outputs go under ignored `benchmark_runs/`; rerunning a named output replaces that local result.

### Enable local models

Start the Ollama desktop app, or run `ollama serve` in another terminal if no server is already running. Then install the models once:

```powershell
ollama pull mxbai-embed-large
ollama pull qwen3.5:9b
.\.venv\Scripts\python.exe code/check_setup.py --ollama --embed
```

`OLLAMA_HOST` defaults to `http://localhost:11434`; an address such as `127.0.0.1:11434` is also accepted. `ollama ps` shows CPU/GPU placement while a model is loaded. Ollama execution and PyTorch CUDA are separate configurations.

### Run hybrid retrieval and small model audits

These commands make live model requests. The first dense run builds corpus embeddings and can take substantially longer than the offline checks. Start audits at one event and concurrency one; increase `--limit` to 50 or 1000 after checking runtime and memory.

```powershell
.\.venv\Scripts\python.exe code/run_autonomous_investigation.py data/example_camera_observation.json --sample-rate 0 --out benchmark_runs/observation_hybrid.json
.\.venv\Scripts\python.exe code/retrieval_diagnostic.py --out benchmark_runs/retrieval_diagnostic.json
.\.venv\Scripts\python.exe code/benchmark_autonomous_retrieval.py --out benchmark_runs/autonomous_hybrid.json
.\.venv\Scripts\python.exe code/audit_agent.py --limit 1 --concurrency 1 --out benchmark_runs/audit_v1.json
.\.venv\Scripts\python.exe code/audit_agent_v2.py --limit 1 --concurrency 1 --out benchmark_runs/audit_v2.json
.\.venv\Scripts\python.exe code/apply_policy_controls.py --raw benchmark_runs/audit_v2.json --out benchmark_runs/audit_v2_controlled.json
```

The observation example uses policy decisions unless `--reasoner` is added. The v2 auditor also writes ignored local learning memory and run history. Small runs check the execution path; they do not establish model quality or reproduce the full-audit results.

### Prepare authentic incident text

This is optional and requires a separate external download. The raw CSV and normalized incident file are not included in a clone. The beginner notebook reads the saved coverage report and does not need either file.

First download and extract the source:

```powershell
.\.venv\Scripts\python.exe code/fetch_osha_incidents.py
```

Do not continue until extraction succeeds and `data/external/osha_sir/January2015toNovember2025.csv` exists. If the download fails, use the manual-download instructions below. Then normalize the CSV:

```powershell
.\.venv\Scripts\python.exe code/osha_incident_pipeline.py --profile benchmark_runs/osha_sir_profile.json
```

After that command succeeds, `data/external/osha_sir/normalized_observations.jsonl` is available for the benchmark:

```powershell
.\.venv\Scripts\python.exe code/benchmark_authentic_incidents.py --sample 50
```

If OSHA refuses the automated download with HTTP 403, download the matching archive from the [OSHA Severe Injury Reports page](https://www.osha.gov/severe-injury-reports), save it in `data/external/osha_sir/`, and rerun the fetch command without `--force`. For a different release, supply its official `--url` to the fetch command and the extracted CSV through the pipeline's `--input` option. Do not describe a different release as the published reference dataset.

Add `--dense` to test hybrid retrieval once Ollama is ready; use `--sample 500` for a larger run. Dense runs also embed the regulation corpus, so reducing the sample does not remove that startup cost. Each authentic-incident benchmark saves a timestamped result under `benchmark_runs/authentic_incidents/` and refuses to overwrite published reference results.

### Rebuild source fixtures (optional)

The corpus, section index, and benchmark log are already included. Run these only in a separate experimental checkout: they replace those files, and changed corpus content requires rebuilding the embedding cache before further dense runs. The eCFR edition is pinned to January 1, 2025, not the current law.

```powershell
.\.venv\Scripts\python.exe code/fetch_regulations.py
.\.venv\Scripts\python.exe code/build_section_index.py
.\.venv\Scripts\python.exe code/generate_yard_log.py --records 1000
```

### Fine-tuning

Fine-tuning is optional, GPU-intensive, and separate from repository validation. The historical `train_lora_dpo.py` filename currently runs QLoRA supervised fine-tuning, not DPO reinforcement learning.

Create a separate environment, then install a CUDA-enabled PyTorch build compatible with your Python, NVIDIA GPU, and driver using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Run its installation command with `.\.venv-train\Scripts\python.exe -m pip` instead of bare `pip`. Installing the CUDA Toolkit alone does not enable CUDA in a CPU-only PyTorch build.

```powershell
python -m venv .venv-train
```

After installing CUDA-enabled PyTorch in that environment:

```powershell
.\.venv-train\Scripts\python.exe -m pip install -r requirements.txt -r requirements-train.txt
.\.venv-train\Scripts\python.exe -m pip check
.\.venv-train\Scripts\python.exe code/train_lora_dpo.py --preflight-only
```

The preflight must report a CUDA device and sufficient free GPU memory before starting training. It does not train or prove that a complete training run will succeed. Use a new output directory for each experiment to preserve existing adapters.

```powershell
.\.venv-train\Scripts\python.exe code/train_lora_dpo.py --output-dir training_runs/qlora_trial_01
.\.venv-train\Scripts\python.exe code/evaluate_finetuned.py --model Qwen/Qwen2.5-3B-Instruct --adapter training_runs/qlora_trial_01 --out benchmark_runs/finetune_trial_01.json
.\.venv-train\Scripts\python.exe code/validate_saved_evaluation.py --report benchmark_runs/finetune_trial_01.json --adapter training_runs/qlora_trial_01
```

Generated adapters, raw incident archives, learned memory, and local model caches are excluded from Git.

## Project layout

| Path | Purpose |
| --- | --- |
| `code/` | Auditing, retrieval, policy, training, and evaluation code |
| `data/` | Synthetic benchmarks, observation examples, and dataset registries |
| `json/` | Legal corpus, saved outputs, and benchmark provenance |
| `tests/` | Regression, leakage, grounding, artifact, and repository tests |
| `main.ipynb` | Reproducible analysis of saved results |

## Limitations

- The yard benchmark is synthetic and does not establish field performance.
- The authentic OSHA source is post-event, severity-selected, and lacks authoritative regulation labels.
- A non-empty retrieval set does not prove the top authority is legally correct.
- Fine-tuning was measured on a small synthetic holdout.
- The current camera interface is an integration contract, not a trained production vision model.
- Legal applicability can depend on facts, jurisdiction, exceptions, and definitions not visible in one event record.
- Continuous learning remains gated and experimental.

This repository is an engineering research system, not legal advice or a replacement for a qualified safety professional.

## Technology

Python, Ollama, Qwen, PyTorch, Transformers, PEFT, bitsandbytes, NumPy, pandas, aiohttp, and local JSON/NumPy indexes.

## License

MIT. See [LICENSE](LICENSE).
