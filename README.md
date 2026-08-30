# Industrial Safety & Compliance Auditor

A local, retrieval-grounded safety auditor for intermodal yard operations. It evaluates operational observations against federal safety authorities, verifies citations, and sends unsupported or ambiguous cases to review.

The project combines structured safety rules, local language models, authentic incident narratives, deterministic controls, and a camera-neutral observation interface. Operational data and model inference remain local.

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
| Single-event dense query | 0.0625 | 0.3056 | 0.7361 | 0.8264 |
| Multi-query lexical | 0.1111 | 0.2292 | 0.3889 | 0.4792 |
| Multi-query hybrid with applicability reranking | **0.1736** | **0.6319** | **0.8681** | **0.9792** |

The hybrid method retrieved the expected authority within the top 16 for 141 of 144 violations. Results and per-hazard ranks are saved in `json/autonomous_retrieval_benchmark.json`.

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

- `VISION_TRAINING.md` defines the staged perception plan.
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

## Running locally

### Prerequisites

- Python 3.11 or compatible
- [Ollama](https://ollama.com)
- `mxbai-embed-large` for embeddings
- `qwen3.5:9b` for local reasoning

```bash
pip install -r requirements.txt
ollama pull mxbai-embed-large
ollama pull qwen3.5:9b
```

### Build the corpus and benchmark log

```bash
python code/fetch_regulations.py
python code/build_section_index.py
python code/generate_yard_log.py --records 1000
```

### Run retrieval and autonomous observation tests

```bash
python code/retrieval_diagnostic.py
python code/benchmark_autonomous_retrieval.py
python code/run_autonomous_investigation.py data/example_camera_observation.json --sample-rate 0
```

### Run the audits

```bash
python code/audit_agent.py --limit 1000
python code/audit_agent_v2.py --limit 1000 --concurrency 4 --out json/audit_results_v2.json
python code/apply_policy_controls.py --raw json/audit_results_v2.json --out json/audit_results_v2_controlled.json
```

### Prepare authentic incident text

```bash
python code/fetch_osha_incidents.py
python code/osha_incident_pipeline.py
python code/benchmark_authentic_incidents.py --sample 500 --dense --out json/authentic_incident_retrieval_hybrid.json
```

### Validate the repository

```bash
python -m compileall -q code tests
python -m unittest discover -s tests -v
python -m json.tool main.ipynb
```

### Fine-tuning

Install a CUDA-enabled PyTorch build compatible with the host driver before installing the training dependencies.

```bash
pip install -r requirements-train.txt
python code/train_lora_dpo.py --preflight-only
python code/train_lora_dpo.py
python code/evaluate_finetuned.py --model Qwen/Qwen2.5-3B-Instruct
python code/validate_saved_evaluation.py
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
| `VISION_TRAINING.md` | Camera-model data and training plan |

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
