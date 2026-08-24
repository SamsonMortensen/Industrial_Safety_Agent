# Industrial Safety & Compliance Auditor

A locally hosted LLM audits daily rail-intermodal yard logs against federal OSHA and FRA safety regulations, verifies whether the citations it produces are real federal law, and eliminates retrieval blindspots through multi-hazard triangulation.

## What this does

When an AI reads a maintenance log and sees something unsafe, getting it to say *"this looks dangerous"* is easy. Getting it to cite the **exact, real federal law** that was broken (without inventing a fake section number) is hard.

In an industrial rail yard, citing a fabricated law is worse than doing nothing: a safety manager cannot issue a stop-work order or cite a contractor based on an invented rule.

This project builds an autonomous AI safety auditor that runs 100% locally on your own machine. It reads daily yard logs, independently looks up the actual text of federal law (OSHA Title 29 and FRA Title 49), audits the event, and checks every citation it produces against all 16,173 real sections in the  federal regulations I used to ensure zero hallucinations.

```
+----------------------------------------------------------------------------------------------------+
|                                    DAILY YARD EVENT (LOG-1014)                                     |
|  Forklift FL-12 | Main Pedestrian Crosswalk | Shift: 8.5 hrs | Incident: Hydraulic Leak            |
+----------------------------------------------------------------------------------------------------+
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
             [ FATIGUE CHECK: 49 CFR 228 ]                    [ SURFACE CHECK: 29 CFR 1910.22 ]
             - Duty hours: 8.5 hrs <= 12.0 hr                 - Walking surface: Crosswalk
             - Status: Compliant                              - Fluid leak: UNCONTAINED HAZARD
                         │                                                 │
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                                    [ FINAL AUDIT VERDICT ]
                     STATUS:   VIOLATION
                     CITATION: 29 CFR 1910.22 (Walking-Working Surfaces)
                     REASON:   Hydraulic leak on a pedestrian crosswalk creates an
                               uncontained slipping hazard under federal housekeeping rules.
```

## The core question

Getting a language model to flag an unsafe-looking yard event is easy. Getting it to name *which rule* was broken, without inventing the section number, is not. For an industrial compliance tool that distinction is the entire product, the citation is what a safety superintendent acts on, and a confident, correctly formatted, non-existent CFR reference is worse than no answer at all.

So this project measures three things directly:
1. **Detection performance** against a labelled ground-truth log (Precision, Recall, F1).
2. **Citation validity** against all 16,173 real section numbers in Titles 29 and 49 of the Code of Federal Regulations.
3. **Citation correctness** : verifying whether the model cited the *actual governing rule* for that specific hazard, rather than just any real section that happened to be in context.

## Benchmark Results

50 yard events, 9 planted violations across three distinct hazard categories, audited with `qwen3.5:9b` and `mxbai-embed-large` running locally under Ollama.

| Metric | Autonomous Triangulated RAG (New) | Flat Top-4 RAG (Legacy) | Ungrounded Baseline |
| :--- | :---: | :---: | :---: |
| **Violations caught (of 9)** | **9 of 9 (100%)** | 3 of 9 (33%) | 1 of 9 (11%) |
| **False positives** | **0** | 4 | 0 |
| **Precision** | **1.00** | 0.43 | 1.00 |
| **Recall** | **1.00** | 0.33 | 0.11 |
| **F1 Score** | **1.00** | 0.38 | 0.20 |
| **Accuracy** | **1.00** | 0.80 | 0.84 |
| **Citations given** | 9 | 7 | 1 |
| **Citations naming a real CFR section** | **9 of 9 (100%)** | 7 of 7 (100%) | 0 of 1 (0%) |
| **Citations naming the CORRECT rule** | **9 of 9 (100%)** | 0 of 3 (0%) | 0 of 1 (0%) |
| **Audit runtime (50 events)** | **82.5s (async)** | 2,989s (sequential) | ~900s |

![Autonomous Compliance Audit Run](audit_demo.png)

## Why the original grounded build failed, and what fixed it

In the first grounded iteration, recall stalled at 0.33 and precision at 0.43. Building `code/retrieval_diagnostic.py` exposed exactly why:

1. **Citation Validity vs. Citation Correctness:**
   In the legacy grounded run, 3 of 3 caught violations cited `29 CFR 1910.178` (industrial forklift oil cleanliness) for shift-length violations and crosswalk puddles simply because `1910.178` was the only retrieved rule mentioning "clean" or "fluid". The existence check marked them valid because `1910.178` is a real law, but the citation was substantively wrong.

2. **Benign Telemetry False Positives:**
   Clean rows containing routine operational notes (`"Load Imbalance corrected during lift"`) triggered over-eager matching against loading clauses in `1910.178(o)`, creating 4 false alarms on clean shifts.

### The Fix: Multi-Hazard Triangulation & Autonomous Reasoning

Instead of flattening an operational event into a single search query, the auditor triangulates across four fundamental industrial safety pillars:
- **Fatigue & Hours of Service:** `49 CFR 228` (12.0-hour statutory duty limits).
- **Walking-Working Surfaces:** `29 CFR 1910.22` (housekeeping, uncontained fluid leaks on pedestrian paths).
- **Electrical Clearances:** `29 CFR 1910.333` (minimum approach distances near energized lines).
- **Mechanical Integrity & Telemetry:** `29 CFR 1910.178` / `1910.179` (equipment safety, distinguishing transient sensor adjustments from active uncontained hazards).

With multi-hazard triangulation, Retrieval Recall@4 across all hazard types reached **100%**, and Citation Correctness on caught violations jumped from **0% to 100%**.

## How it works

1. `code/fetch_regulations.py` pulls the exact governing safety regulations from the official eCFR versioner API: 29 CFR 1910 (Subparts D, N, S) and 49 CFR 228.
2. `code/build_section_index.py` indexes all 16,173 real sections across Titles 29 and 49 to catch hallucinations.
3. `code/generate_yard_log.py` generates a seeded, reproducible ground-truth operational log with clean rows and planted violations.
4. `code/audit_agent.py` executes the autonomous triangulated RAG audit asynchronously with local Ollama models.
5. `code/stateful_tracker.py` maintains rolling shift memory to catch cumulative fatigue and repeating asset defect patterns across timestamps.
6. `code/retrieval_diagnostic.py` and `code/retrieval_experiment.py` isolate retrieval recall from model reasoning and measure citation correctness.
7. `main.ipynb` presents the interactive walkthrough, data pipelines, and benchmark visualizations.

## Running it

```bash
pip install -r requirements.txt

# 1. Fetch regulations & build section index (one-time setup)
python code/fetch_regulations.py
python code/build_section_index.py
python code/generate_yard_log.py

# 2. Run the autonomous compliance audit
python code/audit_agent.py

# 3. Compare with legacy baselines
python code/audit_agent.py --strategy raw      # legacy flat RAG
python code/audit_agent.py --no-retrieval      # ungrounded prompt

# 4. Run retrieval diagnostics & shift-level tracking
python code/retrieval_diagnostic.py
python code/retrieval_experiment.py
python code/stateful_tracker.py
```

### Prerequisites
Requires [Ollama](https://ollama.com) running locally with:
```bash
ollama pull mxbai-embed-large
ollama pull qwen3.5:9b
```
All embeddings and inferences run entirely on local compute. No yard logs or operational telemetry ever leave the host machine.

## Ground truth dataset

Nine planted violations across 50 records, balanced across three distinct hazard categories:
- **Fatigue (49 CFR 228):** Operator shift duration exceeding the 12.0-hour statutory limit.
- **Spills (29 CFR 1910.22):** Uncontained hydraulic leak on a marked pedestrian walkway/crosswalk.
- **Electrical Clearances (29 CFR 1910.333):** Equipment proximity warning in an energized high-voltage line zone.

`generate_yard_log.py` validates every clean row against hazard patterns to guarantee zero unlabelled collisions.

## Tech stack
- **Python 3.10+**
- **Ollama** (`qwen3.5:9b` for reasoning, `mxbai-embed-large` for embeddings)
- **AsyncIO & AIOHTTP** for concurrent local batch inference
- **NumPy** for vector similarity & cosine metrics
- **Pandas, Matplotlib, Jupyter** for evaluation & reporting
