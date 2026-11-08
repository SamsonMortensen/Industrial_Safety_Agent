"""Recompute publication metrics from saved outputs; never train or call a model."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = (
    "json/audit_results.json", "json/full_audit_results.json",
    "json/full_audit_policy_controlled.json", "json/autonomous_retrieval_benchmark.json",
    "json/finetune_evaluation_results.json", "json/authentic_incident_retrieval_hybrid.json",
    "data/public_development_runs.json",
)
FIGURES = ("audit_tradeoffs.png", "person_development.png", "calibration_diagnostics.png")
MODEL_NAMES = {
    "train_majority_baseline": "Training majority", "training_majority": "Training majority",
    "frozen_zero_shot_baseline": "Fixed prompt", "fixed_prompt_baseline": "Fixed prompt",
    "trained_logistic_probe": "Pilot classifier", "frozen_first_pilot": "Frozen pilot",
    "expanded_classifier": "Round 2 classifier", "frozen_round2": "Frozen Round 2",
    "round3_classifier": "Round 3 classifier", "previous_expanded_classifier": "Previous classifier",
    "new_probe_uncalibrated": "New probe, raw", "new_probe_temperature_scaled": "New probe, scaled",
}


def sha256(path):
    # Git may convert text line endings on checkout; figure bytes stay exact.
    content = path.read_bytes()
    if path.suffix != ".png":
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def metrics_equal(expected, actual):
    """Compare recomputed metrics across Python/libm rounding differences.

    Python 3.12 changed floating-point sum accuracy. Permit only tiny finite
    float differences; counts, types, structure and all non-floats stay exact.
    This never replaces the separate byte/hash integrity checks on evidence.
    """
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            metrics_equal(expected[key], actual[key]) for key in expected)
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            metrics_equal(a, b) for a, b in zip(expected, actual))
    if isinstance(expected, float):
        return math.isfinite(expected) and math.isfinite(actual) and math.isclose(
            expected, actual, rel_tol=1e-12, abs_tol=1e-15)
    return expected == actual


def ratio(a, b):
    return a / b if b else None


def exact_group_interval(k, n):
    """Two-sided 95% Clopper-Pearson interval by binomial-tail inversion.

    For small development group counts (<=200); no frame-independence assumption.
    Reference: docs.scipy.org/.../BinomTestResult.proportion_ci.html (method='exact').
    """
    if not isinstance(n, int) or not isinstance(k, int) or not 0 <= k <= n <= 200 or n == 0:
        raise ValueError("Exact group interval requires 0 <= k <= n <= 200, n > 0")

    def solve(start, stop, target, increasing):
        low, high = 0.0, 1.0
        for _ in range(70):
            p = (low + high) / 2
            tail = sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
                       for j in range(start, stop))
            if (tail < target) == increasing:
                low = p
            else:
                high = p
        return (low + high) / 2

    return [0.0 if k == 0 else solve(k, n + 1, .025, True),
            1.0 if k == n else solve(0, k + 1, .025, False)]


def binary_counts(targets, predictions, strict_unparsed=False):
    if not targets or len(targets) != len(predictions) or any(type(y) is not int or y not in (0, 1) for y in targets):
        raise ValueError("Targets must be a nonempty binary sequence matching predictions")
    if any(p not in (0, 1) and not (strict_unparsed and p is None) for p in predictions):
        raise ValueError("Nonbinary prediction")
    counts = {"true_positive": sum(y == 1 and p == 1 for y, p in zip(targets, predictions)),
              "false_positive": sum(y == 0 and p != 0 for y, p in zip(targets, predictions)),
              "true_negative": sum(y == 0 and p == 0 for y, p in zip(targets, predictions)),
              "false_negative": sum(y == 1 and p != 1 for y, p in zip(targets, predictions))}
    tp, fp, tn, fn = (counts[k] for k in ("true_positive", "false_positive", "true_negative", "false_negative"))
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {"clips": len(targets), "confusion": counts, "accuracy": (tp + tn) / len(targets),
            "precision": ratio(tp, tp + fp), "recall": recall, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "equal_cost_error_rate": (fp + fn) / len(targets),
            "false_alert_rate": ratio(fp, fp + tn), "miss_rate": ratio(fn, tp + fn)}


def classification(rows, model):
    if len({r["case"] for r in rows}) != len(rows):
        raise ValueError("Duplicate validation case")
    result = binary_counts([r["target"] for r in rows], [r["predictions"][model] for r in rows])
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r["target"] == r["predictions"][model])
    correct = sum(all(v) for v in groups.values())
    result.update(source_groups=len(groups), fully_correct_groups=correct,
                  group_exact_95=exact_group_interval(correct, len(groups)))
    return result


def probability_metrics(rows, model):
    """Each source group gets equal total weight; each view shares its group weight."""
    groups = defaultdict(list)
    for row in rows:
        p = row["scores"][model]
        if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Scores must be finite and in [0, 1]")
        if type(row["target"]) is not int or row["target"] not in (0, 1):
            raise ValueError("Invalid target")
        groups[row["group"]].append((row["target"], p))
    if not groups:
        raise ValueError("No scored groups")
    brier, loss = [], []
    for values in groups.values():
        brier.append(sum((p - y) ** 2 for y, p in values) / len(values))
        clipped = [(y, min(1 - 1e-12, max(1e-12, p))) for y, p in values]
        loss.append(sum(-y * math.log(p) - (1 - y) * math.log(1 - p) for y, p in clipped) / len(values))
    return {"group_weighted_brier": sum(brier) / len(groups),
            "group_weighted_log_loss": sum(loss) / len(groups)}


def collect(root=ROOT):
    audits = []
    for filename, name in zip(SOURCES[:3], ("Grounded v1", "Raw v2", "Policy-controlled v2")):
        data = read(root / filename)
        rows = data["results"]
        result = binary_counts([r["actual"] for r in rows], [r["predicted"] for r in rows], True)
        saved = data.get("controlled_metrics", data.get("metrics"))
        for key, value in result["confusion"].items():
            if value != saved[key + "s"]:
                raise ValueError(f"Saved audit count mismatch in {filename}: {key}")
        result.update(name=name, source=filename)
        audits.append(result)
    if any(a["clips"] != 1000 for a in audits):
        raise ValueError("Audit cohort changed; revise the narrative")

    retrieval = read(root / SOURCES[3])
    ranks = retrieval["records"]
    retrieval_metrics = {}
    for field, name in (("fused_rank", "Hybrid fusion"), ("reranked_rank", "Applicability reranking")):
        retrieval_metrics[name] = {str(k): sum(r[field] is not None and 1 <= r[field] <= k for r in ranks) / len(ranks)
                                   for k in (1, 4, 8, 16)}

    qlora = {}
    for name, payload in read(root / SOURCES[4])["strategies"].items():
        rows = payload["results"]
        violations = [r for r in rows if r["actual"] == 1]
        qlora[name] = {"cases": len(rows), "parsed": sum(r["status"] != "UNPARSED" for r in rows),
                       "verdict_correct": sum(r["predicted"] == r["actual"] for r in rows),
                       "violation_cases": len(violations),
                       "citation_correct_on_violations": sum(bool(r["citation_correct"]) for r in violations),
                       "joint_correct": sum(r["predicted"] == r["actual"] and bool(r["citation_correct"]) for r in rows)}

    authentic = read(root / SOURCES[5])
    snapshot = read(root / SOURCES[6])
    if snapshot["schema"] != "public_development_snapshot_v1":
        raise ValueError("Unknown snapshot schema")
    person = {}
    for run in snapshot["runs"]:
        if run["independent_evaluation_used"] is not False or run["model_promoted"] is not False:
            raise ValueError("Development snapshots cannot certify independent evaluation or promotion")
        metrics = {model: classification(run["rows"], model) for model in run["models"]}
        scored = {model: probability_metrics(run["rows"], model) for model in run["models"]
                  if all(model in row["scores"] for row in run["rows"])}
        person[run["id"]] = {"coverage": run["coverage"], "metrics": metrics, "score_diagnostics": scored}
        if run["id"] == "calibration":
            person[run["id"]]["temperature"] = run["temperature"]
    result = {"schema": "public_results_metrics_v1", "audit": audits,
            "retrieval": {"cases": len(ranks), "methods": retrieval_metrics}, "qlora": qlora,
            "incident_coverage": {key: authentic["coverage"][key] for key in
                                   ("cases", "triggered", "nonempty_retrieval", "unique_top_authorities")},
            "person": person}
    validate_historical_scope(result)
    return result


def validate_historical_scope(data):
    """Stop before regenerating a historical interpretation against different runs.

    This version tells the story of named saved experiments. New cohorts belong
    in a reviewed report revision, not under the old headings and conclusions.
    """
    expected_audits = [(103, 4, 852, 41), (82, 1, 855, 62), (98, 0, 856, 46)]
    if [tuple(r["confusion"].values()) for r in data["audit"]] != expected_audits:
        raise ValueError("Audit results changed; review the historical interpretation")
    expected_qlora = {"base": (56, 0, 0, 28, 1, 0), "qlora_sft": (56, 56, 50, 28, 16, 41)}
    if {k: tuple(v.values()) for k, v in data["qlora"].items()} != expected_qlora:
        raise ValueError("Fine-tuning cohort changed; review the interpretation")
    r = data["retrieval"]
    if r["cases"] != 144 or any(v["16"] != 1 for v in r["methods"].values()) or any(
            r["methods"]["Hybrid fusion"][k] < r["methods"]["Applicability reranking"][k] for k in ("1", "4", "8")):
        raise ValueError("Retrieval comparison changed; review the interpretation")
    expected = {"pilot": (14, 7, 4, 2), "round2": (52, 26, 10, 5), "round3": (80, 40, 8, 4)}
    if set(data["person"]) != {*expected, "calibration"}:
        raise ValueError("Development run set changed")
    for name, counts in expected.items():
        coverage = data["person"][name]["coverage"]
        actual = tuple(coverage[role][k] for role in ("development_train", "development_validation") for k in ("clips", "groups"))
        if actual != counts:
            raise ValueError("Development cohort changed; review the interpretation")
    comparisons = [("round2", "frozen_first_pilot", 1, 0), ("round2", "expanded_classifier", 1, 0),
                   ("round3", "frozen_first_pilot", 0, 2), ("round3", "frozen_round2", 0, 0),
                   ("round3", "round3_classifier", 0, 0)]
    for run, model, fp, fn in comparisons:
        c = data["person"][run]["metrics"][model]["confusion"]
        if (c["false_positive"], c["false_negative"]) != (fp, fn):
            raise ValueError("Matched comparison changed; review the interpretation")
    calibration = data["person"]["calibration"]
    if calibration["coverage"] != {"classifier_fit": {"clips": 36, "groups": 18},
                                    "temperature_fit": {"clips": 16, "groups": 8},
                                    "development_validation": {"clips": 10, "groups": 5}}:
        raise ValueError("Calibration cohort changed")
    for model, value in (("new_probe_uncalibrated", .0726), ("new_probe_temperature_scaled", .0774)):
        if round(calibration["score_diagnostics"][model]["group_weighted_brier"], 4) != value:
            raise ValueError("Calibration result changed; review the interpretation")
    for m in calibration["metrics"].values():
        if (m["confusion"]["false_positive"], m["confusion"]["false_negative"]) != (1, 0):
            raise ValueError("Calibration error counts changed")


def pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def render_markdown(data):
    lines = ["# Results and analytical decisions", "",
             "Generated from saved outputs by `python code/public_results.py`. These are selected historical runs, not a complete experiment registry or a new model execution.", "",
             "This project demonstrates data profiling, baseline design, matched comparisons, group-aware validation, probability diagnostics, and reproducible reporting. The results below show both gains and experiments that did not improve the chosen metric.", "",
             "## Read the evidence in context", "",
             "| Track | Evidence | What a public reader can reproduce |",
             "| --- | --- | --- |",
             "| Text audit and retrieval | Synthetic benchmark; saved predictions | Counts and retrieval curves from public per-case outputs |",
             "| QLoRA | Synthetic, text-disjoint 56-case holdout | Format, verdict, and citation scores; no model training in this report |",
             "| Person rounds | Selected synthetic development clips; AI-assisted human review | Metrics from anonymous validation targets and predictions |",
             "| Calibration experiment | Previously exposed development validation | Group-weighted scores for raw and scaled predictions |",
             "| Independent evaluation | No independent results included | Remains a separate review and evaluation process |", "",
             "## 1. Audit tradeoffs and retrieval", "",
             "![Audit recall and false alerts alongside retrieval ranking](results/audit_tradeoffs.png)", "",
             "All three audit results use 1,000 synthetic events: 144 violations and 856 nonviolations. Raw and policy-controlled v2 reuse the same saved model outputs. Grounded v1 is a separate model run. These are pipeline results, including deterministic triage, rather than isolated model-only scores.", "",
             "| Pipeline | True positives | False alerts / 856 | Misses / 144 | Precision | Recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for run in data["audit"]:
        c = run["confusion"]
        lines.append(f"| {run['name']} | {c['true_positive']} | {c['false_positive']} | {c['false_negative']} | {pct(run['precision'])} | {pct(run['recall'])} |")
    lines += ["", "**Decision:** policy controls reduced v2 errors on this fixture, but 46 violations were still missed. Grounded v1 retained higher recall. Zero observed false alerts does not establish a zero population error rate.", "",
              f"The separate retrieval task has {data['retrieval']['cases']} known-violation queries. Hybrid fusion placed the expected authority earlier more often than applicability reranking; both reached 144/144 within the top 16. Retrieval success does not establish a correct final decision.", "",
              "## 2. Fine-tuning: distinguish format from domain correctness", "",
              "| Saved output | Parsed / 56 | Correct verdict / 56 | Correct citation on violations / 28 | Verdict + citation / 56 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for key, label in (("base", "Base model"), ("qlora_sft", "QLoRA adapter")):
        r = data["qlora"][key]
        lines.append(f"| {label} | {r['parsed']} | {r['verdict_correct']} | {r['citation_correct_on_violations']} | {r['joint_correct']} |")
    lines += ["", "**Interpretation:** all 56 base outputs failed the required parser, so its zero strict verdict score is largely a format failure. The adapter achieved 50/56 correct verdicts, but only 16/28 correct violation citations. The table does not isolate a reasoning improvement. Training used 70 synthetic cases; holdout separation was by text, not proof of domain or template independence.", "",
              "## 3. Person detection: compare models on the same source groups", "",
              "![Person development comparisons and source-group uncertainty](results/person_development.png)", "",
              "Camera views from the same simulation stay together. The tables compare methods within each round; different rounds have different validation sets and must not be joined into an accuracy improvement curve. Both false alerts and missed people have unit cost. Balanced accuracy is reported separately because class prevalence differs.", ""]
    for key, title in (("pilot", "First pilot"), ("round2", "Round 2"), ("round3", "Round 3")):
        r = data["person"][key]
        training, validation = r["coverage"]["development_train"], r["coverage"]["development_validation"]
        lines += ["", f"### {title}", "", f"Fitting: {training['clips']} clips / {training['groups']} source groups. Validation: {validation['clips']} clips / {validation['groups']} source groups.", "",
                  "| Method | Correct | False alerts / negatives | Misses / positives | Balanced accuracy | Fully correct groups; exact 95% interval |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"]
        for model, m in r["metrics"].items():
            c = m["confusion"]
            lo, hi = m["group_exact_95"]
            lines.append(f"| {MODEL_NAMES[model]} | {c['true_positive'] + c['true_negative']}/{m['clips']} | {c['false_positive']}/{c['false_positive'] + c['true_negative']} | {c['false_negative']}/{c['true_positive'] + c['false_negative']} | {pct(m['balanced_accuracy'])} | {m['fully_correct_groups']}/{m['source_groups']}; {pct(lo)}–{pct(hi)} |")
    lines += ["", "**Decision:** Round 2 tied the frozen pilot on its ten validation clips. On Round 3's eight clips, the frozen pilot missed two people; both Round 2 and Round 3 classified every clip correctly. No classification superiority of Round 3 over Round 2 was established.", "",
              "The 4/4 fully correct groups for Round 3 have a 39.8%–100% exact interval. This interval concerns the fraction of source groups with *every view correct*, not clip accuracy or a guarantee about field performance. It assumes independent, similarly distributed groups; shared simulation assets can violate that assumption. Sparse frame sampling can miss short appearances, and model-assisted review can bias labels.", "",
              "Data-quality check: the first pilot's validation clips 13 and 14 have Person=Absent in the saved predictions. An equipment-collision scenario is not itself a Person label. Public exports omit clip identifiers and media paths.", "",
              "## 4. Calibration: retain the negative result", "",
              "![Development validation probability diagnostics before and after temperature scaling](results/calibration_diagnostics.png)", "",
              "This experiment fitted a new probe on 36 clips / 18 groups and a temperature on 16 clips / 8 separate groups. Its ten validation clips / five groups had already been exposed in Round 2. Only one validation group was negative. Calibration groups were excluded from this new probe but had trained earlier classifiers; the older model is contextual, not an isolated calibration control.", "",
              "| Method | Group-weighted Brier ↓ | Group-weighted log loss ↓ |",
              "| --- | ---: | ---: |"]
    for model, p in data["person"]["calibration"]["score_diagnostics"].items():
        lines.append(f"| {MODEL_NAMES[model]} | {p['group_weighted_brier']:.4f} | {p['group_weighted_log_loss']:.4f} |")
    lines += ["", f"**Decision:** temperature scaling (T={data['person']['calibration']['temperature']:.4f}) worsened validation Brier score for the same new probe, from 0.0726 to 0.0774. All three variants made one false alert and no misses. Positive temperature scaling leaves the 0.5 decision boundary unchanged. This candidate/calibrator was not accepted or promoted.", "",
              "A later calibration-reserve fit is a different experiment and is not pooled into these results. Fit-set improvement alone cannot establish out-of-sample calibration.", "",
              "## 5. Authentic incident data: measure coverage honestly", "",
              f"The saved OSHA incident stress test processed {data['incident_coverage']['cases']} records, triggered {data['incident_coverage']['triggered']} investigations and returned nonempty candidates for {data['incident_coverage']['nonempty_retrieval']} records. It produced {data['incident_coverage']['unique_top_authorities']} distinct top-ranked authorities. All sampled records were known reported incidents and human-flagged by design. This is ingestion/retrieval coverage, not detection recall or legal accuracy.", "",
              "## Reproduce and extend", "",
              "Run `python code/public_results.py --check` to verify the saved report against its inputs and output hashes. To regenerate the figures and tables, install `requirements-results.txt`, then run `python code/public_results.py`. No GPU, model download, service, or private campaign is needed to recompute the public metrics.", "",
              "The [workflow and metric definitions](ANALYTICS_WORKFLOW.md) explain the denominators, anonymous export, and how to add future runs. [metrics.json](results/metrics.json) contains recomputed numbers; [manifest.json](results/manifest.json) records input and output hashes. The [anonymous development snapshot](../data/public_development_runs.json) reproduces historical metrics, not training or human-label verification.", "",
              "Exact group intervals use the [Clopper–Pearson method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html). They are descriptive under the stated group assumptions; no independent evaluation or deployment claim is made here.", ""]
    return "\n".join(lines)


def render_figures(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#263449",
                         "text.color": "#16243a", "figure.facecolor": "#ffffff"})
    teal, blue, orange = "#087e8b", "#355c9b", "#d17922"

    def finish(fig, filename, title, subtitle, left=.08):
        fig.suptitle(title, x=.05, y=.98, ha="left", fontsize=18, weight="bold")
        fig.text(.05, .895, subtitle, fontsize=10)
        fig.subplots_adjust(left=left, right=.97, bottom=.16, top=.77, wspace=.42)
        fig.savefig(out / filename, dpi=160, facecolor="white", metadata={"Software": "Industrial Safety Agent results"})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, a in enumerate(data["audit"]):
        axes[0].bar(i - .17, a["precision"], width=.32, color=blue, label="Precision" if i == 0 else None)
        axes[0].bar(i + .17, a["recall"], width=.32, color=teal, label="Recall" if i == 0 else None)
    axes[0].set(xticks=range(3), xticklabels=["Grounded v1", "Raw v2", "Controlled v2"], ylim=(0, 1.18), title="1,000 synthetic events · 144 violations")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    for i, a in enumerate(data["audit"]):
        axes[0].text(i + .17, a["recall"] + .025, pct(a["recall"]), ha="center", fontsize=9)
    for name, values in data["retrieval"]["methods"].items():
        axes[1].plot(range(4), list(values.values()), "o-", label=name)
    axes[1].set(xticks=range(4), xticklabels=["1", "4", "8", "16"], xlabel="Retrieved candidates (k)", ylim=(0, 1.1), title="Authority retrieval · 144 queries")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel("Recall@k")
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    finish(fig, FIGURES[0], "Accuracy alone hides the tradeoff", "Saved synthetic benchmark outputs · Controlled v2: 0 false alerts, 46 missed violations")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    models = ("frozen_first_pilot", "frozen_round2", "round3_classifier")
    metrics = data["person"]["round3"]["metrics"]
    for i, model in enumerate(models):
        c = metrics[model]["confusion"]
        axes[0].barh(i - .17, c["false_negative"], height=.3, color=orange, label="Missed people" if i == 0 else None)
        axes[0].barh(i + .17, c["false_positive"], height=.3, color=blue, label="False alerts" if i == 0 else None)
        axes[0].text(max(c["false_positive"], c["false_negative"]) + .12, i, f"{c['true_positive'] + c['true_negative']}/8 correct", va="center", fontsize=9)
    axes[0].set(yticks=range(3), yticklabels=["Frozen pilot", "Frozen Round 2", "Round 3"], xlim=(0, 4.2), xticks=range(5), ylim=(2.55, -.6), xlabel="Errors on the same eight clips", title="Round 3 · 4 present / 4 absent")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    for i, model in enumerate(models):
        m = metrics[model]; lo, hi = m["group_exact_95"]; p = m["fully_correct_groups"] / m["source_groups"]
        axes[1].errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=teal, capsize=5)
        axes[1].text(.02, i - .23, f"{m['fully_correct_groups']}/4 groups · {pct(lo)}–{pct(hi)}", fontsize=9)
    axes[1].set(yticks=range(3), yticklabels=[], xlim=(0, 1.05), ylim=(2.55, -.6), xlabel="Source groups with every view correct", title="Four groups · exact 95% intervals")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    fig.text(.05, .04, "Intervals assume independent, similar source groups; shared synthetic assets may violate that assumption.", fontsize=9)
    finish(fig, FIGURES[1], "Perfect on eight clips is still limited evidence", "Round 2 and Round 3 tie · Selected synthetic development data · Independent evaluation separate", left=.16)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = ("new_probe_uncalibrated", "new_probe_temperature_scaled")
    scores = data["person"]["calibration"]["score_diagnostics"]
    for ax, key, title in zip(axes, ("group_weighted_brier", "group_weighted_log_loss"), ("Brier score", "Log loss")):
        vals = [scores[m][key] for m in methods]
        ax.bar([0, 1], vals, color=[teal, orange], width=.5)
        ax.set(xticks=[0, 1], xticklabels=["Same probe: raw", "Same probe: scaled"], ylim=(0, max(vals) * 1.28), title=f"{title} · lower is better")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * .04, f"{v:.4f}", ha="center", weight="bold")
    finish(fig, FIGURES[2], "Calibration did not improve this validation score", "10 previously exposed clips · 5 source groups · 1 negative group · Equal weight per source group")


def artifact_paths():
    return ["docs/RESULTS.md", "docs/results/metrics.json"] + [f"docs/results/{f}" for f in FIGURES]


def manifest_inputs(root):
    paths = list(SOURCES) + ["code/public_results.py", "code/export_public_development.py", "requirements-results.txt"]
    return {p: sha256(root / p) for p in paths}


def check(root=ROOT):
    data = collect(root)
    if not metrics_equal(read(root / "docs/results/metrics.json"), data):
        raise ValueError("Saved metrics are stale")
    if (root / "docs/RESULTS.md").read_text(encoding="utf-8") != render_markdown(data):
        raise ValueError("Results narrative is stale")
    manifest = read(root / "docs/results/manifest.json")
    if manifest["hash_mode"] != "sha256_text_lf_binary_raw_v1":
        raise ValueError("Unknown fingerprint convention")
    if manifest["inputs_sha256"] != manifest_inputs(root):
        raise ValueError("Results input or generator changed; regenerate the report")
    actual = {p: sha256(root / p) for p in artifact_paths()}
    if manifest["outputs_sha256"] != actual:
        raise ValueError("Results output integrity check failed")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check metrics, narrative and hashes without plotting dependencies")
    args = parser.parse_args()
    if args.check:
        check()
        print("Public results verified: saved metrics, narrative, source hashes and figure hashes match.")
        return
    data = collect()
    out = ROOT / "docs/results"
    out.mkdir(parents=True, exist_ok=True)
    render_figures(data, out)
    (out / "metrics.json").write_text(json_text(data), encoding="utf-8")
    (ROOT / "docs/RESULTS.md").write_text(render_markdown(data), encoding="utf-8")
    import matplotlib
    manifest = {"schema": "public_results_manifest_v1", "reproduction_scope": "saved_prediction_metrics_only",
                "hash_mode": "sha256_text_lf_binary_raw_v1",
                "renderer": {"matplotlib": matplotlib.__version__},
                "inputs_sha256": manifest_inputs(ROOT),
                "outputs_sha256": {p: sha256(ROOT / p) for p in artifact_paths()}}
    (out / "manifest.json").write_text(json_text(manifest), encoding="utf-8")
    print("Generated public results, three figures, recomputed metrics and provenance manifest.")


if __name__ == "__main__":
    main()
