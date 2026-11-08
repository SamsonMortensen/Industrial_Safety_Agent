"""Generate and verify a matched development comparison, including regressions."""
import argparse
import csv
import io
import json
from pathlib import Path

from capture_retrieval_run import file_hash, summarize
from public_results import metrics_equal

ROOT = Path(__file__).resolve().parents[1]
DATA = "data/retrieval_comparison"
REPORT = "docs/RETRIEVAL_IMPROVEMENT.md"
FIGURES = ("retrieval_before_after.png", "original_nine_misses.png", "retrieval_rank_cutoffs.png")
COHORTS = {"primary64": "Original 64 development cases", "additional80": "Additional 80 public cases", "all144": "All 144 public cases"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def hit(row, k=16):
    return row["rank"] is not None and row["rank"] <= k


def validate_capture(capture):
    if capture.get("schema") != "lexical_retrieval_development_capture_v1" or type(capture.get("top_k")) is not int or capture["top_k"] != 16:
        raise ValueError("Unsupported capture schema or candidate budget")
    if capture.get("independent_evaluation_used") is not False or capture.get("calibration_reserve_used") is not False:
        raise ValueError("Only development captures may enter this report")
    rows = capture["records"]
    if len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate case identifier")
    if len(rows) != 144 or sum(r["cohort"] == "primary64" for r in rows) != 64 or sum(r["cohort"] == "additional80" for r in rows) != 80:
        raise ValueError("Public development cohorts changed")
    for row in rows:
        sections = row["ranked_sections"]
        if len(sections) > 16 or len(set(sections)) != len(sections):
            raise ValueError("Candidate list exceeds budget or repeats authorities")
        expected_rank = sections.index(row["expected_section"]) + 1 if row["expected_section"] in sections else None
        if (row["rank"] is not None and type(row["rank"]) is not int) or row["rank"] != expected_rank:
            raise ValueError("Rank does not match the saved candidate list")
    recomputed = {c: summarize([r for r in rows if c == "all144" or r["cohort"] == c]) for c in COHORTS}
    if not metrics_equal(capture["metrics"], recomputed):
        raise ValueError("Saved capture metrics do not match case-level ranks")


def compare(before, after):
    for capture in (before, after):
        validate_capture(capture)
    for name in ("daily_yard_log.csv", "json/regulations.json", "json/statutes.json"):
        if before["input_and_code_sha256_lf"][name] != after["input_and_code_sha256_lf"][name]:
            raise ValueError("A matched comparison must keep dataset and corpus fixed")
    old, new = ({r["event_id"]: r for r in c["records"]} for c in (before, after))
    if set(old) != set(new):
        raise ValueError("Case sets differ")
    pairs = []
    for case in old:
        a, b = old[case], new[case]
        if any(a[k] != b[k] for k in ("incident", "hazard", "expected_section", "cohort")):
            raise ValueError("Case labels, text, or cohort changed")
        pairs.append({"event_id": case, "cohort": a["cohort"], "incident": a["incident"],
                      "hazard": a["hazard"], "expected_section": a["expected_section"],
                      "before_rank": a["rank"], "after_rank": b["rank"],
                      "top16_change": "recovered" if not hit(a) and hit(b) else "lost" if hit(a) and not hit(b) else "retained" if hit(a) else "still_missing"})
    summary = {}
    for cohort in COHORTS:
        keys = [r["event_id"] for r in pairs if cohort == "all144" or r["cohort"] == cohort]
        cutoffs = {}
        for k in (1, 4, 8, 16):
            cutoffs[str(k)] = {"gained": sum(not hit(old[i], k) and hit(new[i], k) for i in keys),
                                "lost": sum(hit(old[i], k) and not hit(new[i], k) for i in keys)}
        summary[cohort] = {"before": before["metrics"][cohort], "after": after["metrics"][cohort], "paired_cutoffs": cutoffs}
    return {"schema": "matched_retrieval_development_comparison_v1", "independent_evaluation_used": False,
            "calibration_reserve_used": False, "model_promoted": False, "metrics": summary,
            "original_misses": [r for r in pairs if r["cohort"] == "primary64" and r["before_rank"] is None],
            "remaining_misses": [r for r in pairs if r["after_rank"] is None],
            "rank_regressions": [r for r in pairs if r["before_rank"] is not None and
                                 (r["after_rank"] is None or r["after_rank"] > r["before_rank"])],
            "top1_losses": [r for r in pairs if r["before_rank"] == 1 and r["after_rank"] != 1], "cases": pairs}


def csv_text(data):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(data["cases"][0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(data["cases"])
    return stream.getvalue()


def markdown(data, before, after, exploration):
    lines = ["# Retrieval improvement: what changed and what it means", "",
             "## Public overview", "",
             "The public CPU walkthrough originally missed the expected authority in nine of 64 synthetic cases. A focused query-planning change recovered eight of those nine. It also improved the other 80 cases in the same public fixture after the candidate was frozen. All results below compare the same cases, corpus, labels, retrieval method, and 16-authority budget.", "",
             "![Before-and-after authority retrieval with counts and percentages](results/retrieval_before_after.png)", "",
             "| Development cohort | Before: found / cases | After: found / cases | Change in Recall@16 | Gained / lost at top 16 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for key, label in COHORTS.items():
        m = data["metrics"][key]; a, b = m["before"], m["after"]; t = m["paired_cutoffs"]["16"]
        lines.append(f"| {label} | {a['hits_at_16']}/{a['cases']} ({a['recall_at_16']:.1%}) | {b['hits_at_16']}/{b['cases']} ({b['recall_at_16']:.1%}) | {(b['recall_at_16'] - a['recall_at_16']) * 100:+.1f} percentage points | {t['gained']} / {t['lost']} |")
    lines += ["", "**Reading the result:** Recall@16 asks whether the benchmark's expected authority appears anywhere in the 16 returned authorities. It does not measure whether a final decision or citation is correct. The first 64 cases were used to diagnose and select this change. The other 80 are a development regression check from the same known synthetic dataset, not independent evaluation. The all-144 row includes both subsets; do not add it as a third cohort.", "",
              "**The implementation change is small:** keep the incident description in context queries and remove added generic phrases such as `federal safety requirement` and `applicable limit or minimum requirement`. Observed words are preserved, even when a user actually mentions those phrases. No citation numbers, case IDs, expected labels, or hazard-to-authority lookup rules were added to retrieval.", "",
              "## The original nine misses", "",
              "![Each original miss and its new position](results/original_nine_misses.png)", "",
              "| Case | Hazard family | Expected authority | Before | After |",
              "| --- | --- | --- | --- | --- |"]
    for r in data["original_misses"]:
        rank = f"Rank {r['after_rank']}" if r["after_rank"] is not None else "Still outside top 16"
        lines.append(f"| {r['event_id']} | {r['hazard'].replace('_', ' ')} | {r['expected_section']} | Outside top 16 | {rank} |")
    unique = len({r["incident"] for r in data["original_misses"]})
    lines += ["", f"Those nine rows contain only **{unique} distinct incident wordings**, repeated with different equipment, location, or measurement fields. They are not nine independent demonstrations of generalization.", "",
              "## Why the queries were missing useful evidence", "",
              "The query planner originally produced an incident query, relation queries, and separate equipment/location/measurement queries. When explicit action and condition fields were empty, the context queries omitted the incident. Several also appended generic regulatory wording. With multi-query rank fusion, repeated background terms could accumulate support for unrelated authorities.", "",
              "For `LOG-1011`, the observation is: **Hydraulic fluid pooled across the marked walking route and left unattended.** The actual shift-measurement query changed as follows:", "",
              "**Before**", "", "```text", before["records"][[r["event_id"] for r in before["records"]].index("LOG-1011")]["queries"][3], "```", "",
              "**After**", "", "```text", after["records"][[r["event_id"] for r in after["records"]].index("LOG-1011")]["queries"][3], "```", "",
              "The expected housekeeping authority was outside the original top 16 and now appears at rank 14. The incident-only query was already useful; the change prevents background fields and generated legal wording from dominating the combined query set.", "",
              "The diagnosis also found a chunk-pool bottleneck. Long sections can occupy many of the 50 per-query chunk slots before the retriever reduces them to distinct authorities. However, simply diversifying that pool made the 64-case result worse, so it was not included. The accepted change alters **QueryPlanner only**; BM25 scoring, fusion weights, pool size, applicability reranking, corpus text, and citation/decision gates are unchanged.", "",
              "## Earlier ranks and regressions", "",
              "![Matched before-and-after results at four rank cutoffs](results/retrieval_rank_cutoffs.png)", "",
              "| All 144 cases | Before | After | Cases entering cutoff | Cases leaving cutoff |",
              "| --- | ---: | ---: | ---: | ---: |"]
    full = data["metrics"]["all144"]
    for k in (1, 4, 8, 16):
        t = full["paired_cutoffs"][str(k)]
        lines.append(f"| Recall@{k} | {full['before'][f'hits_at_{k}']}/144 | {full['after'][f'hits_at_{k}']}/144 | {t['gained']} | {t['lost']} |")
    lines += ["", f"Mean reciprocal rank, truncated at 16, rose from **{full['before']['mean_reciprocal_rank_at_16']:.3f} to {full['after']['mean_reciprocal_rank_at_16']:.3f}**. Higher is better: a correct authority at rank 1 contributes 1; at rank 4 it contributes 1/4; a miss contributes zero.", "",
              f"No previously found authority left the top 16. Nevertheless, **{len(data['rank_regressions'])} cases worsened in rank**. Four fell from rank 1 to rank 2:", "",
              "| Case | Hazard family | Before → after |", "| --- | --- | --- |"]
    for r in data["top1_losses"]:
        lines.append(f"| {r['event_id']} | {r['hazard'].replace('_', ' ')} | {r['before_rank']} → {r['after_rank']} |")
    lines += ["", "These regressions matter if a downstream user reads only the first result. The complete per-case table includes every improvement and regression, rather than only the recovered cases.", "",
              "## Four cases still need work", "",
              "| Case | Cohort | Incident | Expected authority |", "| --- | --- | --- | --- |"]
    for r in data["remaining_misses"]:
        lines.append(f"| {r['event_id']} | {r['cohort']} | {r['incident']} | {r['expected_section']} |")
    lines += ["", "For the remaining original miss, `LOG-1442`, the best target-chunk position across its five new queries is 53, just outside the unchanged 50-chunk pool. Its short 'covered switch employee' wording has weak lexical overlap with the stored train-employee statute excerpt. This is evidence of a candidate-generation limitation, not justification to inject the expected citation. The other misses show that vocabulary differences and equipment context can still matter.", "",
              "The next bounded experiments should separately test semantic retrieval, general lexical normalization, and candidate generation on development data, with the cost and rank tradeoffs recorded. Any revised corpus should retain official source provenance and be evaluated as its own change. The independent evaluation remains reserved for an approved frozen candidate.", "",
              "## Technical appendix: experimental design", "",
              "1. Preserved the earlier publication candidate and the 55/64 rehearsal output. Recreated the exact original case-level ranks before changing anything.",
              "2. Used the original 64 cases for diagnosis and prototype selection. Kept unsuccessful variants in the exploration record.",
              "3. Froze the selected query-planner source before inspecting its results on the remaining 80 public cases. No further retrieval tuning followed that inspection.",
              "4. Compared all 144 cases with unchanged source data, labels, top-k budget, and retrieval method. Recorded paired gains/losses and all returned authority lists.",
              "5. Ran query-boundary regressions, the public suite, saved-result checks, notebook validation, and the updated live walkthrough. Execution evidence is recorded separately from retrieval effectiveness.", "",
              "### Variants examined on the original 64", "",
              "| Prototype | Hits@1 | Hits@4 | Hits@8 | Hits@16 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, experiment in exploration.items():
        counts = experiment["counts"]
        lines.append(f"| {name.replace('_', ' ')} | {counts['1']} | {counts['4']} | {counts['8']} | {counts['16']} |")
    lines += ["", "The pool-only prototype collapsed repeated authorities before the pool cutoff but retained the existing per-chunk fusion. The authority-document prototypes concatenated chunks per authority for lexical indexing. Neither index change was accepted. These are development-selection results, not six independent tests of a final model.", "",
              "### How the unchanged rank fusion works", "",
              "The existing retriever uses a weighted reciprocal-rank-fusion variant. Query q has weight `1 / (1 + 0.25*q)`; a chunk at rank r contributes `weight / (60 + r)`. Its fused score adds the contributions and twice its strongest contribution, then authorities are ranked by their best chunk. As a result, query content matters even when the first incident query stays unchanged. The [original reciprocal rank fusion paper](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/) provides the background; the weighting and peak term here are this project's existing implementation.", "",
              "### What these numbers do and do not establish", "",
              "These percentages are exact counts on a fixed, selected synthetic fixture. Repeated templates, public exposure, and prototype selection prevent interpreting them as unbiased estimates of field performance. No new confidence interval or significance claim is made for generalization. The additional 80 cases were not used to choose this candidate during this experiment, but they belong to the same previously available benchmark. They are not an independent holdout.", "",
              "The benchmark labels select the violation subset and score retrieval; they do not form queries. A retrieved authority is not a legal conclusion. The corpus is the same pinned historical corpus as before, not a verification of current law. No private calibration reserve, independent review bundle, model training, or promotion is part of this experiment.", "",
              "### Reproduce the comparison", "",
              "```powershell", "python code/capture_retrieval_run.py --out benchmark_runs/retrieval_current.json",
              "python code/report_retrieval_comparison.py --check --candidate-capture benchmark_runs/retrieval_current.json",
              "```", "",
              "The first command runs the current lexical pipeline with network calls forbidden. The second verifies the fresh queries/ranks against the saved after-run, then recomputes the report and checks its evidence hashes. Counts, queries, ranks and hashes must match exactly. Recomputed floating-point metrics allow only rounding differences across supported Python versions (`rel_tol=1e-12`, `abs_tol=1e-15`); saved evidence bytes and their hashes remain exact. Use a new output name on each capture; existing evidence is not overwritten.", "",
              "To reproduce the before-run, run the capture script in a separate process with `--repo PATH_TO_PREVIOUS_PUBLICATION_CHECKOUT`. The retrieval code at GitHub base `10eb13d26bacbbd9ff0563affc82e4fc232bad69` has the same retrieval behavior; the earlier reliability patch should be applied if exact recorded hashes for other modules are required. The capture rejects any dataset other than the pinned public synthetic fixture.", "",
              "Install `requirements-results.txt`, then run `python code/report_retrieval_comparison.py` to regenerate this Markdown and its large, single-chart PNGs. The `--check` path uses the standard library. Charts start at zero, show exact counts and denominators, and keep the two disjoint subsets separate from their combined total.", "",
              "Files: [before capture](../data/retrieval_comparison/before.json), [after capture](../data/retrieval_comparison/after.json), [matched metrics](results/retrieval_comparison.json), [every case as CSV](results/retrieval_cases.csv), [protocol](../data/retrieval_comparison/protocol.json), [candidate freeze](../data/retrieval_comparison/candidate_freeze.json), [integrity manifest](results/retrieval_manifest.json).", ""]
    return "\n".join(lines)


def draw(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#14263e", "axes.labelcolor": "#14263e"})
    gray, teal, orange = "#697b90", "#087e8b", "#ae5b1c"

    def title(fig, heading, subtitle, footnote):
        fig.text(.055, .945, heading, fontsize=24, weight="bold")
        fig.text(.055, .88, subtitle, fontsize=14, color="#526176")
        fig.text(.055, .045, footnote, fontsize=12, color="#526176")

    fig, ax = plt.subplots(figsize=(12, 7.5))
    keys = ["primary64", "additional80"]
    for i, key in enumerate(keys):
        m = data["metrics"][key]
        for offset, variant, color in ((-.18, "before", gray), (.18, "after", teal)):
            s = m[variant]; v = s["recall_at_16"]
            ax.barh(i + offset, v, height=.28, color=color, label=variant.title() if i == 0 else None)
            ax.text(.02, i + offset, f"{variant.title()}: {s['hits_at_16']}/{s['cases']}  ·  {v:.1%}", va="center", fontsize=18, color="white", weight="bold")
    ax.set(yticks=[0, 1], yticklabels=["Original 64\nUsed for selection", "Additional 80\nRegression check"], xlim=(0, 1.02), ylim=(1.6, -.55), xlabel="Expected authority present within top 16")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks([0, .25, .5, .75, 1])
    fig.subplots_adjust(left=.22, right=.96, bottom=.20, top=.78)
    title(fig, "Eight of the original nine misses recovered", "Same cases, corpus and 16-authority budget · Only the query planner changed", "All 144 combined: 126/144 → 140/144. Development evidence; independent evaluation remains separate.")
    fig.savefig(out / FIGURES[0], dpi=160, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8.5)); ax.axis("off")
    rows = data["original_misses"]
    columns = [[r["event_id"], r["hazard"].replace("_", " "), "Outside top 16",
                f"Rank {r['after_rank']}" if r["after_rank"] is not None else "Still missing"] for r in rows]
    table = ax.table(cellText=columns, colLabels=["Case", "Family", "Before", "After"], loc="center", cellLoc="left",
                     colLoc="left", colWidths=[.19, .25, .27, .25])
    table.auto_set_font_size(False); table.set_fontsize(17); table.scale(1, 2.6)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#e9eff4" if i == 0 else "#f5f7f9" if i % 2 else "white")
        if i == 0: cell.set_text_props(weight="bold")
        if i and j == 3: cell.set_text_props(color=teal if rows[i - 1]["after_rank"] is not None else orange, weight="bold")
    fig.subplots_adjust(left=.045, right=.97, bottom=.12, top=.79)
    title(fig, "Every original miss stays in the record", "Eight recovered · One remains outside the candidate list · Lower rank is better", "Nine rows reuse five incident wordings. These are selected synthetic cases, not independent field trials.")
    fig.savefig(out / FIGURES[1], dpi=160, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7.5)); m = data["metrics"]["all144"]
    for i, k in enumerate((1, 4, 8, 16)):
        for offset, version, color in ((-.18, "before", gray), (.18, "after", teal)):
            value = m[version][f"hits_at_{k}"]
            ax.barh(i + offset, value, height=.29, color=color, label=version.title() if i == 0 else None)
            ax.text(value + 2, i + offset, f"{value}/144", va="center", fontsize=16, weight="bold")
    ax.set(yticks=range(4), yticklabels=["First result", "Top 4", "Top 8", "Top 16"], xlim=(0, 165), ylim=(3.65, -.65), xlabel="Cases with the expected authority within the cutoff")
    ax.set_xticks([0, 36, 72, 108, 144])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.965, .855),
               ncol=2, frameon=False, fontsize=14)
    fig.subplots_adjust(left=.15, right=.96, bottom=.2, top=.79)
    title(fig, "Better coverage, with ranking tradeoffs", "All 144 public synthetic cases · Before and after use identical inputs", "Four former #1 results moved to #2; 15 case ranks worsened overall. No top-16 targets were lost.")
    fig.savefig(out / FIGURES[2], dpi=160, facecolor="white"); plt.close(fig)


def sources():
    return [f"{DATA}/{name}.json" for name in ("before", "after", "exploration", "protocol", "candidate_freeze")] + [
        "code/capture_retrieval_run.py", "code/report_retrieval_comparison.py", "code/autonomous_investigator.py",
        "code/public_results.py"]


def outputs():
    return [REPORT, "docs/results/retrieval_comparison.json", "docs/results/retrieval_cases.csv"] + ["docs/results/" + name for name in FIGURES]


def prepare(root):
    before, after = (read(root / DATA / (name + ".json")) for name in ("before", "after"))
    frozen = read(root / DATA / "candidate_freeze.json")
    if frozen["source_lf_sha256"] != after["input_and_code_sha256_lf"]["code/autonomous_investigator.py"]:
        raise ValueError("After-run does not match the frozen candidate")
    for name, expected in after["input_and_code_sha256_lf"].items():
        if file_hash(root / name) != expected:
            raise ValueError("Current source/input differs from the after-run: " + name)
    data = compare(before, after)
    # Narrative describes this named historical experiment; changed outcomes
    # require an explicitly reviewed interpretation rather than stale prose.
    if [(data["metrics"][c]["before"]["hits_at_16"], data["metrics"][c]["after"]["hits_at_16"]) for c in COHORTS] != [(55, 63), (71, 77), (126, 140)]:
        raise ValueError("Comparison outcomes changed; review the interpretation")
    if len(data["top1_losses"]) != 4 or len(data["rank_regressions"]) != 15:
        raise ValueError("Rank regressions changed; review the interpretation")
    return data, markdown(data, before, after, read(root / DATA / "exploration.json"))


def check(root=ROOT, fresh=None):
    data, prose = prepare(root)
    if fresh is not None:
        saved = read(root / DATA / "after.json"); candidate = read(fresh)
        validate_capture(candidate)
        for key in ("records", "input_and_code_sha256_lf", "mode", "top_k"):
            if candidate[key] != saved[key]:
                raise ValueError("Fresh candidate run differs from saved after-run: " + key)
        if not metrics_equal(saved["metrics"], candidate["metrics"]):
            raise ValueError("Fresh candidate run differs from saved after-run: metrics")
    expected_text = {REPORT: prose, "docs/results/retrieval_comparison.json": json_text(data),
                     "docs/results/retrieval_cases.csv": csv_text(data)}
    for path, value in expected_text.items():
        if (root / path).read_text(encoding="utf-8") != value:
            raise ValueError("Generated comparison is stale: " + path)
    manifest = read(root / "docs/results/retrieval_manifest.json")
    if manifest["inputs_sha256_lf"] != {p: file_hash(root / p) for p in sources()} or manifest["outputs_sha256_lf_text_raw_png"] != {p: artifact_hash(root / p) for p in outputs()}:
        raise ValueError("Comparison integrity check failed")
    return data


def artifact_hash(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.suffix == ".png" else file_hash(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--candidate-capture", type=Path)
    args = parser.parse_args()
    if args.check:
        check(fresh=args.candidate_capture)
        print("Matched retrieval comparison verified: inputs, counts, ranks, report and figures agree.")
        return
    data, prose = prepare(ROOT)
    out = ROOT / "docs/results"; out.mkdir(parents=True, exist_ok=True)
    draw(data, out)
    (ROOT / REPORT).write_text(prose, encoding="utf-8")
    (out / "retrieval_comparison.json").write_text(json_text(data), encoding="utf-8")
    (out / "retrieval_cases.csv").write_text(csv_text(data), encoding="utf-8", newline="")
    manifest = {"schema": "retrieval_comparison_manifest_v1", "inputs_sha256_lf": {p: file_hash(ROOT / p) for p in sources()},
                "outputs_sha256_lf_text_raw_png": {p: artifact_hash(ROOT / p) for p in outputs()}}
    (out / "retrieval_manifest.json").write_text(json_text(manifest), encoding="utf-8")
    print("Generated readable retrieval comparison, three figures, all-case CSV and integrity manifest.")


if __name__ == "__main__":
    main()
