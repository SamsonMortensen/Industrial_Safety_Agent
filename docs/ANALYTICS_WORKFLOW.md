# How the analytics are produced

The [results report](RESULTS.md) follows the analytical work from data preparation to a decision: profile the source, define the target and split, compare a baseline, inspect errors, quantify uncertainty, and retain the evidence. This is a portfolio of methods and saved experiments, including ties and unsuccessful changes.

## Reproduce the report

From the repository root, Python 3.11 or newer can recompute the comparison metrics and verify the report with no additional packages:

```powershell
python code/public_results.py --check
python -m unittest discover -s tests -p test_public_results.py -v
```

To regenerate the Markdown report, three PNG figures, metrics JSON, and provenance manifest:

```powershell
python -m pip install -r requirements-results.txt
python code/public_results.py
python code/public_results.py --check
```

Use the same Python interpreter for both commands. The plots use Matplotlib's noninteractive renderer; no browser, GPU, private source folder, or model service is required. `--check` verifies numerical recomputation, narrative generation, source hashes, and stored figure hashes. It does not rerun model inference or independently validate human labels. The public manifest normalizes text to LF before hashing, so Windows and Linux checkouts agree; PNGs are hashed byte-for-byte. Private completion-artifact hashes retain their original raw-byte convention. Figure bytes can differ with rendering environments; numerical outputs should agree.

## Evidence and metric definitions

The Person experiments use a frozen `google/siglip2-base-patch16-224` image encoder at revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, sampled frame features, and a logistic classification head. The rounds sample 16 frames per new interval and aggregate features across those frames. A Person target means visible somewhere in the reviewed interval, not in every frame. The Round 2/3 classification threshold is fixed at 0.5. The public snapshot records the encoder revision and a hash of each original protocol without exporting its private paths.

The historical Person footage comes from NVIDIA's **PhysicalAI SDG-Warehouse**, recorded by the local intake registry as `nvidia/PhysicalAI-SDG-WareHouse` revision `d5b88d3abcf659f304a107f4336b71b4e2159133`. The publisher's [current dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes) identifies NVIDIA as the owner and OpenMDW-1.1 as its license. This report publishes project-generated prediction/target summaries, not footage or upstream annotation files. The Person targets were reviewed for this project and are distinct from the publisher's scenario tags. These runs predate the separate SmartSpaces intake; SmartSpaces results are not included here.

| Analytical choice | Implementation and purpose |
| --- | --- |
| Report denominators | Every error count is paired with its positive, negative, clip, or source-group denominator. |
| Equal error costs | Equal-cost error = (false positives + false negatives) / all clips. |
| Class balance | Balanced accuracy = (recall + specificity) / 2. This differs from accuracy when class prevalence differs. |
| Correlated camera views | Views from one source group are not counted as independent trials. A group succeeds only when all its views are correct. |
| Uncertainty | Two-sided 95% Clopper–Pearson bounds on fully correct groups. They assume independent, similarly distributed source groups, which synthetic assets may not satisfy. |
| Probability diagnostics | Brier = mean squared score error. Log loss penalizes confident errors. Each source group gets equal total weight; its views share that weight. |
| Matched comparisons | Compare models on the same validation rows within a run. Do not compare percentages across different rounds as a learning curve. |
| Output parsing | The language-model benchmark uses strict parsing. Unparsed outputs are errors; formatting performance is shown separately from citation correctness. |
| Provenance | SHA-256 input/output records and completion-manifest checks connect public summaries to saved runs. Hashes establish identity, not truth or independence. |

For a single-class slice, the absent class's rate and balanced accuracy are undefined, not zero. Exact intervals describe the group success rate, not the population probability that a random frame is safe. Synthetic development results do not estimate field prevalence.

## Exporting a development snapshot

The committed `data/public_development_runs.json` contains anonymous validation case/group identifiers, binary Person targets, predictions, model scores, numeric coverage, encoder revision, completion timestamps, and source-artifact hashes. Group IDs are local to each run; repeated validation cohorts must not be pooled as new data. The calibration experiment reuses Round 2's validation cohort.

The exporter accepts explicit completed development runs. It verifies completion hashes, excludes training/calibration rows from the published prediction table, rejects independent-evaluation or promoted-model inputs, and recomputes counts, group intervals, and calibration metrics before writing. No media, original sample IDs, reviewer identities, approval receipts, private paths, or free-form label comments are copied. Source hashes point to private artifacts that public readers cannot inspect; public readers can reproduce the exported metrics, not audit the underlying footage.

Maintainers with the source run folders can regenerate that snapshot:

```powershell
python code/export_public_development.py --pilot PATH_TO_PILOT --round2 PATH_TO_ROUND2 --round3 PATH_TO_ROUND3 --calibration PATH_TO_CALIBRATION --out data/public_development_runs.json
python code/public_results.py
python code/public_results.py --check
```

The report intentionally covers four named historical experiments. New run types require an explicit exporter/report update and reviewed interpretation. Do not overwrite an old experiment to manufacture a continuous improvement curve. Keep calibration fitting, development validation, and independently reviewed evaluation separately identified.

## Adding the next result

1. Freeze the question, comparison models, source-group split, target definition, and primary metric before a run. Record fit, calibration, and evaluation roles.
2. Save predictions and probabilities with model/data/code fingerprints. Report both false alerts and misses, including all compared baselines and an unsuccessful candidate when relevant.
3. Export only the intended development evidence. Verify the anonymous export against the saved completion manifest; keep sealed evaluation data out of the publication workflow.
4. Regenerate figures, tables, and hashes. Review the interpretation, sample limitations, source rights, and differences between fitting and out-of-sample results.
5. Run the report checker and normal repository tests. Publish the reviewed result as a new entry. Source publication and model promotion remain separate actions.

The current public snapshot does not reproduce vision training or the raw-media review process. Those require the separately versioned research tooling and permitted source data. No model is promoted by generating this report.
