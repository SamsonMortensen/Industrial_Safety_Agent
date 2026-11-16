# A useful answer should help with the condition

An observation can describe a credible hazard without giving us enough information for a final citation. The response should still help: explain the concern, suggest a practical next step, show the relevant evidence, and identify what remains uncertain. `REVIEW` should not end the conversation.

The first implementation covers a bounded set of development situations. The [saved results](USEFUL_ASSESSMENT_RESULTS.md) show what improved, which decisions remain unresolved, and the cost of withholding unsupported conclusions. This document explains that approach and the next data experiment; it does not claim that the system can resolve every situation or replace a site assessment.

## The elevated charger example

Consider a machine charger sitting on a platform about six feet above an aisle. It is not bolted down and can be moved easily. A useful response should start with the condition:

> The charger may shift or fall if it is unstable, pulled by its cable, or struck. If that could expose people below, keep them clear of the possible fall area and report the condition for an authorized inspection. Have qualified staff check the support, retention, cable routing and installation requirements before changing the installation.

That advice does not depend on identifying the exact truck first. The equipment model can help determine the permitted mounting method later. “Not bolted” is not the same as “no acceptable restraint,” and a charger designed to be portable needs different consideration from a fixed installation.

The supporting evidence should match the concern:

- [29 CFR 1910.303(b)(8)(i)](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.303) addresses securing mounted electrical equipment. Paragraph (b)(2) addresses instructions included in listing or labeling. These are installation questions, not proof that bolts are required in every case.
- [29 CFR 1910.28(c)](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.28) addresses employees exposed to falling objects. An excerpt about a person falling through a hole does not explain a charger falling from a platform. The height of the charger also does not establish that a person is working at that height.
- If this is industrial-truck battery charging, [29 CFR 1910.178(g)](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.178) may add relevant charging-area considerations. Its [charging-only directive](https://www.osha.gov/enforcement/directives/std-01-11-004) must be considered alongside paragraph (g)(2); charging, battery removal and maintenance are different activities.

Two useful follow-up questions are whether people can enter the area below and what prevents the charger from moving off its support. The exact model and manual can follow when installation details matter. Questions should refine the answer, not delay the initial explanation of an observed concern.

This pattern also applies to the other notebook situations. Separate a supported concern from missing facts, a verified control from an assumed control, and an unchanged scene from a safety clearance. A retrieved citation remains a candidate until its requirements fit the facts.

## More data has to address a measured gap

The default notebook uses word-based retrieval and scoped policy checks. It makes no language-model requests and does not train. Feeding a model more data cannot change that execution path by itself.

First, improve how the walkthrough finds and presents evidence. Retrieve the relevant paragraph with its surrounding conditions and exceptions. Give useful guidance for conditions outside the existing automatic-decision rules. Preserve the original observation, and label inferred possibilities as possibilities.

Next, expand the evidence collection deliberately:

1. **Versioned regulations and interpretations.** Use [eCFR developer resources](https://www.ecfr.gov/reader-aids/ecfr-developer-resources) and [GPO bulk data](https://www.govinfo.gov/developers). Preserve subsection structure, jurisdiction, effective or edition dates, retrieval date, source URL and content hash. Keep the existing benchmark corpus frozen. A current-source collection needs its own version.
2. **Relevant equipment instructions.** Add the exact manufacturer's manual when known, with model, revision and reuse permissions. Keep instructions tied to the equipment they describe. A manual for another charger is not evidence about this one.
3. **Reviewed incident and normal-operation examples.** [OSHA incident datasets](https://www.osha.gov/data) can broaden vocabulary and observed conditions. [Severe Injury Reports](https://www.osha.gov/severe-injury-reports) cover reported severe injuries under federal OSHA jurisdiction; they exclude fatalities and state-plan incidents. They are not a balanced set of safe and unsafe operations or automatic violation labels. Add normal conditions, verified controls, uncertainty and plausible but irrelevant matches.

Keep approved knowledge retrievable with its provenance. Do not treat every retrieved document, generated answer or automatic development label as a trusted precedent. Source updates need review for applicability and supersession as well as text changes.

## A bounded first batch

Start with a target of **120 reviewed development examples from at least 30 source groups**: roughly 40 observed hazards, 40 conditions with verified controls or no relevant exposure, and 40 incomplete or ambiguous conditions. Include elevated equipment, charging, electrical access, spills, lifting equipment and worker fall exposure. Variations from one incident or manual example belong to the same group.

Assign whole groups before tuning. Aim for about 80 examples for development and 40 frozen for one comparison; group integrity takes priority over exact counts. AI can suggest labels and evidence, but the comparison needs reviewed references. Automatic development labels remain distinct from those references. If we inspect the frozen comparison to improve the candidate, record that use and prepare fresh groups for the next comparison.

Keep the existing calibration reserve and independent evaluation set separate. This small batch checks targeted development behavior; it cannot establish field performance or a reliable rare-event error rate.

## Measure whether the answer helps

Compare the same cases before and after each change. Report missed hazards and false alerts separately and weight them equally in the first comparison. Also record unresolved cases, decision coverage, and whether a reviewer found the suggested next step useful and supported by the observation.

For evidence, report retrieval at ranks 1, 5 and 16, plus whether the selected passage actually supports the claim. Citation membership and citation entailment are different checks. Record latency, model requests and compute cost alongside quality.

Try semantic retrieval or a reranker if relevant passages still rank poorly. Consider fine-tuning only for errors that remain after evidence and routing work, such as unsupported conclusions, poor fact extraction or unreliable output structure. Test one change at a time against the frozen baseline. A larger corpus or training run earns its place by improving those measurements.
