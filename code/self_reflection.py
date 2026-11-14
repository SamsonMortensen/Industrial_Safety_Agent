"""Check audit replies and ask for a critique when the checks fail.

Citation membership and a few explicit event checks can reject unsupported
answers. They do not establish legal correctness or eliminate model errors.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
from ollama_config import ollama_base_url
from audit_outcomes import guard_grounded_decision, parse_audit_response

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
VALID_SECTIONS_FILE = JSON_DIR / "valid_sections.json"
STATUTES_FILE = JSON_DIR / "statutes.json"

OLLAMA = ollama_base_url()
CHAT_MODEL = "qwen3.5:9b"

# Ollama otherwise allocates the model's full context window (262144 for
# qwen3.5:9b), which reserves roughly 15GB of KV cache and pushes the model
# onto CPU. Audit prompts run to about 1,500 tokens.
NUM_CTX = int(os.getenv("AUDIT_NUM_CTX", "8192"))

# Matches the audit path: a critique retry on CPU is minutes, not seconds.
CRITIQUE_TIMEOUT = int(os.getenv("AUDIT_TIMEOUT", "900"))


class StatutoryGroundedVerifier:
    """Verifies citation existence against federal law and factual consistency with telemetry."""

    def __init__(self, valid_sections_file: Path = VALID_SECTIONS_FILE):
        self.valid_sections_file = valid_sections_file
        self.valid_sections = set()
        self._load()

    def _load(self):
        if self.valid_sections_file.exists():
            try:
                data = json.loads(self.valid_sections_file.read_text(encoding="utf-8"))
                self.valid_sections = {
                    s for group in data.get("by_title", {}).values() for s in group
                }
            except Exception as e:
                print(f"[Verifier] Error loading valid sections: {e}")
        if STATUTES_FILE.exists():
            try:
                statutes = json.loads(STATUTES_FILE.read_text(encoding="utf-8"))
                self.valid_sections.update(statutes.get("sections", []))
            except Exception as e:
                print(f"[Verifier] Error loading statutes: {e}")

    def verify(
        self,
        parsed_result: Dict[str, Any],
        row: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check grounding validity and logical consistency.

        Returns:
            {
                "is_valid": bool,
                "confidence": float,
                "issues": List[str],
                "retrieved_sections": List[str],
                "citation_exists": bool,
                "citation_retrieved": bool,
            }
        """
        status = parsed_result.get("status", "UNPARSED")
        citation = parsed_result.get("citation")
        issues = []
        confidence = 1.0

        retrieved_sections = {
            c.get("section") for c in retrieved_chunks if "section" in c
        }

        # 1. Format and parsing check
        if status not in {"CLEAR", "VIOLATION"}:
            issues.append(
                "The decision remains unresolved and cannot be verified as a binary outcome."
                if status == "REVIEW" else "Response failed strict formatting syntax."
            )
            confidence = 0.0
            return {
                "is_valid": False,
                "confidence": confidence,
                "issues": issues,
                "retrieved_sections": list(retrieved_sections),
                "citation_exists": False,
                "citation_retrieved": False,
            }

        # 2. Logic check between Status and Citation
        if status == "CLEAR" and citation is not None:
            issues.append(
                f"Status is CLEAR but cited section {citation}. If CLEAR, citation must be NONE."
            )
            confidence *= 0.5

        if status == "VIOLATION" and citation is None:
            issues.append(
                "Status is VIOLATION but no specific section number was cited."
            )
            confidence *= 0.3

        # Sections 228.5 and 228.7 define terms and how duty time is counted.
        # They do not independently impose a maximum-duty prohibition.
        if status == "VIOLATION" and citation in {"228.5", "228.7"}:
            issues.append(
                f"Cited section {citation} is non-dispositive and cannot independently establish a violation."
            )
            confidence *= 0.3

        # 3. Federal statutory existence check
        citation_exists = False
        if citation:
            citation_exists = citation in self.valid_sections
            if not citation_exists:
                issues.append(
                    f"Cited section '{citation}' does NOT exist in Title 29 or Title 49 CFR (Hallucinated Law)."
                )
                confidence = 0.0

        # 4. Retrieval grounding check
        citation_retrieved = False
        if citation:
            citation_retrieved = citation in retrieved_sections
            if not citation_retrieved and citation_exists:
                issues.append(
                    f"Cited section '{citation}' exists in federal law, but was NOT in the retrieved context excerpts."
                )
                confidence *= 0.7

        # 5. Statutory threshold check
        try:
            shift_hrs = float(row.get("Operator_Shift_Hours", 0.0))
        except (ValueError, TypeError):
            shift_hrs = 0.0

        if citation in {"21103", "228.405"}:
            if shift_hrs <= 12.0:
                issues.append(
                    f"Cited an hours-of-service duty limit for shift duration of {shift_hrs} hrs, "
                    f"which does not exceed the 12.0-hour statutory limit."
                )
                confidence *= 0.3
        incident_scope = " ".join(
            str(row.get(field, ""))
            for field in ("Equipment_Type", "Location", "Reported_Incident")
        ).lower()
        if citation == "21103" and not any(
            term in incident_scope
            for term in ("train employee", "switch employee", "covered service")
        ):
            issues.append(
                "49 U.S.C. 21103 requires train-employee covered-service scope, which the event does not establish."
            )
            confidence *= 0.3
        if citation == "228.405" and not any(
            term in incident_scope
            for term in ("commuter", "intercity passenger", "passenger train")
        ):
            issues.append(
                "49 CFR 228.405 is limited to commuter or intercity passenger train employees."
            )
            confidence *= 0.3

        is_valid = len(issues) == 0

        return {
            "is_valid": is_valid,
            "confidence": round(confidence, 3),
            "issues": issues,
            "retrieved_sections": sorted(list(retrieved_sections)),
            "citation_exists": citation_exists if citation else None,
            "citation_retrieved": citation_retrieved if citation else None,
        }


class SelfReflectionCritic:
    """Dispatches self-reflection and re-grounding critique when potential hallucinations or errors occur."""

    def __init__(self, verifier: StatutoryGroundedVerifier, model: str = CHAT_MODEL):
        self.verifier = verifier
        self.model = model

    async def critique_and_reground_async(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        parsed_result: Dict[str, Any],
        row: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        sem: Any,
        *,
        record_preference: bool = True,
    ) -> Tuple[Dict[str, Any], bool, str]:
        """Run a critique if verification fails.

        Evaluation callers can disable preference recording so their cases do
        not become persistent training candidates.

        Returns:
            (refined_parsed_result, was_corrected, critique_log)
        """
        verification = self.verifier.verify(parsed_result, row, retrieved_chunks)
        retrieved_sections = {chunk.get("section") for chunk in retrieved_chunks if chunk.get("section")}
        if parsed_result.get("status") == "REVIEW":
            # A request for missing evidence is not a formatting failure to retry away.
            result = dict(parsed_result, confidence=0.0, critique_applied=False)
            return result, False, "Decision remains unresolved pending evidence or review."
        if verification["is_valid"]:
            result = guard_grounded_decision(parsed_result, retrieved_sections, self.verifier.valid_sections)
            result["confidence"] = verification["confidence"] if result["status"] != "REVIEW" else 0.0
            result["critique_applied"] = False
            return result, False, ""

        # Construct Critique Prompt
        issues_str = "\n".join(f"- {issue}" for issue in verification["issues"])
        critique_prompt = f"""{prompt}

CRITIQUE & SELF-REFLECTION REVIEW:
Your previous output was:
STATUS: {parsed_result.get("status")}
CITATION: {parsed_result.get("citation_raw")}
REASON: {parsed_result.get("reason")}

VERIFICATION FOUND THE FOLLOWING ISSUES:
{issues_str}

Please carefully re-evaluate the YARD EVENT against the RETRIEVED REGULATIONS.
- If the event does NOT violate any retrieved regulation or statutory limit, output STATUS: CLEAR and CITATION: NONE.
- If the governing authority, a required fact, or the asserted control cannot be established, output STATUS: REVIEW and CITATION: NONE. Absence of evidence is not a clearance.
- Do NOT cite any section number that does not appear in RETRIEVED REGULATIONS.

Answer strictly in this format and nothing else:
STATUS: CLEAR, VIOLATION, or REVIEW
CITATION: exact section number present in the retrieved regulations, or NONE
REASON: one concise sentence.
"""
        async with sem:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": critique_prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_predict": 250, "num_ctx": NUM_CTX},
            }
            try:
                async with session.post(
                    f"{OLLAMA}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=CRITIQUE_TIMEOUT),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    raw_corrected = data["message"]["content"]
            except Exception as e:
                print(f"[SelfReflection] LLM critique call failed: {e}")
                raw_corrected = ""

        # A failed critique remains failed; it must not inherit the prior verdict.
        refined = {
            **parse_audit_response(raw_corrected),
            "critique_applied": True,
            "original_verdict": f"{parsed_result.get('status')}: {parsed_result.get('citation')}",
            "critique_issues": verification["issues"],
        }
        refined = guard_grounded_decision(refined, retrieved_sections, self.verifier.valid_sections)

        # Second verification pass on refined result
        second_verif = self.verifier.verify(refined, row, retrieved_chunks)
        refined["confidence"] = second_verif["confidence"]
        refined["second_pass_valid"] = second_verif["is_valid"]
        if not second_verif["is_valid"] and refined["status"] in {"CLEAR", "VIOLATION"}:
            refined.update(status="REVIEW", citation=None, citation_raw="NONE", confidence=0.0,
                           reason="The critique did not resolve the evidence checks; reviewer assessment is required.",
                           decision_source="critique_verification_guard")

        was_corrected = second_verif["is_valid"] and (
            refined["status"] != parsed_result.get("status") or refined["citation"] != parsed_result.get("citation")
        )
        critique_summary = (
            f"Critique resolved issues: {', '.join(verification['issues'])}"
            if second_verif["is_valid"] else "Critique left the decision unresolved."
        )

        # If correction occurred, record preference optimization pair to disk
        if was_corrected and record_preference:
            dpo_file = JSON_DIR / "dpo_pairs_raw.json"
            existing_pairs = []
            if dpo_file.exists():
                try:
                    existing_pairs = json.loads(dpo_file.read_text(encoding="utf-8"))
                except Exception:
                    existing_pairs = []

            pair_record = {
                "log_id": row.get("Log_ID"),
                "prompt": prompt,
                "chosen": f"STATUS: {refined['status']}\nCITATION: {refined['citation_raw']}\nREASON: {refined['reason']}",
                "rejected": f"STATUS: {parsed_result.get('status')}\nCITATION: {parsed_result.get('citation_raw')}\nREASON: {parsed_result.get('reason')}",
                "critique_issues": verification["issues"],
                "row": row,
            }
            # Append if not already recorded
            if not any(
                p.get("log_id") == pair_record["log_id"] for p in existing_pairs
            ):
                existing_pairs.append(pair_record)
                dpo_file.parent.mkdir(parents=True, exist_ok=True)
                dpo_file.write_text(
                    json.dumps(existing_pairs, indent=2), encoding="utf-8"
                )

        return refined, was_corrected, critique_summary
