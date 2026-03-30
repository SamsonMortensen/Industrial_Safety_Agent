# Autonomous AI Agent: Industrial Safety & Compliance Auditor

## Overview

Manual review of intermodal yard logs is slow, error‑prone, and ineffective at identifying cascading safety risks before they escalate into incidents. This project implements an autonomous AI agent—built with LangChain and a locally hosted large language model—to continuously audit heavy logistics operations against federal safety and compliance standards, with **zero cloud data exposure**.

The system demonstrates how agentic AI can augment industrial safety oversight by automating regulatory reasoning at operational scale.

## Technology Stack

**Languages & Frameworks**

- Python  
- LangChain  
- Pandas  

**Core Techniques**

- Agentic AI workflows  
- Prompt engineering for regulatory reasoning  
- Automated ETL for unstructured text  
- Ground‑truth validation and evaluation  

**Data Sources**

- Synthetic intermodal yard operations logs  
- Sanitized federal regulatory text (OSHA, FRA)

## Problem Statement

Operations managers cannot realistically cross‑reference thousands of daily equipment movements against hundreds of pages of federal compliance codes. As a result, violations often surface only during infrequent external audits—after risk has already accumulated.

This project addresses that gap through three tightly integrated components:

1. **Regulatory ETL Pipeline** (`main.ipynb`)

Parses and structures unstructured OSHA and FRA regulatory text into a queryable, machine‑interpretable format.

2. **Synthetic Ground‑Truth Dataset** (`daily_yard_log.csv`)
    
   A realistic operations log seeded with hidden, real‑world violation patterns, including:
   - Operator fatigue and shift overruns  
   - Crane clearance and equipment proximity breaches  
   - Hazardous materials exposure risks  

3. **Autonomous Compliance Agent**
   
   A fully local, LLM‑powered agent that iterates through operational logs, maps activities to regulatory constraints, flags violations, and produces structured executive‑level audit reports—without transmitting proprietary data to external services.

## Key Results

**Accurate violation detection**

The agent successfully identified a 14.5‑hour operator shift and a hydraulic spill adjacent to a pedestrian zone; both valid FRA/OSHA violations that typically emerge only during formal audit cycles.

**Operational automation**

The system functions as a continuous digital safety monitor, reducing manual log review and allowing human supervisors to focus on remediation, escalation, and prevention rather than compliance triage.

**Ground‑truth validation**

Evaluation against a pre‑seeded dataset with known violations enables quantifiable precision and recall, providing a more rigorous assessment than qualitative or prompt‑only demonstrations.

## Impact

This project illustrates how locally deployed, agentic AI systems can:
- Scale regulatory oversight without increasing headcount  
- Detect safety risks earlier in operational workflows  
- Preserve data sovereignty in sensitive industrial environments  

The approach generalizes to other compliance‑heavy domains where rules are complex, data is proprietary, and real‑time oversight is critical.
