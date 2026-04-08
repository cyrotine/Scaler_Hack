---
title: InsureLink-v1
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
- openenv
- insurance
- finance
- tool-use
- reasoning
---

# InsureLink-v1: Car & Bike Insurance Agent Environment

InsureLink-v1 is a specialized evaluation environment designed to test LLM agents on their ability to act as insurance claims and policy assistants. It moves beyond simple text retrieval, requiring agents to perform stateful updates and complex logic-based claim arbitrations.

## 🌟 Motivation
In the insurance industry, **First Notice of Loss (FNOL)** and policy management are high-volume tasks. This environment evaluates if an agent can:
1.  **Retrieve** accurate policy data.
2.  **Execute** stateful modifications (e.g., updating vehicle identifiers).
3.  **Reason** through policy constraints (e.g., determining if a specific coverage type covers "own-damage").

---

## 🕹️ Action & Observation Space

### Action Space (Tools)
The agent interacts with the environment using a suite of specialized tools:
* `get_policy_details(policy_id: str)`: Fetches full policy data, including vehicle info, deductible, and coverage type.
* `update_vehicle(policy_id: str, vin: str)`: Updates the Vehicle Identification Number in the environment state.
* `calculate_claim_payout(policy_id: str, repair_cost: float)`: A complex tool that evaluates eligibility based on `coverage_type` and calculates net payout after deductibles.

### Observation Space
Observations are returned as structured JSON objects containing:
* **Policy Fields**: `coverage_type` (Comprehensive/Third-Party), `deductible`, `vehicle_year`, etc.
* **Status Messages**: Success/Failure confirmations for updates.
* **Logic Payloads**: `eligible` (boolean), `payout` (float), and `reason` (string explanation).

---

## 📋 Task Descriptions

| Task | Difficulty | Description | Expected Reasoning |
| :--- | :--- | :--- | :--- |
| **Coverage Check** | **Easy** | Identify deductible for a specific user. | Direct retrieval and extraction of numerical data. |
| **Policy Update** | **Medium** | Update a vehicle's VIN following a new purchase. | Stateful interaction; verifying the change after update. |
| **Claim Arbitration** | **Hard** | Determine if a $2,000 repair is covered under a policy. | Logic-based denial (e.g., Third-Party policies do not cover own-damage). |

---

## 📈 Baseline Scores
The following scores were achieved using the **Qwen2.5-72B-Instruct** model via the Hugging Face Router.

| Task | Status | Score | Efficiency |
| :--- | :--- | :--- | :--- |
| 📋 Coverage Check | ✅ Passed | **100%** | 5 Steps |
| 🔧 Policy Update | ✅ Passed | **100%** | 5 Steps |
| ⚖️ Claim Arbitration | ✅ Passed | **100%** | 5 Steps |
| **OVERALL** | 🏆 **SUCCESS** | **100%** | **Avg: 5.0 Steps** |

---

## 🛠️ Setup and Usage

### Prerequisites
* Python 3.10+
* `uv` package manager (recommended)
* Docker (for local container testing)

### Local Installation
```bash
# Clone the repository
git clone [https://huggingface.co/spaces/prahr0526/InsureLink-v1](https://huggingface.co/spaces/prahr0526/InsureLink-v1)
cd InsureLink-v1

# Install dependencies and sync lockfile
uv sync
uv lock
```
# Scaler_Hack
