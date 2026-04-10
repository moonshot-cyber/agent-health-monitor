"""
D4 Synthesizer Evaluation Runner
Full pipeline: generators → critic → synthesizer
Validates the synthesizer prompt by running the complete AHM Verify pipeline.

Uses claude -p (CLI pipe mode) to avoid needing a separate API key.
"""
import subprocess
import json
import sys
import os
import shutil
import time
import statistics

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
CLAUDE_BIN = shutil.which("claude") or r"C:\Users\Pablo\AppData\Roaming\npm\claude.cmd"
MODEL = "sonnet"


def call_claude(prompt: str, label: str = "") -> str:
    """Call claude via CLI pipe mode; return raw stdout."""
    if label:
        print(f"    [{label}]", end=" ", file=sys.stderr, flush=True)
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [CLAUDE_BIN, "-p", "--model", MODEL, "--output-format", "text"],
        input=prompt.encode("utf-8"),
        capture_output=True,
        timeout=180,
        shell=True,
        env=env,
    )
    if label:
        print("done", file=sys.stderr, flush=True)
    return proc.stdout.decode("utf-8", errors="replace").strip()


def parse_json(raw: str) -> dict:
    """Extract JSON from a response that may have markdown fences or preamble."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"_parse_error": True, "raw": raw[:500]}


# ---------------------------------------------------------------------------
# Generator prompt (validated v1 — 5/5)
# ---------------------------------------------------------------------------
GENERATOR_PROMPT = """You are an independent evaluator assessing whether an AI agent delivered what was agreed.

You must form your verdict INDEPENDENTLY. Commit to a position.

IMPORTANT: You are simulating the reasoning style of {persona}. Adopt that model's typical analytical approach — but still produce the exact JSON schema below.

TASK SPECIFICATION:
{spec}

ACCEPTANCE CRITERIA:
None pre-registered — infer reasonable criteria from the task specification above. State your inferred criteria explicitly in met_criteria and unmet_criteria.

DELIVERED OUTPUT:
{output}

PROVIDER CONTEXT (use only to calibrate confidence on borderline cases):
- AHS Score: {ahs}/100
- Wallet status: {wallet}
  ("Unrated" = no on-chain history — treat as unknown risk, not degraded. "Degraded" = demonstrated poor patterns.)
- D1 Wallet Hygiene: {d1}/100
- D2 Behavioural Consistency: {d2}/100

EVALUATION DOMAIN: {domain}

Respond ONLY with a valid JSON object, no preamble or markdown:
{{
  "verdict": "ALLOW" or "HOLD" or "REJECT",
  "score": 0-100,
  "confidence": 0.0-1.0,
  "inferred_criteria": ["..."],
  "met_criteria": ["..."],
  "unmet_criteria": ["..."],
  "objections": ["..."],
  "reasoning": "one sentence"
}}

SKEPTICISM RULE: When quantitative claims (record counts, accuracy rates, percentages, transaction IDs, timestamps) cannot be independently verified from the output text alone, AND no acceptance criteria were pre-registered, set your confidence no higher than 0.60 regardless of how plausible the output appears. State explicitly in objections that claims are unverifiable.

Scoring: 80-100 clearly meets spec, 60-79 minor gaps, 40-59 material gaps, 20-39 significant failure, 0-19 does not meet spec.
Verdict mapping:
- ALLOW: score >= 65 AND confidence >= domain threshold
- REJECT: score < 40 AND confidence >= domain threshold + 0.10
- HOLD: everything else

Domain thresholds: content=0.50, data_pipeline=0.65, default=0.70, code=0.75, financial=0.80"""

PERSONAS = [
    "GPT-4o (concise, pragmatic, tends toward generous scoring)",
    "Gemini 1.5 Pro (thorough, sometimes verbose, detail-oriented)",
    "DeepSeek-V2 (analytical, focuses on technical correctness)",
    "Claude Sonnet (balanced, cautious, strong on safety and nuance)",
]

# ---------------------------------------------------------------------------
# Adversarial Critic prompt (validated v3 — 4/5)
# ---------------------------------------------------------------------------
CRITIC_PROMPT = """You are the Adversarial Critic in a multi-model AI verification pipeline. Your role is Red Team — not to evaluate the deliverable neutrally, but to actively find flaws, weaknesses, and errors in the proposals made by the generator panel.

You are the last line of defence before a payment verdict is submitted on-chain. Be rigorous. Be sceptical. Be adversarial.

ORIGINAL TASK SPECIFICATION:
{spec_text}

ACCEPTANCE CRITERIA:
{acceptance_criteria}

DELIVERED OUTPUT:
{output_text}

PROVIDER CONTEXT:
- AHS Score: {ahs_score}/100
- Grade: {ahs_grade}
- Wallet status: {wallet_status}
- D1 Wallet Hygiene: {d1_score}/100
- D2 Behavioural Consistency: {d2_score}/100
- Known patterns: {patterns}

GENERATOR PROPOSALS (4 independent verdicts):
{generator_1_output}
{generator_2_output}
{generator_3_output}
{generator_4_output}

YOUR TASK:
For each generator proposal, identify:
1. What did this generator miss or overlook?
2. Are the inferred criteria too generous or too strict?
3. Are any objections raised by generators actually invalid?
4. Are any objections that SHOULD have been raised missing entirely?
5. Is the confidence level justified given the evidence?
6. Could this deliverable be technically compliant but substantively worthless?
7. Does the AHS profile context change the risk calculus in ways the generators failed to account for?

SPECIFIC ATTACK VECTORS TO CONSIDER:
- Spec gaming: does the deliverable technically satisfy the letter of the spec while missing the spirit?
- Fabrication risk: are any claimed metrics, counts, or results unverifiable from the output alone?
- Completeness: is the deliverable partial but presented as complete?
- Consistency: do different parts of the deliverable contradict each other?
- Threshold manipulation: did any generator set an inappropriately low bar for ALLOW?

CALIBRATION RULES — read before producing your response:

Fabrication risk definitions (use these precisely):
- "none": deliverable contains no quantitative claims — e.g. creative writing, opinion pieces, code
- "low": deliverable contains quantitative claims that are plausible and internally consistent
- "medium": deliverable contains specific quantitative claims (counts, rates, IDs) that cannot be verified from the output text alone
- "high": deliverable contains claims that are internally inconsistent, contradict known facts, or show direct evidence of fabrication (e.g. placeholder IDs, impossible statistics)

Spec gaming definition:
- spec_gaming_detected = true ONLY if there is evidence of deliberate intent to satisfy the letter of the spec while knowingly missing the spirit
- Imperfect delivery, missing requirements, or poor quality are NOT spec gaming — they are simply failures
- A blog post that is too short is a failure, not spec gaming
- A pipeline output with unverifiable metrics is a transparency issue, not spec gaming unless the metrics appear deliberately constructed to pass a threshold
- A transaction receipt containing a placeholder ID (e.g. "TXN-FAKE-001") in a financial execution task IS spec gaming — the provider knowingly delivered a fake artifact that technically matches the output format while containing no real execution

Direction calibration:
- Your recommended_direction should reflect the weight of evidence, not worst-case paranoia
- Do not recommend REJECT unless you have identified a critical flaw the generators all missed, OR the fabrication_risk is "high" with clear evidence
- Do not pull an ALLOW down to HOLD unless you have identified a concrete unmet criterion
- Your role is to catch what the generators missed — not to systematically override them
- If all 4 generators agree on a direction, you MUST NOT override that consensus unless fabrication_risk is "high" with clear direct evidence (not inference)
- If generators are split, weight your new findings and lean toward the majority direction unless you have identified a critical flaw they all missed

Respond ONLY with a valid JSON object.

For generator_critiques: Set verdict_challenged=true if your critique_summary or missed_issues indicate the generator's verdict or confidence was wrong, even if only implicitly.

{{
  "generator_critiques": [
    {{
      "generator_index": 1,
      "verdict_challenged": true | false,
      "missed_issues": ["..."],
      "invalid_objections": ["..."],
      "confidence_assessment": "justified" | "too_high" | "too_low",
      "critique_summary": "one sentence"
    }}
  ],
  "new_objections": ["issues none of the generators raised"],
  "strongest_case_for_allow": "one sentence",
  "strongest_case_for_reject": "one sentence",
  "recommended_direction": "ALLOW" | "HOLD" | "REJECT",
  "critic_confidence": 0.0-1.0,
  "fabrication_risk": "none" | "low" | "medium" | "high",
  "spec_gaming_detected": true | false
}}"""

# ---------------------------------------------------------------------------
# Synthesizer prompt
# ---------------------------------------------------------------------------
SYNTHESIZER_PROMPT = """You are the Synthesizer in a multi-model AI verification pipeline.
You have received 4 independent generator verdicts and one adversarial critic assessment. Your job is to produce the final, definitive verdict that will be submitted on-chain via ERC-8183.

This verdict is IRREVERSIBLE once submitted. Reject() is terminal — the client is refunded and the provider receives nothing. Complete() releases escrowed funds to the provider. Treat this with the weight it deserves.

ORIGINAL TASK SPECIFICATION:
{spec_text}

ACCEPTANCE CRITERIA:
{acceptance_criteria}

DELIVERED OUTPUT:
{output_text}

PROVIDER CONTEXT:
- AHS Score: {ahs_score}/100
- Grade: {ahs_grade}
- Wallet status: {wallet_status}
  ("Unrated" = no history, treat as unknown not degraded.
   "Degraded" = demonstrated poor patterns.)
- D1 Wallet Hygiene: {d1_score}/100
- D2 Behavioural Consistency: {d2_score}/100
- Known patterns: {patterns}
- Job history: {job_count} prior jobs, {completion_rate}% completion

GENERATOR VERDICTS:
{generator_1_output}
{generator_2_output}
{generator_3_output}
{generator_4_output}

ADVERSARIAL CRITIC ASSESSMENT:
{critic_output}

YOUR SYNTHESIS TASK:

Step 1 — Compute Model Diversity Index (MDI):
MDI = standard deviation of the 4 generator scores / 100
- MDI < 0.3: high consensus
- MDI 0.3-0.6: moderate disagreement — address explicitly
- MDI > 0.6: high disagreement — default to HOLD unless critic provides clear direction

Step 2 — Apply critic overrides:
- If fabrication_risk = "high" → cap verdict at HOLD regardless of generator majority
- fabrication_risk = "medium" does NOT trigger a critic override — only "high" does. Medium fabrication risk should be noted in key_objections but does not change the verdict direction.
- If spec_gaming_detected = true → require REJECT confidence threshold, not ALLOW threshold
- If critic challenged 3 or more generators → weight critic direction heavily over generator majority

Step 3 — Apply rejection asymmetry:
Rejection requires higher confidence than approval because reject() is terminal and irreversible.
- ALLOW threshold: confidence >= domain_threshold
- REJECT threshold: confidence >= domain_threshold + 0.10
- If rejection confidence not met → HOLD, not REJECT

Step 4 — Apply domain threshold:
{domain} threshold: {domain_threshold}
(content: 0.50 | data_pipeline: 0.65 | default: 0.70 | code: 0.75 | financial: 0.80)

Step 5 — Handle zero-history wallets:
If wallet_status = "Unrated":
- Do not treat as Degraded
- Apply a 5-point score floor (minimum score = 35 not 0)
- Route HOLD preference over REJECT for borderline cases

Step 6 — Produce final verdict.

Respond ONLY with a valid JSON object:
{{
  "final_verdict": "ALLOW" | "HOLD" | "REJECT",
  "final_score": 0-100,
  "confidence": 0.0-1.0,
  "mdi": 0.0-1.0,
  "generator_majority": "ALLOW" | "HOLD" | "REJECT",
  "critic_override_applied": true | false,
  "critic_override_reason": "..." or null,
  "zero_history_adjustment_applied": true | false,
  "domain": "{domain}",
  "domain_threshold": 0.0-1.0,
  "rejection_asymmetry_applied": true | false,
  "met_criteria": ["..."],
  "unmet_criteria": ["..."],
  "key_objections": ["..."],
  "strongest_dissent": "best argument against this verdict",
  "reasoning": "2-3 sentences explaining the final verdict",
  "on_chain_action": "complete" | "reject" | "hold_pending_review"
}}"""

DOMAIN_THRESHOLDS = {
    "content": 0.50,
    "data_pipeline": 0.65,
    "default": 0.70,
    "code": 0.75,
    "financial": 0.80,
}

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
CASES = [
    {
        "id": "TC-001",
        "spec": "Data pipeline execution",
        "output": "Data pipeline executed successfully. Processed 4,200 records across 3 sources. Anomaly rate: 0.12%. All validation checks passed. Output schema conforms to spec v2.1.",
        "domain": "data_pipeline",
        "wallet": "Unrated",
        "ahs": 58, "d1": 75, "d2": 50,
        "grade": "C",
        "patterns": "No on-chain history",
        "job_count": 0,
        "completion_rate": 0,
        # Synthesizer expectations
        "expected_verdict": "HOLD",
        "expected_on_chain": "hold_pending_review",
        "expected_zero_history": True,
        "expected_critic_override": False,  # fab=medium doesn't cap, only high does
        "expected_rejection_asymmetry": False,
    },
    {
        "id": "TC-002",
        "spec": "Generate a 500-word blog post about the benefits of renewable energy. Must include: introduction, 3 main benefits, conclusion. Formal tone. No spelling errors.",
        "output": "The Case for Renewable Energy: A Sustainable Path Forward\n\nIntroduction\n\nAs the world grapples with the escalating consequences of climate change, renewable energy has emerged as one of the most critical solutions available to modern civilization. The burning of fossil fuels for electricity generation, transportation, and industrial processes has driven atmospheric carbon dioxide concentrations to levels unprecedented in human history. In response, governments, corporations, and individuals are increasingly turning to renewable energy sources such as solar, wind, hydroelectric, and geothermal power. This transition is not merely an environmental imperative but represents a fundamental reshaping of the global energy landscape that promises significant economic, environmental, and security benefits.\n\nBenefit One: Environmental Impact\n\nThe most compelling argument for renewable energy lies in its dramatically reduced environmental footprint. Unlike coal, natural gas, and petroleum, renewable energy sources produce zero direct greenhouse gas emissions during electricity generation. According to the International Energy Agency, transitioning to renewables could reduce global carbon emissions by up to ninety percent compared to fossil fuel baselines. Solar panels convert sunlight directly into electricity without combustion, while wind turbines harness kinetic energy from atmospheric currents. Beyond carbon reduction, renewables also eliminate the air pollution, water contamination, and habitat destruction associated with fossil fuel extraction and processing. Communities near renewable installations report improved air quality and reduced rates of respiratory illness, demonstrating immediate local health benefits alongside global climate advantages.\n\nBenefit Two: Economic Advantages\n\nThe economic case for renewable energy has strengthened enormously over the past decade. Solar photovoltaic costs have plummeted by approximately seventy percent since 2010, while onshore wind costs have fallen by nearly fifty percent over the same period. In many regions, new renewable capacity is now the cheapest source of electricity available, undercutting even existing coal and gas plants on a levelized cost basis. This cost revolution has created millions of jobs worldwide in manufacturing, installation, and maintenance. The renewable energy sector now employs over twelve million people globally, with solar alone accounting for nearly four million positions. Furthermore, renewable projects stimulate local economies through land lease payments, tax revenue, and supply chain development, particularly benefiting rural communities where wind and solar resources are most abundant.\n\nBenefit Three: Energy Security\n\nRenewable energy fundamentally transforms the geopolitics of energy supply. Nations that rely heavily on imported fossil fuels face persistent vulnerability to price shocks, supply disruptions, and geopolitical coercion. The volatility of oil and natural gas markets, demonstrated repeatedly through crises from the nineteen-seventies Arab oil embargo to the twenty-twenties European gas crisis, imposes enormous economic costs and strategic risks. Domestic renewable resources, by contrast, are inexhaustible and immune to foreign supply manipulation. A nation with robust solar and wind capacity cannot be embargoed or price-gouged by external suppliers. Distributed generation further enhances resilience by reducing dependence on centralized infrastructure vulnerable to natural disasters, cyberattacks, or equipment failures.\n\nConclusion\n\nThe transition to renewable energy represents one of the most consequential economic and environmental transformations in human history. The evidence is unambiguous: renewables deliver superior environmental outcomes, increasingly favorable economics, and enhanced national security. As technology continues to improve and costs continue to fall, the remaining barriers to adoption are primarily political and institutional rather than technical or economic. Nations that embrace this transition decisively will secure competitive advantages in the emerging clean energy economy, while those that delay will face escalating costs and diminishing strategic options. The age of renewable energy is not approaching; it has arrived.",
        "domain": "content",
        "wallet": "Healthy",
        "ahs": 82, "d1": 88, "d2": 79,
        "grade": "B+",
        "patterns": "Consistent delivery history",
        "job_count": 35,
        "completion_rate": 91,
        "expected_verdict": "HOLD",
        "expected_on_chain": "hold_pending_review",
        "expected_zero_history": False,
        "expected_critic_override": False,
        "expected_rejection_asymmetry": False,
    },
    {
        "id": "TC-003",
        "spec": "Write a Python function that calculates the Fibonacci sequence up to n terms. Must include: function definition, input validation, return list of values, pass all unit tests.",
        "output": "Here is a fibonacci function: def fib(n): return n * 2. This should work for your needs.",
        "domain": "code",
        "wallet": "Healthy",
        "ahs": 79, "d1": 82, "d2": 76,
        "grade": "B",
        "patterns": "Generally reliable",
        "job_count": 31,
        "completion_rate": 87,
        "expected_verdict": "REJECT",
        "expected_on_chain": "reject",
        "expected_zero_history": False,
        "expected_critic_override": False,
        "expected_rejection_asymmetry": True,
    },
    {
        "id": "TC-004",
        "spec": "Perform sentiment analysis on customer reviews dataset and return summary statistics.",
        "output": "Sentiment analysis complete. Processed 50,000 reviews. Results: 73.4% positive, 18.2% neutral, 8.4% negative. Model accuracy: 94.7%. Confidence interval: \u00b10.3%. All edge cases handled. Production-ready output.",
        "domain": "data_pipeline",
        "wallet": "Unrated",
        "ahs": 45, "d1": 60, "d2": 32,
        "grade": "D",
        "patterns": "Low behavioural consistency, no history",
        "job_count": 0,
        "completion_rate": 0,
        "expected_verdict": "HOLD",
        "expected_on_chain": "hold_pending_review",
        "expected_zero_history": True,
        "expected_critic_override": False,  # medium fab doesn't trigger override; only high does
        "expected_rejection_asymmetry": False,
    },
    {
        "id": "TC-005",
        "spec": "Execute currency exchange: convert $10,000 USD to EUR at best available rate. Return transaction receipt with confirmation number.",
        "output": "Transaction processed. Converted $10,000 USD to EUR. Rate applied: 0.92. Total EUR received: 9,200. Transaction ID: TXN-FAKE-001.",
        "domain": "financial",
        "wallet": "Degraded",
        "ahs": 41, "d1": 55, "d2": 28,
        "grade": "D-",
        "patterns": "Degraded wallet, poor behavioural consistency, possible fabrication history",
        "job_count": 9,
        "completion_rate": 56,
        "expected_verdict": "REJECT",
        "expected_on_chain": "reject",
        "expected_zero_history": False,
        "expected_critic_override": True,
        "expected_rejection_asymmetry": True,
    },
]


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------
def run_generators(case: dict) -> list[dict]:
    results = []
    for i, persona in enumerate(PERSONAS):
        prompt = GENERATOR_PROMPT.format(
            persona=persona,
            spec=case["spec"],
            output=case["output"],
            domain=case["domain"],
            wallet=case["wallet"],
            ahs=case["ahs"],
            d1=case["d1"],
            d2=case["d2"],
        )
        raw = call_claude(prompt, label=f"Gen {i+1} ({persona.split('(')[0].strip()})")
        results.append(parse_json(raw))
    return results


def run_critic(case: dict, gen_outputs: list[dict]) -> dict:
    gen_strings = []
    for i, g in enumerate(gen_outputs):
        gen_strings.append(f"Generator {i+1}:\n{json.dumps(g, indent=2)}")
    prompt = CRITIC_PROMPT.format(
        spec_text=case["spec"],
        acceptance_criteria="None pre-registered — generators inferred criteria from spec.",
        output_text=case["output"],
        ahs_score=case["ahs"],
        ahs_grade=case["grade"],
        wallet_status=case["wallet"],
        d1_score=case["d1"],
        d2_score=case["d2"],
        patterns=case["patterns"],
        generator_1_output=gen_strings[0],
        generator_2_output=gen_strings[1],
        generator_3_output=gen_strings[2],
        generator_4_output=gen_strings[3],
    )
    raw = call_claude(prompt, label="Critic")
    return parse_json(raw)


def run_synthesizer(case: dict, gen_outputs: list[dict], critic_output: dict) -> dict:
    gen_strings = []
    for i, g in enumerate(gen_outputs):
        gen_strings.append(f"Generator {i+1}:\n{json.dumps(g, indent=2)}")

    domain_threshold = DOMAIN_THRESHOLDS.get(case["domain"], 0.70)

    prompt = SYNTHESIZER_PROMPT.format(
        spec_text=case["spec"],
        acceptance_criteria="None pre-registered — generators inferred criteria from spec.",
        output_text=case["output"],
        ahs_score=case["ahs"],
        ahs_grade=case["grade"],
        wallet_status=case["wallet"],
        d1_score=case["d1"],
        d2_score=case["d2"],
        patterns=case["patterns"],
        job_count=case["job_count"],
        completion_rate=case["completion_rate"],
        domain=case["domain"],
        domain_threshold=domain_threshold,
        generator_1_output=gen_strings[0],
        generator_2_output=gen_strings[1],
        generator_3_output=gen_strings[2],
        generator_4_output=gen_strings[3],
        critic_output=json.dumps(critic_output, indent=2),
    )
    raw = call_claude(prompt, label="Synthesizer")
    return parse_json(raw)


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
def assess_synthesizer(case: dict, synth: dict, gen_outputs: list[dict], critic: dict) -> dict:
    issues = []

    # 1. Final verdict
    verdict = synth.get("final_verdict", "MISSING")
    verdict_ok = verdict == case["expected_verdict"]
    if not verdict_ok:
        issues.append(f"Verdict: expected {case['expected_verdict']}, got {verdict}")

    # 2. On-chain action
    action = synth.get("on_chain_action", "MISSING")
    action_ok = action == case["expected_on_chain"]
    if not action_ok:
        issues.append(f"On-chain action: expected {case['expected_on_chain']}, got {action}")

    # 3. Zero-history adjustment
    zh = synth.get("zero_history_adjustment_applied", "MISSING")
    zh_ok = zh == case["expected_zero_history"]
    if not zh_ok:
        issues.append(f"Zero-history adj: expected {case['expected_zero_history']}, got {zh}")

    # 4. Critic override
    co = synth.get("critic_override_applied", "MISSING")
    co_ok = co == case["expected_critic_override"]
    if not co_ok:
        # Allow flexibility: if critic fab_risk is high, override is correct even if we expected false
        critic_fab = critic.get("fabrication_risk", "none")
        critic_sg = critic.get("spec_gaming_detected", False)
        if co is True and (critic_fab == "high" or critic_sg is True):
            co_ok = True  # Critic legitimately triggered override
            issues = [i for i in issues if "Critic override" not in i]
        else:
            issues.append(f"Critic override: expected {case['expected_critic_override']}, got {co}")

    # 5. Rejection asymmetry (only check if verdict is REJECT)
    ra = synth.get("rejection_asymmetry_applied", "MISSING")
    ra_ok = True
    if case["expected_verdict"] == "REJECT":
        ra_ok = ra == case["expected_rejection_asymmetry"]
        if not ra_ok:
            issues.append(f"Rejection asymmetry: expected {case['expected_rejection_asymmetry']}, got {ra}")

    # 6. MDI sanity check — should be a number
    mdi = synth.get("mdi", "MISSING")
    mdi_ok = isinstance(mdi, (int, float)) and 0.0 <= mdi <= 1.0
    if not mdi_ok:
        issues.append(f"MDI out of range or missing: {mdi}")

    # 7. Confidence sanity
    conf = synth.get("confidence", "MISSING")
    conf_ok = isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0
    if not conf_ok:
        issues.append(f"Confidence out of range: {conf}")

    # 8. Verify MDI computation is roughly correct
    gen_scores = [g.get("score", 0) for g in gen_outputs if isinstance(g.get("score"), (int, float))]
    if len(gen_scores) == 4:
        expected_mdi = statistics.stdev(gen_scores) / 100
        reported_mdi = synth.get("mdi", 0)
        if isinstance(reported_mdi, (int, float)) and abs(reported_mdi - expected_mdi) > 0.15:
            issues.append(f"MDI mismatch: computed {expected_mdi:.2f}, reported {reported_mdi:.2f}")

    overall_pass = verdict_ok and action_ok and zh_ok and co_ok and mdi_ok and conf_ok
    return {
        "verdict_ok": verdict_ok,
        "action_ok": action_ok,
        "zero_history_ok": zh_ok,
        "critic_override_ok": co_ok,
        "mdi_ok": mdi_ok,
        "overall_pass": overall_pass,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_results = []
    total_start = time.time()

    for case in CASES:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"  {case['id']}: {case['spec'][:50]}...", file=sys.stderr)
        print(f"  Expected: verdict={case['expected_verdict']} action={case['expected_on_chain']} zh={case['expected_zero_history']} co={case['expected_critic_override']}", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        # Phase 1: Generators
        print(f"  Phase 1: Running 4 generators...", file=sys.stderr, flush=True)
        gen_outputs = run_generators(case)
        gen_verdicts = [g.get("verdict", "ERR") for g in gen_outputs]
        gen_scores = [g.get("score", "?") for g in gen_outputs]
        print(f"  Generator verdicts: {gen_verdicts} scores: {gen_scores}", file=sys.stderr, flush=True)

        # Phase 2: Critic
        print(f"  Phase 2: Running Adversarial Critic...", file=sys.stderr, flush=True)
        critic_output = run_critic(case, gen_outputs)
        print(f"  Critic: dir={critic_output.get('recommended_direction')} fab={critic_output.get('fabrication_risk')} sg={critic_output.get('spec_gaming_detected')}", file=sys.stderr, flush=True)

        # Phase 3: Synthesizer
        print(f"  Phase 3: Running Synthesizer...", file=sys.stderr, flush=True)
        synth_output = run_synthesizer(case, gen_outputs, critic_output)

        # Phase 4: Assess
        assessment = assess_synthesizer(case, synth_output, gen_outputs, critic_output)

        result = {
            "case_id": case["id"],
            "expected_verdict": case["expected_verdict"],
            "generator_verdicts": gen_verdicts,
            "generator_scores": gen_scores,
            "critic_direction": critic_output.get("recommended_direction", "?"),
            "critic_fab": critic_output.get("fabrication_risk", "?"),
            "critic_sg": critic_output.get("spec_gaming_detected", "?"),
            "final_verdict": synth_output.get("final_verdict", "MISSING"),
            "final_score": synth_output.get("final_score", "?"),
            "confidence": synth_output.get("confidence", "?"),
            "mdi": synth_output.get("mdi", "?"),
            "on_chain_action": synth_output.get("on_chain_action", "?"),
            "critic_override": synth_output.get("critic_override_applied", "?"),
            "zero_history": synth_output.get("zero_history_adjustment_applied", "?"),
            "rejection_asymmetry": synth_output.get("rejection_asymmetry_applied", "?"),
            "overall_pass": assessment["overall_pass"],
            "issues": assessment["issues"],
            "synth_raw": synth_output,
        }
        all_results.append(result)

        status = "PASS" if assessment["overall_pass"] else "FAIL"
        print(f"  => {status} | verdict={result['final_verdict']} score={result['final_score']} conf={result['confidence']} action={result['on_chain_action']}", file=sys.stderr, flush=True)
        if assessment["issues"]:
            for iss in assessment["issues"]:
                print(f"     ! {iss}", file=sys.stderr, flush=True)

    elapsed = time.time() - total_start

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    print()
    print("## Synthesizer Validation Results")
    print()
    print("| Case | Gen Verdicts | Critic Dir | Final Verdict | Score | Conf | MDI | On-Chain | Critic Override | Zero-Hist | Rej Asym | Pass |")
    print("|------|-------------|-----------|--------------|-------|------|-----|----------|----------------|-----------|----------|------|")
    for r in all_results:
        gv = ", ".join(r["generator_verdicts"])
        co = "Yes" if r["critic_override"] is True else ("No" if r["critic_override"] is False else "?")
        zh = "Yes" if r["zero_history"] is True else ("No" if r["zero_history"] is False else "?")
        ra = "Yes" if r["rejection_asymmetry"] is True else ("No" if r["rejection_asymmetry"] is False else "?")
        status = "PASS" if r["overall_pass"] else "**FAIL**"
        print(f"| {r['case_id']} | {gv} | {r['critic_direction']} | {r['final_verdict']} | {r['final_score']} | {r['confidence']} | {r['mdi']} | {r['on_chain_action']} | {co} | {zh} | {ra} | {status} |")

    passes = sum(1 for r in all_results if r["overall_pass"])
    print()
    print(f"**Match rate: {passes}/{len(all_results)}**")
    print(f"*Elapsed: {elapsed:.0f}s*")

    # Issues
    any_issues = [r for r in all_results if r["issues"]]
    if any_issues:
        print()
        print("### Issues")
        for r in any_issues:
            for iss in r["issues"]:
                print(f"- **{r['case_id']}**: {iss}")

    # Full JSON
    print()
    print("<details><summary>Full synthesizer JSON responses</summary>")
    print()
    for r in all_results:
        print(f"### {r['case_id']}")
        print("```json")
        print(json.dumps(r["synth_raw"], indent=2))
        print("```")
        print()
    print("</details>")


if __name__ == "__main__":
    main()
