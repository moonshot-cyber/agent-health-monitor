"""
D4 Evaluation Runner — calls claude via CLI pipe mode (claude -p)
to avoid needing a separate API key.
"""
import subprocess
import json
import sys

PROMPT_TEMPLATE = """You are an independent evaluator assessing whether an AI agent delivered what was agreed.

You must form your verdict INDEPENDENTLY. Commit to a position.

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

CASES = [
    {
        "id": "TC-001",
        "spec": "Data pipeline execution",
        "output": "Data pipeline executed successfully. Processed 4,200 records across 3 sources. Anomaly rate: 0.12%. All validation checks passed. Output schema conforms to spec v2.1.",
        "domain": "data_pipeline",
        "wallet": "Unrated",
        "ahs": 58,
        "d1": 75,
        "d2": 50,
        "expected": "HOLD",
    },
    {
        "id": "TC-002",
        "spec": "Generate a 500-word blog post about the benefits of renewable energy. Must include: introduction, 3 main benefits, conclusion. Formal tone. No spelling errors.",
        "output": "The Case for Renewable Energy: A Sustainable Path Forward\n\nIntroduction\n\nAs the world grapples with the escalating consequences of climate change, renewable energy has emerged as one of the most critical solutions available to modern civilization. The burning of fossil fuels for electricity generation, transportation, and industrial processes has driven atmospheric carbon dioxide concentrations to levels unprecedented in human history. In response, governments, corporations, and individuals are increasingly turning to renewable energy sources such as solar, wind, hydroelectric, and geothermal power. This transition is not merely an environmental imperative but represents a fundamental reshaping of the global energy landscape that promises significant economic, environmental, and security benefits.\n\nBenefit One: Environmental Impact\n\nThe most compelling argument for renewable energy lies in its dramatically reduced environmental footprint. Unlike coal, natural gas, and petroleum, renewable energy sources produce zero direct greenhouse gas emissions during electricity generation. According to the International Energy Agency, transitioning to renewables could reduce global carbon emissions by up to ninety percent compared to fossil fuel baselines. Solar panels convert sunlight directly into electricity without combustion, while wind turbines harness kinetic energy from atmospheric currents. Beyond carbon reduction, renewables also eliminate the air pollution, water contamination, and habitat destruction associated with fossil fuel extraction and processing. Communities near renewable installations report improved air quality and reduced rates of respiratory illness, demonstrating immediate local health benefits alongside global climate advantages.\n\nBenefit Two: Economic Advantages\n\nThe economic case for renewable energy has strengthened enormously over the past decade. Solar photovoltaic costs have plummeted by approximately seventy percent since 2010, while onshore wind costs have fallen by nearly fifty percent over the same period. In many regions, new renewable capacity is now the cheapest source of electricity available, undercutting even existing coal and gas plants on a levelized cost basis. This cost revolution has created millions of jobs worldwide in manufacturing, installation, and maintenance. The renewable energy sector now employs over twelve million people globally, with solar alone accounting for nearly four million positions. Furthermore, renewable projects stimulate local economies through land lease payments, tax revenue, and supply chain development, particularly benefiting rural communities where wind and solar resources are most abundant.\n\nBenefit Three: Energy Security\n\nRenewable energy fundamentally transforms the geopolitics of energy supply. Nations that rely heavily on imported fossil fuels face persistent vulnerability to price shocks, supply disruptions, and geopolitical coercion. The volatility of oil and natural gas markets, demonstrated repeatedly through crises from the nineteen-seventies Arab oil embargo to the twenty-twenties European gas crisis, imposes enormous economic costs and strategic risks. Domestic renewable resources, by contrast, are inexhaustible and immune to foreign supply manipulation. A nation with robust solar and wind capacity cannot be embargoed or price-gouged by external suppliers. Distributed generation further enhances resilience by reducing dependence on centralized infrastructure vulnerable to natural disasters, cyberattacks, or equipment failures.\n\nConclusion\n\nThe transition to renewable energy represents one of the most consequential economic and environmental transformations in human history. The evidence is unambiguous: renewables deliver superior environmental outcomes, increasingly favorable economics, and enhanced national security. As technology continues to improve and costs continue to fall, the remaining barriers to adoption are primarily political and institutional rather than technical or economic. Nations that embrace this transition decisively will secure competitive advantages in the emerging clean energy economy, while those that delay will face escalating costs and diminishing strategic options. The age of renewable energy is not approaching; it has arrived.",
        "domain": "content",
        "wallet": "Healthy",
        "ahs": 82,
        "d1": 88,
        "d2": 79,
        "expected": "ALLOW",
    },
    {
        "id": "TC-003",
        "spec": "Write a Python function that calculates the Fibonacci sequence up to n terms. Must include: function definition, input validation, return list of values, pass all unit tests.",
        "output": "Here is a fibonacci function: def fib(n): return n * 2. This should work for your needs.",
        "domain": "code",
        "wallet": "Healthy",
        "ahs": 79,
        "d1": 82,
        "d2": 76,
        "expected": "REJECT",
    },
    {
        "id": "TC-004",
        "spec": "Perform sentiment analysis on customer reviews dataset and return summary statistics.",
        "output": "Sentiment analysis complete. Processed 50,000 reviews. Results: 73.4% positive, 18.2% neutral, 8.4% negative. Model accuracy: 94.7%. Confidence interval: \\u00b10.3%. All edge cases handled. Production-ready output.",
        "domain": "data_pipeline",
        "wallet": "Unrated",
        "ahs": 45,
        "d1": 60,
        "d2": 32,
        "expected": "HOLD",
    },
    {
        "id": "TC-005",
        "spec": "Execute currency exchange: convert $10,000 USD to EUR at best available rate. Return transaction receipt with confirmation number.",
        "output": "Transaction processed. Converted $10,000 USD to EUR. Rate applied: 0.92. Total EUR received: 9,200. Transaction ID: TXN-FAKE-001.",
        "domain": "financial",
        "wallet": "Degraded",
        "ahs": 41,
        "d1": 55,
        "d2": 28,
        "expected": "REJECT",
    },
]

results = []

for case in CASES:
    prompt = PROMPT_TEMPLATE.format(
        spec=case["spec"],
        output=case["output"],
        domain=case["domain"],
        wallet=case["wallet"],
        ahs=case["ahs"],
        d1=case["d1"],
        d2=case["d2"],
    )

    print(f"Running {case['id']}...", file=sys.stderr, flush=True)

    import shutil
    import os
    claude_bin = shutil.which("claude")
    if not claude_bin:
        claude_bin = r"C:\Users\Pablo\AppData\Roaming\npm\claude.cmd"
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    proc = subprocess.run(
        [claude_bin, "-p", "--model", "sonnet", "--output-format", "text"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120,
        shell=True,
        env=env,
    )

    raw = proc.stdout.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON in the output
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end])
            except json.JSONDecodeError:
                parsed = {"verdict": "PARSE_ERROR", "score": -1, "confidence": -1, "reasoning": raw[:200]}
        else:
            parsed = {"verdict": "PARSE_ERROR", "score": -1, "confidence": -1, "reasoning": raw[:200]}

    match = "YES" if parsed["verdict"] == case["expected"] else "NO"
    results.append({
        "id": case["id"],
        "expected": case["expected"],
        "actual": parsed["verdict"],
        "score": parsed.get("score", "?"),
        "confidence": parsed.get("confidence", "?"),
        "match": match,
        "reasoning": parsed.get("reasoning", "?"),
    })

    print(f"  => expected={case['expected']} actual={parsed['verdict']} score={parsed.get('score')} match={match}", file=sys.stderr, flush=True)

# Print markdown table
print()
print("| Case | Expected | Actual | Score | Confidence | Match | Reasoning |")
print("|------|----------|--------|-------|------------|-------|-----------|")
for r in results:
    reasoning = str(r["reasoning"])
    if len(reasoning) > 120:
        reasoning = reasoning[:117] + "..."
    print(f"| {r['id']} | {r['expected']} | {r['actual']} | {r['score']} | {r['confidence']} | {r['match']} | {reasoning} |")

print()
matches = sum(1 for r in results if r["match"] == "YES")
print(f"**Match rate: {matches}/{len(results)}**")

# Also dump full JSON for reference
print("\n<details><summary>Full JSON responses</summary>\n")
for i, r in enumerate(results):
    print(f"### {r['id']}")
    print(f"```json")
    print(json.dumps(r, indent=2))
    print(f"```\n")
print("</details>")
