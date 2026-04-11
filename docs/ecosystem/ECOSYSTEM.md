# AHM Ecosystem Integration — agntcy / Internet of Agents

## Position in the IoA Stack

AHM occupies the **trust evaluation layer** in the agntcy Internet of Agents (IoA) architecture:

```
Discovery (OASF + Agent Directory)
        ↓
  AHM Trust Scoring          ← we are here
        ↓
  Compose / Deploy (SLIM + Agent Workflows)
```

Agents are discovered via the OASF-powered Agent Directory Service (ADS). AHM evaluates their trustworthiness using metadata, behavioural telemetry, and provenance signals, producing D2/D3 trust scores. Those scores inform downstream composition and deployment decisions — which agents to include in multi-agent workflows, and under what trust constraints.

AHM also publishes trust scores as **W3C Verifiable Credentials** using the agntcy Identity Service, making scores portable and cryptographically verifiable by any relying party.

## Artifacts

| File | Description |
|------|-------------|
| [oasf-record.json](oasf-record.json) | OASF agent record for listing AHM in the agntcy Agent Directory |
| [trust-score-vc.json](trust-score-vc.json) | Sample Trust Score Verifiable Credential (W3C VCDM 2.0) |
| [agent-card.json](agent-card.json) | A2A Agent Card served at `/.well-known/agent.json` |

## OASF Metadata as Scoring Input

OASF records are rich, structured, and machine-parseable — a strong D2/D3 input signal. Fields like `authors`, Sigstore signatures, CIDs, `skills` taxonomy alignment, `locators`, and evaluation module metrics all map to AHM scoring dimensions (reliability, safety, compliance, performance, provenance).

**Caveat:** OASF metadata is **self-reported** by agent publishers. AHM treats it as one input signal among many — cross-referenced with runtime behavioural telemetry, incident history, and independent benchmarks. Self-reported claims are never trusted at face value.

## Contribution Roadmap

### Phase 1 — Schema Contribution (low friction)
- Open an OASF Discussion proposing trust-scoring skill taxonomy entries
- Submit a PR to `agntcy/oasf` adding `trust_evaluation` skills to the skills catalog
- Submit a PR adding an AHM evaluation module schema to `schema/modules/`

### Phase 2 — Identity Integration (medium effort)
- Open an issue on `agntcy/identity-spec` proposing a `TrustScoreBadge` VC type
- Contribute a JSON-LD context definition for trust score credentials
- Implement VC issuance using the `agntcy/identity-service` SDK

### Phase 3 — Directory Registration (low friction, after Phase 1)
- Register AHM as an agent in the hosted Agent Directory
- Publish the AHM OASF record via the CLI (`agntcy dir publish`)

### Phase 4 — Working Group / Standards Track (high effort, high impact)
- Propose an "Agent Trust & Evaluation" working group in `agntcy/governance`
- Contribute trust-scoring requirements to the IETF draft (`draft-mp-agntcy-ads-00`)
- Position AHM as a reference implementation for agent trust evaluation in the IoA

## Next Actions

- [ ] Publish `/.well-known/agent.json` to production (unblocked on this PR merging)
- [ ] POST to [prassanna-ravishankar/a2a-registry](https://github.com/prassanna-ravishankar/a2a-registry) — submit the live URL, auto-fetches card
- [ ] Submit to [a2a.ac](https://a2a.ac/)
- [ ] Submit to [A2ABaseAI/A2ARegistry](https://github.com/A2ABaseAI/A2ARegistry)
- [ ] Validate skill hierarchy names against the OASF skills catalog at [schema.oasf.agntcy.org](https://schema.oasf.agntcy.org/)
- [ ] Propose `TrustScoreBadge` VC type to [agntcy/identity-spec](https://github.com/agntcy/identity-spec)
- [ ] Register AHM in the agntcy Agent Directory via `agntcy dir publish`
- [ ] Open OASF Discussion for trust-scoring skill taxonomy proposal

## References

- [AGNTCY Documentation](https://docs.agntcy.org/)
- [OASF Schema](https://schema.oasf.agntcy.org/)
- [OASF Contribution Guide](https://docs.agntcy.org/oasf/contributing/)
- [Agent Directory Trust Model](https://docs.agntcy.org/dir/trust-model/)
- [Identity Service — Verifiable Credentials](https://docs.agntcy.org/identity/credentials/)
- [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)
- [IETF Draft: Agent Directory Service](https://datatracker.ietf.org/doc/draft-mp-agntcy-ads/)
