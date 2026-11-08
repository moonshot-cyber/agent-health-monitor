import { Buffer } from "node:buffer";
import { Type, type Static } from "typebox";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

const BASE_URL = "https://agenthealthmonitor.xyz";
const USER_AGENT = "AHM-OpenClaw-Plugin/1.0.0";
const DEFAULT_TIMEOUT_MS = 30_000;
const HEAVY_TIMEOUT_MS = 60_000;
const ADDRESS_RE = /^0x[0-9a-fA-F]{40}$/;

type ErrorCode = "INVALID_ADDRESS" | "PAYMENT_REQUIRED" | "TIMEOUT" | "RATE_LIMITED" | "SERVER_ERROR";

type ToolSuccess = { success: true; endpoint: string; price: string; data: unknown; note?: string };
type ToolError = { success: false; error: ErrorCode; message: string; details: Record<string, unknown> };
type ToolOutcome = ToolSuccess | ToolError;

type EndpointSpec = {
  tool: string;
  label: string;
  description: string;
  method: "GET" | "POST";
  /** Paid path; `{address}` is substituted. */
  path: string;
  /** Slug for the path-based coupon route `/coupon/{slug}/{CODE}/{address}`; null when none exists. */
  couponSlug: string | null;
  price: string;
  timeoutMs: number;
  acceptsAgentUrl?: boolean;
};

const ENDPOINTS: EndpointSpec[] = [
  {
    tool: "ahm_risk_check",
    label: "AHM Risk Check",
    description:
      "Quick 0-100 risk score (LOW/MEDIUM/HIGH/CRITICAL) and one-line verdict for an EVM wallet. Cheapest pre-flight check before paying or trusting a counterparty. Costs $0.001 USDC on Base via x402.",
    method: "GET",
    path: "/risk/{address}",
    couponSlug: "risk",
    price: "$0.001",
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    tool: "ahm_premium_risk",
    label: "AHM Premium Risk",
    description:
      "Risk score plus Nansen labels, PnL summary and operational health (tx failure rate, nonce gaps) for an EVM wallet. Costs $0.05 USDC on Base via x402.",
    method: "GET",
    path: "/risk/premium/{address}",
    couponSlug: "risk-premium",
    price: "$0.05",
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    tool: "ahm_health",
    label: "AHM Wallet Health",
    description:
      "Full wallet health diagnosis: success rate, gas efficiency, nonce health, wasted gas and recommendations. Costs $0.50 USDC on Base via x402.",
    method: "GET",
    path: "/health/{address}",
    couponSlug: "health",
    price: "$0.50",
    timeoutMs: HEAVY_TIMEOUT_MS,
  },
  {
    tool: "ahm_wash",
    label: "AHM Wallet Wash",
    description:
      "Wallet hygiene scan: dust, spam tokens, gas efficiency, failure patterns and nonce gaps, with a 0-100 cleanliness score (Spotless/Clean/Needs Attention/Dirty/Critical). Read-only. Costs $0.50 USDC on Base via x402.",
    method: "POST",
    path: "/wash/{address}",
    couponSlug: "wash",
    price: "$0.50",
    timeoutMs: HEAVY_TIMEOUT_MS,
  },
  {
    tool: "ahm_health_score",
    label: "AHM Agent Health Score",
    description:
      "Agent Health Score (AHS): composite 0-100 with A-F grade, per-dimension breakdown (D1 Wallet Hygiene, D2 Behavioural, optional D3 Infrastructure), cross-dimensional patterns (e.g. Zombie Agent, Gas War Casualty) and a trust routing recommendation. Pass agent_url to enable D3 infrastructure probing. Costs $1.00 USDC on Base via x402.",
    method: "GET",
    path: "/ahs/{address}",
    couponSlug: "ahs",
    price: "$1.00",
    timeoutMs: HEAVY_TIMEOUT_MS,
    acceptsAgentUrl: true,
  },
  {
    tool: "ahm_trust_route",
    label: "AHM Trust Route",
    description:
      "Trust routing decision (instant_settle / escrow / reject) for an agent-to-agent payment, from the wallet's latest cached AHS. Fails if the wallet has never been scored: run ahm_health_score first. No coupon route exists for this endpoint. Costs $0.01 USDC on Base via x402.",
    method: "GET",
    path: "/ahs/route/{address}",
    couponSlug: null,
    price: "$0.01",
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    tool: "ahm_counterparties",
    label: "AHM Counterparties",
    description:
      "Know Your Counterparty: the wallet's top counterparties with labels, interaction counts and volume. Costs $0.10 USDC on Base via x402.",
    method: "GET",
    path: "/counterparties/{address}",
    couponSlug: "counterparties",
    price: "$0.10",
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
  {
    tool: "ahm_network_map",
    label: "AHM Network Map",
    description:
      "Related wallets (funders, deployers, multisig relations) for funding-source investigation. Costs $0.10 USDC on Base via x402.",
    method: "GET",
    path: "/network-map/{address}",
    couponSlug: "network-map",
    price: "$0.10",
    timeoutMs: DEFAULT_TIMEOUT_MS,
  },
];

const AddressParam = Type.String({
  description: "EVM wallet address: 0x followed by 40 hex characters.",
});
const CouponParam = Type.Optional(
  Type.String({ description: "Optional AHM coupon code for free test access instead of an x402 payment." }),
);

const BaseParams = Type.Object({ address: AddressParam, coupon: CouponParam });
const AhsParams = Type.Object({
  address: AddressParam,
  coupon: CouponParam,
  agent_url: Type.Optional(
    Type.String({
      description: "Optional public http(s) URL of the agent's endpoint. Enables D3 Infrastructure Health (3D mode).",
    }),
  ),
});
type ToolParams = Static<typeof AhsParams>;

function fail(error: ErrorCode, message: string, details: Record<string, unknown> = {}): ToolError {
  return { success: false, error, message, details };
}

function buildUrl(spec: EndpointSpec, address: string, coupon: string | undefined, agentUrl: string | undefined) {
  const useCoupon = Boolean(coupon) && spec.couponSlug !== null;
  const path = useCoupon
    ? `/coupon/${spec.couponSlug}/${encodeURIComponent(coupon!.trim())}/${address}`
    : spec.path.replace("{address}", address);
  const url = new URL(path, BASE_URL);
  if (agentUrl && spec.acceptsAgentUrl) url.searchParams.set("agent_url", agentUrl);
  // Coupon routes are all GET, including the coupon mirror of POST /wash.
  return { url: url.toString(), method: useCoupon ? "GET" : spec.method, useCoupon };
}

/** x402 v2 sends requirements base64-encoded in the `payment-required` header; v1 sent them in the body. */
function decodePaymentRequirements(res: Response, bodyText: string): unknown {
  const header = res.headers.get("payment-required");
  if (header) {
    try {
      return JSON.parse(Buffer.from(header, "base64").toString("utf8"));
    } catch {
      return { raw: header };
    }
  }
  try {
    const body = JSON.parse(bodyText);
    if (body && typeof body === "object" && Object.keys(body).length > 0) return body;
  } catch {
    // fall through
  }
  return null;
}

function parseBody(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function errorDetail(body: unknown): string | undefined {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  return typeof body === "string" && body ? body.slice(0, 500) : undefined;
}

async function callEndpoint(spec: EndpointSpec, params: ToolParams, signal?: AbortSignal): Promise<ToolOutcome> {
  const address = (params.address ?? "").trim();
  if (!ADDRESS_RE.test(address)) {
    return fail("INVALID_ADDRESS", "Address must be 0x followed by 40 hexadecimal characters.", { address });
  }

  const { url, method, useCoupon } = buildUrl(spec, address, params.coupon, params.agent_url);
  const note =
    params.coupon && spec.couponSlug === null
      ? `${spec.path} has no coupon route; the coupon was ignored and the paid endpoint was called.`
      : undefined;

  const timeout = AbortSignal.timeout(spec.timeoutMs);
  const combined = signal ? AbortSignal.any([signal, timeout]) : timeout;

  let res: Response;
  try {
    res = await fetch(url, {
      method,
      headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
      signal: combined,
    });
  } catch (err) {
    if (timeout.aborted) {
      return fail("TIMEOUT", `AHM did not respond within ${spec.timeoutMs / 1000}s.`, {
        endpoint: spec.path,
        timeoutMs: spec.timeoutMs,
      });
    }
    return fail("SERVER_ERROR", "Network error contacting Agent Health Monitor.", {
      endpoint: spec.path,
      cause: err instanceof Error ? err.message : String(err),
    });
  }

  const text = await res.text();

  if (res.ok) {
    return { success: true, endpoint: `${method} ${spec.path}`, price: useCoupon ? "coupon" : spec.price, data: parseBody(text), ...(note ? { note } : {}) };
  }

  const body = parseBody(text);
  const detail = errorDetail(body);
  const base = {
    endpoint: `${method} ${spec.path}`,
    httpStatus: res.status,
    ...(detail ? { detail } : {}),
    ...(note ? { note } : {}),
  };

  switch (res.status) {
    case 402:
      return fail(
        "PAYMENT_REQUIRED",
        `This call costs ${spec.price} USDC on Base (eip155:8453) via x402. This plugin has no x402 payment client, so it cannot pay. Sign the payment requirements below with an x402 client, or retry with a coupon code.`,
        { ...base, price: spec.price, paymentRequirements: decodePaymentRequirements(res, text) },
      );
    case 403:
      return fail("PAYMENT_REQUIRED", detail ?? "Coupon rejected; payment is required.", base);
    case 400:
    case 422:
      return fail(
        "INVALID_ADDRESS",
        detail ?? "AHM rejected the request parameters (address or agent_url).",
        base,
      );
    case 429:
      return fail("RATE_LIMITED", detail ?? "Rate limited by AHM. Retry later.", {
        ...base,
        retryAfter: res.headers.get("retry-after"),
      });
    default:
      return fail("SERVER_ERROR", detail ?? `AHM returned HTTP ${res.status}.`, base);
  }
}

export default definePluginEntry({
  id: "ahm-agent-health-monitor",
  name: "Agent Health Monitor",
  description:
    "On-chain agent health diagnostics, trust scoring, and wallet hygiene analysis via x402 micropayments on Base.",
  register(api) {
    for (const spec of ENDPOINTS) {
      api.registerTool({
        name: spec.tool,
        label: spec.label,
        description: spec.description,
        parameters: spec.acceptsAgentUrl ? AhsParams : BaseParams,
        resultContentSource: "network",
        async execute(_toolCallId, params, signal) {
          const outcome = await callEndpoint(spec, params as ToolParams, signal);
          const details =
            !outcome.success && outcome.error === "TIMEOUT"
              ? { ...outcome, timedOut: true, timeoutMs: spec.timeoutMs }
              : outcome;
          return {
            content: [{ type: "text", text: JSON.stringify(outcome, null, 2) }],
            details,
          };
        },
      });
    }
  },
});
