import type {
  BatchInput,
  ExplainResponse,
  ModelInfo,
  OpenSetPredictResponse,
  OpenSetProfile,
  PredictResponse,
} from "@/lib/types";

const API_BASE = "/api/relationscope";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function getModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/v1/model");
}

export async function predictRelation(
  text: string,
  head: BatchInput["head"],
  tail: BatchInput["tail"],
  topK = 5,
): Promise<PredictResponse> {
  return request<PredictResponse>("/v1/relations:predict", {
    method: "POST",
    body: JSON.stringify({ text, head, tail, top_k: topK, profile }),
  });
}

export async function predictOpenSet(
  text: string,
  head: BatchInput["head"],
  tail: BatchInput["tail"],
  topK = 5,
  profile: OpenSetProfile = "balanced",
): Promise<OpenSetPredictResponse> {
  return request<OpenSetPredictResponse>("/v1/relations:predict-open-set", {
    method: "POST",
    body: JSON.stringify({ text, head, tail, top_k: topK }),
  });
}

export async function explainRelation(
  text: string,
  head: BatchInput["head"],
  tail: BatchInput["tail"],
  topEvidence = 12,
): Promise<ExplainResponse> {
  return request<ExplainResponse>("/v1/relations:explain", {
    method: "POST",
    body: JSON.stringify({
      text,
      head,
      tail,
      top_evidence: topEvidence,
    }),
  });
}

export async function predictBatch(
  examples: BatchInput[],
  topK = 1,
): Promise<PredictResponse[]> {
  const payload = await request<{
    items: Omit<PredictResponse, "latency_ms" | "model_id">[];
    latency_ms: number;
    model_id: string;
    batch_size: number;
  }>("/v1/relations:predict-batch", {
    method: "POST",
    body: JSON.stringify({
      examples: examples.map(({ text, head, tail }) => ({ text, head, tail })),
      top_k: topK,
    }),
  });

  return payload.items.map((item) => ({
    ...item,
    latency_ms: payload.latency_ms / Math.max(payload.batch_size, 1),
    model_id: payload.model_id,
  }));
}
