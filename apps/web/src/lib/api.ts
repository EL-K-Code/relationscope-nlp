import type {
  BatchInput,
  BatchJobResponse,
  BatchJobSummary,
  FeedbackResponse,
  InfluenceExplainResponse,
  MetricsSummary,
  ModelInfo,
  ModelRegistry,
  OpenSetPredictResponse,
  RelationCatalog,
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
  return request<ModelInfo>("/v2/model");
}

export async function predictRelation(
  text: string,
  head: BatchInput["head"],
  tail: BatchInput["tail"],
  topK = 5,
): Promise<PredictResponse> {
  return request<PredictResponse>("/v1/relations:predict", {
    method: "POST",
    body: JSON.stringify({ text, head, tail, top_k: topK }),
  });
}

export async function predictOpenSet(
  text: string,
  head: BatchInput["head"],
  tail: BatchInput["tail"],
  topK = 5,
  profile: OpenSetProfile = "balanced",
): Promise<OpenSetPredictResponse> {
  return request<OpenSetPredictResponse>("/v2/relations:predict-open-set", {
    method: "POST",
    body: JSON.stringify({ text, head, tail, top_k: topK, profile }),
  });
}

export async function explainRelation(
  text: string,
  head: BatchInput["head"],
  tail: BatchInput["tail"],
  profile: OpenSetProfile = "balanced",
  targetRelationId?: string,
): Promise<InfluenceExplainResponse> {
  return request<InfluenceExplainResponse>("/v2/relations:explain", {
    method: "POST",
    body: JSON.stringify({
      text,
      head,
      tail,
      profile,
      target_relation_id: targetRelationId ?? null,
    }),
  });
}

export async function submitFeedback(input: {
  text: string;
  head: BatchInput["head"];
  tail: BatchInput["tail"];
  prediction: {
    model_id: string;
    relation_id: string;
    relation_name: string;
    confidence: number;
    decision: "KNOWN_RELATION" | "UNKNOWN_RELATION";
    profile: OpenSetProfile;
    threshold: number;
  };
  verdict: "confirm" | "correct" | "unknown" | "skip";
  corrected_relation_id?: string | null;
  corrected_relation_name?: string | null;
  note?: string | null;
  consent_to_store_text: boolean;
}): Promise<FeedbackResponse> {
  return request<FeedbackResponse>("/v2/feedback", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function createBatchJob(
  examples: BatchInput[],
  profile: OpenSetProfile,
  topK = 1,
): Promise<BatchJobSummary> {
  return request<BatchJobSummary>("/v2/batch-jobs", {
    method: "POST",
    body: JSON.stringify({
      examples: examples.map(({ text, head, tail }) => ({ text, head, tail })),
      top_k: topK,
      profile,
    }),
  });
}

export async function getBatchJob(jobId: string): Promise<BatchJobResponse> {
  return request<BatchJobResponse>(`/v2/batch-jobs/${jobId}`);
}

export async function getMetricsSummary(limit = 500): Promise<MetricsSummary> {
  return request<MetricsSummary>(`/v2/metrics/summary?limit=${limit}`);
}

export async function getModelRegistry(): Promise<ModelRegistry> {
  return request<ModelRegistry>("/v2/models");
}


export async function getRelationCatalog(): Promise<RelationCatalog> {
  return request<RelationCatalog>("/v2/relations");
}
