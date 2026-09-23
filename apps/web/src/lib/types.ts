export type EntitySpan = {
  text: string;
  start: number;
  end: number;
};

export type OpenSetProfile = "high_coverage" | "balanced" | "high_safety";

export type Prediction = {
  relation_id: string;
  relation_name: string;
  confidence: number;
};

export type PredictResponse = {
  head: string;
  tail: string;
  marked_text: string;
  predictions: Prediction[];
  latency_ms: number;
  model_id: string;
};

export type OpenSetPredictResponse = PredictResponse & {
  decision: "KNOWN_RELATION" | "UNKNOWN_RELATION";
  accepted: boolean;
  profile: OpenSetProfile;
  threshold: number;
  temperature: number;
};

export type ExplanationToken = {
  token: string;
  position: number;
  confidence_drop: number;
};

export type ExplainResponse = {
  model_id: string;
  relation: Prediction;
  baseline_confidence: number;
  evidence: ExplanationToken[];
  method: "token_occlusion";
  latency_ms: number;
};

export type ModelInfo = {
  model_id: string;
  device: string;
  num_relations: number;
  max_length: number;
  pooling_mode: string;
  input_view: string;
  open_set_enabled: boolean;
  default_open_set_profile: OpenSetProfile | null;
  open_set_profiles: OpenSetProfile[];
};

export type BatchInput = {
  id: string;
  text: string;
  head: EntitySpan;
  tail: EntitySpan;
};

export type BatchResult = BatchInput & {
  relation?: Prediction;
  error?: string;
};


export type WordInfluence = {
  word: string;
  start: number;
  end: number;
  confidence_drop: number;
  normalized_influence: number;
  direction: "supports" | "opposes" | "neutral";
};

export type InfluenceExplainResponse = {
  model_id: string;
  relation: Prediction;
  decision: "KNOWN_RELATION" | "UNKNOWN_RELATION";
  profile: OpenSetProfile;
  threshold: number;
  baseline_confidence: number;
  marked_text: string;
  influences: WordInfluence[];
  method: "word_occlusion";
  latency_ms: number;
};

export type FeedbackVerdict = "confirm" | "correct" | "unknown" | "skip";

export type FeedbackResponse = {
  feedback_id: string;
  stored_text: boolean;
  created_at: string;
};

export type BatchJobSummary = {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  created_at: string;
  updated_at: string;
  total: number;
  processed: number;
  model_id: string | null;
  error: string | null;
};

export type BatchJobItem = {
  head: string;
  tail: string;
  relation: Prediction;
  decision: "KNOWN_RELATION" | "UNKNOWN_RELATION";
  accepted: boolean;
  profile: OpenSetProfile;
  threshold: number;
};

export type BatchJobResponse = BatchJobSummary & {
  results: BatchJobItem[] | null;
};

export type MetricsSummary = {
  window_events: number;
  total_predictions: number;
  known_rate: number;
  unknown_rate: number;
  mean_confidence: number;
  mean_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  profiles: { profile: string; count: number }[];
  models: { model_id: string; count: number }[];
};

export type ModelRegistryEntry = {
  model_version: string;
  gcs_prefix: string;
  promoted: boolean;
  created_at: string | null;
};

export type ModelRegistry = {
  current_model_id: string;
  entries: ModelRegistryEntry[];
};


export type RelationCatalogItem = {
  relation_id: string;
  relation_name: string;
};

export type RelationCatalog = {
  items: RelationCatalogItem[];
};
