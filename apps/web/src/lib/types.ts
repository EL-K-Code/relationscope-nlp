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
