variable "project_id" {
  type        = string
  description = "Google Cloud project ID."
}

variable "region" {
  type        = string
  description = "Cloud Run / Artifact Registry region."
  default     = "europe-west1"
}

variable "artifact_repository" {
  type    = string
  default = "relationscope"
}

variable "cloud_build_submitter_member" {
  type        = string
  description = "Optional IAM member allowed to submit Cloud Build jobs, e.g. user:name@example.com."
  default     = null
  nullable    = true

  validation {
    condition = (
      var.cloud_build_submitter_member == null ||
      can(regex("^(user|serviceAccount):[^[:space:]]+$", var.cloud_build_submitter_member))
    )
    error_message = "cloud_build_submitter_member must be null or an IAM member such as user:name@example.com."
  }
}

variable "model_bucket_name" {
  type        = string
  description = "Globally unique GCS bucket for versioned model bundles."
}

variable "model_prefix" {
  type        = string
  description = "GCS prefix containing the promoted model bundle."
  default     = "models/current"
}

variable "api_image" {
  type        = string
  description = "Full Artifact Registry image reference for the FastAPI container."
}

variable "web_image" {
  type        = string
  description = "Full Artifact Registry image reference for the Next.js container."
}

variable "api_service_name" {
  type    = string
  default = "relationscope-api"
}

variable "web_service_name" {
  type    = string
  default = "relationscope-web"
}

variable "api_cpu" {
  type    = string
  default = "2"
}

variable "api_memory" {
  type    = string
  default = "4Gi"
}

variable "web_cpu" {
  type    = string
  default = "1"
}

variable "web_memory" {
  type    = string
  default = "512Mi"
}

variable "max_api_instances" {
  type    = number
  default = 3
}

variable "max_web_instances" {
  type    = number
  default = 3
}


variable "ops_bucket_name" {
  type        = string
  description = "Optional globally unique GCS bucket for v2 telemetry, feedback and batch job state. Defaults to <project_id>-relationscope-ops."
  default     = null
  nullable    = true
}

variable "ops_retention_days" {
  type        = number
  description = "Lifecycle retention for v2 operational JSON objects."
  default     = 90
}

variable "task_queue_name" {
  type        = string
  description = "Cloud Tasks queue used for asynchronous batch inference."
  default     = "relationscope-batch"
}

variable "task_max_concurrent_dispatches" {
  type        = number
  description = "Maximum number of batch worker callbacks dispatched concurrently."
  default     = 1
}

variable "task_max_dispatches_per_second" {
  type        = number
  description = "Maximum Cloud Tasks dispatch rate for batch inference."
  default     = 1
}


variable "ner_model" {
  type        = string
  description = "Optional Hugging Face/local token-classification model used by V3 document intelligence. Empty uses the deterministic heuristic fallback."
  default     = ""
}

variable "enable_challenger" {
  type        = bool
  description = "Provision the optional private challenger API used for canary evaluation."
  default     = false
}

variable "challenger_service_name" {
  type        = string
  description = "Cloud Run service name for the optional V3 challenger."
  default     = "relationscope-api-challenger"
}

variable "challenger_api_image" {
  type        = string
  description = "Optional immutable challenger API image. Defaults to api_image when null."
  default     = null
  nullable    = true
}

variable "challenger_model_prefix" {
  type        = string
  description = "Immutable model-bucket prefix loaded by the challenger service."
  default     = "models/releases/modernbert-open-set-v1.1"
}

variable "canary_percent" {
  type        = number
  description = "Percentage of eligible public inference requests routed to the challenger."
  default     = 0

  validation {
    condition     = var.canary_percent >= 0 && var.canary_percent <= 100
    error_message = "canary_percent must be between 0 and 100."
  }
}
