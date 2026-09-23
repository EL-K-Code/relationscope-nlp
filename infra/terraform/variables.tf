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
