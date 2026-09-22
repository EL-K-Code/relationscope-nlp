output "web_url" {
  value       = google_cloud_run_v2_service.web.uri
  description = "Public RelationScope frontend URL."
}

output "api_url" {
  value       = google_cloud_run_v2_service.api.uri
  description = "Private RelationScope API URL."
}

output "model_bucket" {
  value = google_storage_bucket.models.name
}

output "artifact_repository" {
  value = google_artifact_registry_repository.containers.name
}
