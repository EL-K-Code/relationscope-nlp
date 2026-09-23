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


output "operations_bucket" {
  value       = google_storage_bucket.operations.name
  description = "Private bucket for v2 telemetry, feedback and batch job state."
}

output "batch_task_queue" {
  value       = google_cloud_tasks_queue.batch.name
  description = "Cloud Tasks queue used by v2 asynchronous batch inference."
}
