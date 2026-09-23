locals {
  cloud_build_submitter_roles = toset([
    "roles/cloudbuild.builds.editor",
    "roles/serviceusage.serviceUsageConsumer",
  ])

  required_services = toset([
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudtasks.googleapis.com",
    "iamcredentials.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
  ])

  ops_bucket_name = coalesce(
    var.ops_bucket_name,
    "${var.project_id}-relationscope-ops",
  )
}

resource "google_project_service" "required" {
  for_each = local.required_services

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_project_iam_member" "cloud_build_submitter" {
  for_each = var.cloud_build_submitter_member == null ? toset([]) : local.cloud_build_submitter_roles

  project = var.project_id
  role    = each.value
  member  = var.cloud_build_submitter_member
}

resource "google_artifact_registry_repository" "containers" {
  location      = var.region
  repository_id = var.artifact_repository
  description   = "RelationScope container images"
  format        = "DOCKER"

  depends_on = [
    google_project_service.required["artifactregistry.googleapis.com"],
  ]
}

resource "google_storage_bucket" "models" {
  name                        = var.model_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 10
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [
    google_project_service.required["storage.googleapis.com"],
  ]
}

resource "google_storage_bucket" "operations" {
  name                        = local.ops_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  lifecycle_rule {
    condition {
      age = var.ops_retention_days
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [
    google_project_service.required["storage.googleapis.com"],
  ]
}

resource "google_service_account" "api" {
  account_id   = "relationscope-api"
  display_name = "RelationScope API"
}

resource "google_service_account" "web" {
  account_id   = "relationscope-web"
  display_name = "RelationScope Web"
}

resource "google_service_account" "batch_tasks" {
  account_id   = "relationscope-tasks"
  display_name = "RelationScope Cloud Tasks caller"
}

resource "google_storage_bucket_iam_member" "api_reads_models" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_storage_bucket_iam_member" "api_manages_operations" {
  bucket = google_storage_bucket.operations.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_cloud_tasks_queue" "batch" {
  name     = var.task_queue_name
  location = var.region

  rate_limits {
    max_concurrent_dispatches = var.task_max_concurrent_dispatches
    max_dispatches_per_second = var.task_max_dispatches_per_second
  }

  retry_config {
    max_attempts       = 3
    max_retry_duration = "3600s"
    min_backoff        = "5s"
    max_backoff        = "60s"
    max_doublings      = 3
  }

  depends_on = [
    google_project_service.required["cloudtasks.googleapis.com"],
  ]
}

resource "google_project_iam_member" "api_enqueues_batch_tasks" {
  project = var.project_id
  role    = "roles/cloudtasks.enqueuer"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_service_account_iam_member" "api_acts_as_batch_task_caller" {
  service_account_id = google_service_account.batch_tasks.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.api.email}"
}

resource "google_cloud_run_v2_service" "api" {
  name                = var.api_service_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.api.email
    timeout         = "900s"

    scaling {
      min_instance_count = 0
      max_instance_count = var.max_api_instances
    }

    containers {
      name  = "api"
      image = var.api_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.api_cpu
          memory = var.api_memory
        }
      }

      env {
        name  = "RELATIONSCOPE_MODEL_DIR"
        value = "/tmp/relationscope/model"
      }

      env {
        name  = "RELATIONSCOPE_MODEL_GCS_URI"
        value = "gs://${google_storage_bucket.models.name}/${var.model_prefix}"
      }

      env {
        name  = "RELATIONSCOPE_MODEL_RELEASES_PREFIX"
        value = "models/releases"
      }

      env {
        name  = "RELATIONSCOPE_DEVICE"
        value = "cpu"
      }

      env {
        name  = "RELATIONSCOPE_OPS_BUCKET"
        value = google_storage_bucket.operations.name
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "RELATIONSCOPE_TASK_LOCATION"
        value = var.region
      }

      env {
        name  = "RELATIONSCOPE_TASK_QUEUE"
        value = google_cloud_tasks_queue.batch.name
      }

      env {
        name  = "RELATIONSCOPE_TASK_SERVICE_ACCOUNT"
        value = google_service_account.batch_tasks.email
      }

      startup_probe {
        initial_delay_seconds = 1
        timeout_seconds       = 2
        period_seconds        = 5
        failure_threshold     = 12

        http_get {
          path = "/health"
          port = 8080
        }
      }

      liveness_probe {
        timeout_seconds = 2
        period_seconds  = 30

        http_get {
          path = "/health"
          port = 8080
        }
      }
    }
  }

  depends_on = [
    google_project_service.required["run.googleapis.com"],
    google_storage_bucket_iam_member.api_reads_models,
    google_storage_bucket_iam_member.api_manages_operations,
    google_project_iam_member.api_enqueues_batch_tasks,
    google_service_account_iam_member.api_acts_as_batch_task_caller,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "web_invokes_api" {
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.web.email}"
}

resource "google_cloud_run_v2_service_iam_member" "batch_tasks_invoke_api" {
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.batch_tasks.email}"
}

resource "google_cloud_run_v2_service" "web" {
  name                = var.web_service_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.web.email

    scaling {
      min_instance_count = 0
      max_instance_count = var.max_web_instances
    }

    containers {
      name  = "web"
      image = var.web_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.web_cpu
          memory = var.web_memory
        }
      }

      env {
        name  = "RELATIONSCOPE_API_URL"
        value = google_cloud_run_v2_service.api.uri
      }

      env {
        name  = "RELATIONSCOPE_API_AUDIENCE"
        value = google_cloud_run_v2_service.api.uri
      }

      startup_probe {
        initial_delay_seconds = 1
        timeout_seconds       = 2
        period_seconds        = 5
        failure_threshold     = 12

        http_get {
          path = "/"
          port = 8080
        }
      }
    }
  }

  depends_on = [
    google_cloud_run_v2_service_iam_member.web_invokes_api,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public_web" {
  project  = var.project_id
  location = google_cloud_run_v2_service.web.location
  name     = google_cloud_run_v2_service.web.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
