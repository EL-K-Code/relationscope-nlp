"use client";

import { Activity, Network, RefreshCw, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";

import { getMetricsSummary, getModelRegistry } from "@/lib/api";
import type { MetricsSummary, ModelInfo, ModelRegistry } from "@/lib/types";

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function MonitoringPanel({ model }: { model: ModelInfo | null }) {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [registry, setRegistry] = useState<ModelRegistry | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setBusy(true);
    setError(null);
    try {
      const [summary, models] = await Promise.all([
        getMetricsSummary(500),
        getModelRegistry(),
      ]);
      setMetrics(summary);
      setRegistry(models);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Monitoring data unavailable.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  return (
    <div className="monitoringPage">
      <section className="metricGrid monitoringMetrics">
        <article className="metricTile">
          <span>Predictions</span>
          <strong>{metrics?.total_predictions ?? 0}</strong>
          <small>{metrics?.window_events ?? 0} telemetry events</small>
        </article>
        <article className="metricTile">
          <span>Known rate</span>
          <strong>{percent(metrics?.known_rate ?? 0)}</strong>
          <small>Accepted predictions</small>
        </article>
        <article className="metricTile">
          <span>Unknown rate</span>
          <strong>{percent(metrics?.unknown_rate ?? 0)}</strong>
          <small>Open-set abstentions</small>
        </article>
        <article className="metricTile">
          <span>P95 latency</span>
          <strong>{(metrics?.p95_latency_ms ?? 0).toFixed(0)} ms</strong>
          <small>Production telemetry</small>
        </article>
      </section>

      {error && <div className="notice noticeError">{error}</div>}

      <div className="monitoringGrid">
        <section className="surface telemetryCard">
          <div className="surfaceHeader">
            <div>
              <span className="sectionKicker">TELEMETRY</span>
              <h2>Production health</h2>
            </div>
            <button className="iconButton" onClick={() => void load()} disabled={busy} title="Refresh">
              <RefreshCw size={17} />
            </button>
          </div>
          <div className="healthRows">
            <div>
              <span><Activity size={15} /> Mean confidence</span>
              <strong>{percent(metrics?.mean_confidence ?? 0)}</strong>
            </div>
            <div>
              <span><Network size={15} /> P50 latency</span>
              <strong>{(metrics?.p50_latency_ms ?? 0).toFixed(1)} ms</strong>
            </div>
            <div>
              <span><ShieldCheck size={15} /> Active model</span>
              <strong>{registry?.current_model_id ?? model?.model_id ?? "—"}</strong>
            </div>
          </div>

          <div className="distributionBlock">
            <div className="distributionBar">
              <i style={{ width: `${(metrics?.known_rate ?? 0) * 100}%` }} />
            </div>
            <div className="distributionLegend">
              <span>Known {percent(metrics?.known_rate ?? 0)}</span>
              <span>Unknown {percent(metrics?.unknown_rate ?? 0)}</span>
            </div>
          </div>
        </section>

        <section className="surface registryCard">
          <div className="surfaceHeader">
            <div>
              <span className="sectionKicker">MODEL REGISTRY</span>
              <h2>Promotion state</h2>
            </div>
          </div>
          <div className="registryHero">
            <span>Current champion</span>
            <strong>{registry?.current_model_id ?? model?.model_id ?? "—"}</strong>
          </div>
          <div className="registryList">
            {(registry?.entries ?? []).length ? (
              registry!.entries.map((entry) => (
                <div className="registryRow" key={`${entry.model_version}-${entry.gcs_prefix}`}>
                  <div>
                    <strong>{entry.model_version}</strong>
                    <small>{entry.gcs_prefix}</small>
                  </div>
                  <span className={entry.promoted ? "statusBadge succeeded" : "statusBadge queued"}>
                    {entry.promoted ? "promoted" : "registered"}
                  </span>
                </div>
              ))
            ) : (
              <div className="emptyInline">No additional model releases registered.</div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
