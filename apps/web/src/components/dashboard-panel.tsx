"use client";

import {
  Activity,
  ArrowRight,
  FileSearch,
  FileUp,
  Gauge,
  GitBranch,
  Network,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useEffect, useState } from "react";

import { getLearningSummary, getMetricsSummary } from "@/lib/api";
import { FINAL_METRICS } from "@/lib/model-report";
import type { LearningSummary, MetricsSummary, ModelInfo } from "@/lib/types";

type DashboardTab = "analyze" | "relation" | "batch" | "learning" | "monitoring" | "model";

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function DashboardPanel({
  model,
  onNavigate,
}: {
  model: ModelInfo | null;
  onNavigate: (tab: DashboardTab) => void;
}) {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [learning, setLearning] = useState<LearningSummary | null>(null);

  useEffect(() => {
    void Promise.allSettled([getMetricsSummary(500), getLearningSummary()]).then(
      ([metricsResult, learningResult]) => {
        if (metricsResult.status === "fulfilled") setMetrics(metricsResult.value);
        if (learningResult.status === "fulfilled") setLearning(learningResult.value);
      },
    );
  }, []);

  return (
    <div className="dashboardPage">
      <section className="dashboardHero">
        <div>
          <span className="sectionKicker">RELATION INTELLIGENCE</span>
          <h1>Turn documents into structured knowledge.</h1>
          <p>
            Extract entities, relations and review-ready evidence from text and PDF
            in one production workspace.
          </p>
          <div className="heroActions">
            <button className="button buttonPrimary" onClick={() => onNavigate("analyze")}>
              <Sparkles size={17} />
              Analyze document
            </button>
            <button className="button buttonSecondary" onClick={() => onNavigate("relation")}>
              Test a relation
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
        <div className="heroSignalCard">
          <div className="signalTopline">
            <span className={model ? "liveDot live" : "liveDot"} />
            <span>{model ? "Production model online" : "Connecting to model"}</span>
          </div>
          <strong>{model?.model_id ?? "RelationScope"}</strong>
          <div className="signalGrid">
            <div>
              <span>Known relations</span>
              <b>{model?.num_relations ?? 56}</b>
            </div>
            <div>
              <span>Open set</span>
              <b>{model?.open_set_enabled ? "On" : "—"}</b>
            </div>
            <div>
              <span>Runtime</span>
              <b>{model?.device?.toUpperCase() ?? "CPU"}</b>
            </div>
            <div>
              <span>Default</span>
              <b>{model?.default_open_set_profile ?? "balanced"}</b>
            </div>
          </div>
        </div>
      </section>

      <section className="metricGrid dashboardMetrics">
        <article className="metricTile">
          <div className="metricIcon"><ShieldCheck size={18} /></div>
          <span>Known macro-F1</span>
          <strong>{percent(FINAL_METRICS.knownMacroF1)}</strong>
          <small>Frozen evaluation</small>
        </article>
        <article className="metricTile">
          <div className="metricIcon"><Gauge size={18} /></div>
          <span>Unknown rejection</span>
          <strong>{percent(FINAL_METRICS.unknownRejection)}</strong>
          <small>Balanced profile</small>
        </article>
        <article className="metricTile">
          <div className="metricIcon"><Activity size={18} /></div>
          <span>Production P50</span>
          <strong>{(metrics?.p50_latency_ms ?? FINAL_METRICS.warmClientMedianMs).toFixed(0)} ms</strong>
          <small>{metrics ? "Live telemetry" : "Warm client benchmark"}</small>
        </article>
        <article className="metricTile">
          <div className="metricIcon"><GitBranch size={18} /></div>
          <span>Review queue</span>
          <strong>{learning?.pending_review ?? 0}</strong>
          <small>{learning?.drift.status ?? "No drift signal yet"}</small>
        </article>
      </section>

      <section className="pageSection">
        <div className="sectionHeading">
          <div>
            <span className="sectionKicker">WORKSPACE</span>
            <h2>Start with the task, not the model.</h2>
          </div>
        </div>
        <div className="actionCardGrid">
          <button className="actionCard" onClick={() => onNavigate("analyze")}>
            <span className="actionIcon"><FileSearch size={20} /></span>
            <strong>Analyze a document</strong>
            <p>Paste text or upload a PDF and build a relation graph.</p>
            <span className="actionLink">Open analyzer <ArrowRight size={14} /></span>
          </button>
          <button className="actionCard" onClick={() => onNavigate("relation")}>
            <span className="actionIcon"><Network size={20} /></span>
            <strong>Inspect one relation</strong>
            <p>Test a head-tail pair with confidence, abstention and word influence.</p>
            <span className="actionLink">Open relation test <ArrowRight size={14} /></span>
          </button>
          <button className="actionCard" onClick={() => onNavigate("batch")}>
            <span className="actionIcon"><FileUp size={20} /></span>
            <strong>Run a batch</strong>
            <p>Queue durable CSV jobs and export structured predictions.</p>
            <span className="actionLink">Open batch workspace <ArrowRight size={14} /></span>
          </button>
        </div>
      </section>

      <section className="pageSection compactSection">
        <div className="sectionHeading">
          <div>
            <span className="sectionKicker">SYSTEM</span>
            <h2>Production state</h2>
          </div>
          <button className="textButton" onClick={() => onNavigate("monitoring")}>
            View monitoring <ArrowRight size={14} />
          </button>
        </div>
        <div className="systemStrip">
          <div>
            <span>Predictions</span>
            <strong>{metrics?.total_predictions ?? 0}</strong>
          </div>
          <div>
            <span>Known rate</span>
            <strong>{percent(metrics?.known_rate ?? 0)}</strong>
          </div>
          <div>
            <span>Unknown rate</span>
            <strong>{percent(metrics?.unknown_rate ?? 0)}</strong>
          </div>
          <div>
            <span>Drift</span>
            <strong>{learning?.drift.status ?? "insufficient data"}</strong>
          </div>
          <div>
            <span>Canary</span>
            <strong>{learning?.canary.enabled ? `${learning.canary.traffic_percent.toFixed(0)}%` : "off"}</strong>
          </div>
        </div>
      </section>
    </div>
  );
}
