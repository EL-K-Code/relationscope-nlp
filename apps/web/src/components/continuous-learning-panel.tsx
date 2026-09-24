"use client";

import {
  Activity,
  GitBranch,
  RefreshCw,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import { useEffect, useState } from "react";

import { getActiveLearningPolicy, getLearningSummary } from "@/lib/api";
import type {
  ActiveLearningPolicy,
  LearningSummary,
} from "@/lib/types";

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function ContinuousLearningPanel() {
  const [summary, setSummary] = useState<LearningSummary | null>(null);
  const [policy, setPolicy] = useState<ActiveLearningPolicy | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setBusy(true);
    setError(null);
    try {
      const [nextSummary, nextPolicy] = await Promise.all([
        getLearningSummary(),
        getActiveLearningPolicy(),
      ]);
      setSummary(nextSummary);
      setPolicy(nextPolicy);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Continuous-learning state unavailable.",
      );
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  return (
    <div className="learningGrid">
      <section className="panel learningHero">
        <div className="panelHeader">
          <div>
            <span className="step">V3 CONTINUOUS LEARNING</span>
            <h2>Production signals become reviewed training evidence</h2>
          </div>
          <button
            className="iconButton"
            onClick={() => void load()}
            disabled={busy}
            title="Refresh"
          >
            <RefreshCw size={18} />
          </button>
        </div>
        <p className="muted">
          RelationScope never self-trains from raw user traffic. Uncertain,
          explicitly consented examples enter a private review queue, then a
          versioned dataset, retraining run, promotion gate and optional canary.
        </p>

        {error && <div className="errorBanner">{error}</div>}

        <div className="learningMetricGrid">
          <div>
            <small>PENDING REVIEW</small>
            <strong>{summary?.pending_review ?? 0}</strong>
          </div>
          <div>
            <small>REVIEWED</small>
            <strong>{summary?.reviewed_examples ?? 0}</strong>
          </div>
          <div>
            <small>DISMISSED</small>
            <strong>{summary?.dismissed_examples ?? 0}</strong>
          </div>
          <div>
            <small>DRIFT STATUS</small>
            <strong>{summary?.drift.status.toUpperCase() ?? "—"}</strong>
          </div>
          <div>
            <small>CANARY</small>
            <strong>
              {summary?.canary.enabled
                ? `${summary.canary.traffic_percent.toFixed(0)}%`
                : "OFF"}
            </strong>
          </div>
          <div>
            <small>CHAMPION</small>
            <strong>{summary?.canary.champion_model_id ?? "—"}</strong>
          </div>
        </div>
      </section>

      <section className="panel driftCard">
        <div className="panelHeader">
          <div>
            <span className="step">DRIFT</span>
            <h2>Rolling production-distribution comparison</h2>
          </div>
          <ShieldAlert size={20} />
        </div>
        <p className="muted">{summary?.drift.message ?? "Loading drift state…"}</p>
        <div className="driftRows">
          {(summary?.drift.metrics ?? []).map((metric) => (
            <div
              className={metric.triggered ? "driftRow triggered" : "driftRow"}
              key={metric.name}
            >
              <div>
                <strong>{metric.name.replaceAll("_", " ")}</strong>
                <span>
                  baseline {metric.baseline.toFixed(3)} → current{" "}
                  {metric.current.toFixed(3)}
                </span>
              </div>
              <em>{metric.delta >= 0 ? "+" : ""}{metric.delta.toFixed(3)}</em>
            </div>
          ))}
          {summary?.drift.status !== "insufficient_data" && (
            <div className="driftRow">
              <div>
                <strong>relation JS divergence</strong>
                <span>distribution shift across predicted relation IDs</span>
              </div>
              <em>{summary?.drift.relation_js_divergence.toFixed(3)}</em>
            </div>
          )}
        </div>
      </section>

      <section className="panel policyCard">
        <div className="panelHeader">
          <div>
            <span className="step">ACTIVE LEARNING</span>
            <h2>Review prioritization policy</h2>
          </div>
          <Sparkles size={20} />
        </div>
        <div className="policyWeights">
          <div>
            <span>Uncertainty</span>
            <strong>{percent(policy?.uncertainty_weight ?? 0)}</strong>
          </div>
          <div>
            <span>Threshold proximity</span>
            <strong>{percent(policy?.boundary_weight ?? 0)}</strong>
          </div>
          <div>
            <span>UNKNOWN bonus</span>
            <strong>{percent(policy?.unknown_bonus ?? 0)}</strong>
          </div>
          <div>
            <span>Queue threshold</span>
            <strong>{percent(policy?.queue_threshold ?? 0)}</strong>
          </div>
        </div>
      </section>

      <section className="panel canaryCard">
        <div className="panelHeader">
          <div>
            <span className="step">CHAMPION / CHALLENGER</span>
            <h2>Safe model evolution</h2>
          </div>
          <GitBranch size={20} />
        </div>
        <div className="canaryFlow">
          <span>Reviewed dataset</span>
          <b>→</b>
          <span>Retraining</span>
          <b>→</b>
          <span>Frozen evaluation</span>
          <b>→</b>
          <span>Promotion gates</span>
          <b>→</b>
          <span>Canary</span>
          <b>→</b>
          <span>Promote / rollback</span>
        </div>
        <div className="canaryState">
          <Activity size={17} />
          <div>
            <small>CURRENT STATE</small>
            <strong>
              {summary?.canary.enabled
                ? `Canary active at ${summary.canary.traffic_percent.toFixed(0)}%`
                : "Champion-only traffic"}
            </strong>
            <span>
              challenger{" "}
              {summary?.canary.challenger_model_version ?? "not registered"}
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}
