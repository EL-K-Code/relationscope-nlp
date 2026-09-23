"use client";

import {
  Activity,
  BarChart3,
  Braces,
  BrainCircuit,
  CheckCircle2,
  FileUp,
  FlaskConical,
  MessageSquareText,
  Network,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { KnowledgeGraph } from "@/components/knowledge-graph";
import { RiskCoverageChart } from "@/components/risk-coverage-chart";
import {
  createBatchJob,
  explainRelation,
  getBatchJob,
  getMetricsSummary,
  getModelInfo,
  getModelRegistry,
  getRelationCatalog,
  predictOpenSet,
  predictRelation,
  submitFeedback,
} from "@/lib/api";
import { exportBatchResults, parseBatchCsv } from "@/lib/csv";
import {
  EVALUATION_PROTOCOL,
  FINAL_METRICS,
  PROFILE_METRICS,
} from "@/lib/model-report";
import type {
  BatchInput,
  BatchJobSummary,
  BatchResult,
  InfluenceExplainResponse,
  MetricsSummary,
  ModelInfo,
  ModelRegistry,
  OpenSetPredictResponse,
  OpenSetProfile,
  PredictResponse,
  RelationCatalogItem,
} from "@/lib/types";

type Tab = "extract" | "batch" | "ops" | "lab";

const PROFILE_ORDER: OpenSetProfile[] = [
  "high_coverage",
  "balanced",
  "high_safety",
];

const DEMO_TEXT = "Marie Curie was born in Warsaw in 1867.";

function resolveUniqueSpan(text: string, mention: string) {
  const start = text.indexOf(mention);
  if (start < 0) {
    throw new Error(`Entity "${mention}" was not found in the sentence.`);
  }
  if (text.indexOf(mention, start + 1) >= 0) {
    throw new Error(
      `Entity "${mention}" occurs more than once. Use explicit offsets in batch mode.`,
    );
  }
  return { text: mention, start, end: start + mention.length };
}

function confidence(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function wait(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export function RelationScopeStudio() {
  const [tab, setTab] = useState<Tab>("extract");
  const [text, setText] = useState(DEMO_TEXT);
  const [headText, setHeadText] = useState("Marie Curie");
  const [tailText, setTailText] = useState("Warsaw");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [openSet, setOpenSet] = useState<OpenSetPredictResponse | null>(null);
  const [profile, setProfile] = useState<OpenSetProfile>("balanced");
  const [explanation, setExplanation] =
    useState<InfluenceExplainResponse | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [batchRows, setBatchRows] = useState<BatchInput[]>([]);
  const [batchResults, setBatchResults] = useState<BatchResult[]>([]);
  const [batchJob, setBatchJob] = useState<BatchJobSummary | null>(null);
  const [feedbackConsent, setFeedbackConsent] = useState(false);
  const [correction, setCorrection] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [registry, setRegistry] = useState<ModelRegistry | null>(null);
  const [relations, setRelations] = useState<RelationCatalogItem[]>([]);
  const [opsBusy, setOpsBusy] = useState(false);

  useEffect(() => {
    getModelInfo()
      .then((info) => {
        setModel(info);
        if (info.default_open_set_profile) {
          setProfile(info.default_open_set_profile);
        }
      })
      .catch((reason: unknown) =>
        setModelError(reason instanceof Error ? reason.message : "Model offline"),
      );

    getRelationCatalog()
      .then((catalog) => setRelations(catalog.items))
      .catch(() => setRelations([]));
  }, []);

  useEffect(() => {
    if (tab === "ops") {
      void loadOps();
    }
  }, [tab]);

  const topSupporting = useMemo(
    () =>
      [...(explanation?.influences ?? [])]
        .filter((item) => item.confidence_drop > 0)
        .sort((a, b) => b.confidence_drop - a.confidence_drop)
        .slice(0, 5),
    [explanation],
  );

  async function runExtraction() {
    setBusy(true);
    setError(null);
    setPrediction(null);
    setOpenSet(null);
    setExplanation(null);
    setFeedbackStatus(null);

    try {
      const head = resolveUniqueSpan(text, headText);
      const tail = resolveUniqueSpan(text, tailText);

      if (model?.open_set_enabled !== false) {
        const selective = await predictOpenSet(text, head, tail, 5, profile);
        setOpenSet(selective);
        setPrediction(selective);

        const targetRelation = selective.predictions[0]?.relation_id;
        void explainRelation(text, head, tail, profile, targetRelation)
          .then(setExplanation)
          .catch(() => setExplanation(null));
      } else {
        setPrediction(await predictRelation(text, head, tail, 5));
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }

  async function sendFeedback(
    verdict: "confirm" | "correct" | "unknown" | "skip",
  ) {
    if (!openSet || !openSet.predictions[0]) return;
    if (verdict === "correct" && !correction.trim()) {
      setFeedbackStatus("Enter the corrected relation name first.");
      return;
    }

    try {
      const head = resolveUniqueSpan(text, headText);
      const tail = resolveUniqueSpan(text, tailText);
      const top = openSet.predictions[0];

      const response = await submitFeedback({
        text,
        head,
        tail,
        prediction: {
          model_id: openSet.model_id,
          relation_id: top.relation_id,
          relation_name: top.relation_name,
          confidence: top.confidence,
          decision: openSet.decision,
          profile: openSet.profile,
          threshold: openSet.threshold,
        },
        verdict,
        corrected_relation_id:
          verdict === "correct"
            ? relations.find(
                (item) =>
                  item.relation_name.toLowerCase() === correction.trim().toLowerCase(),
              )?.relation_id ?? null
            : null,
        corrected_relation_name:
          verdict === "correct" ? correction.trim() : null,
        consent_to_store_text: feedbackConsent,
      });

      setFeedbackStatus(
        response.stored_text
          ? "Feedback saved with training-data consent."
          : "Feedback saved privately without raw text.",
      );
    } catch (reason) {
      setFeedbackStatus(
        reason instanceof Error ? reason.message : "Feedback could not be saved.",
      );
    }
  }

  async function handleCsv(file: File) {
    setError(null);
    setBatchResults([]);
    setBatchJob(null);
    try {
      const rows = parseBatchCsv(await file.text());
      if (rows.length > 2000) {
        throw new Error("V2 accepts at most 2,000 rows per asynchronous job.");
      }
      setBatchRows(rows);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Invalid CSV.");
    }
  }

  async function runBatch() {
    if (!batchRows.length) return;
    setBusy(true);
    setError(null);
    setBatchResults([]);

    try {
      const created = await createBatchJob(batchRows, profile, 1);
      setBatchJob(created);

      let current = await getBatchJob(created.job_id);
      setBatchJob(current);

      for (let attempt = 0; attempt < 120; attempt += 1) {
        if (current.status === "succeeded" || current.status === "failed") {
          break;
        }
        await wait(2000);
        current = await getBatchJob(created.job_id);
        setBatchJob(current);
      }

      if (current.status === "failed") {
        throw new Error(current.error ?? "Asynchronous batch job failed.");
      }
      if (current.status !== "succeeded") {
        throw new Error(
          "The job is still running. Its durable job ID is shown below; refresh Ops later.",
        );
      }

      setBatchResults(
        batchRows.map((row, index) => ({
          ...row,
          relation: current.results?.[index]?.relation,
        })),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Batch failed.");
    } finally {
      setBusy(false);
    }
  }

  async function loadOps() {
    setOpsBusy(true);
    try {
      const [summary, models] = await Promise.all([
        getMetricsSummary(500),
        getModelRegistry(),
      ]);
      setMetrics(summary);
      setRegistry(models);
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Operational metrics unavailable.",
      );
    } finally {
      setOpsBusy(false);
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">R</div>
          <div>
            <strong>RelationScope</strong>
            <span>NLP Studio · V2</span>
          </div>
        </div>

        <nav>
          <button
            className={tab === "extract" ? "navActive" : ""}
            onClick={() => setTab("extract")}
          >
            <Sparkles size={18} /> Extract
          </button>
          <button
            className={tab === "batch" ? "navActive" : ""}
            onClick={() => setTab("batch")}
          >
            <FileUp size={18} /> Async Batch
          </button>
          <button
            className={tab === "ops" ? "navActive" : ""}
            onClick={() => setTab("ops")}
          >
            <BarChart3 size={18} /> Ops
          </button>
          <button
            className={tab === "lab" ? "navActive" : ""}
            onClick={() => setTab("lab")}
          >
            <FlaskConical size={18} /> Model Lab
          </button>
        </nav>

        <div className="modelStatus">
          <span className={model ? "statusDot online" : "statusDot"} />
          <div>
            <small>MODEL</small>
            <strong>{model?.model_id ?? "Not loaded"}</strong>
            <span>
              {model
                ? `${model.num_relations} relations · ${model.device}`
                : modelError ?? "Waiting for API"}
            </span>
          </div>
        </div>
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">RELATION EXTRACTION · OPEN SET · MLOPS</p>
            <h1>
              From text to <span>structured knowledge.</span>
            </h1>
          </div>
          <div className="headerPills">
            <span><BrainCircuit size={15} /> word influence</span>
            <span><ShieldCheck size={15} /> calibrated</span>
            <span><Activity size={15} /> observable</span>
          </div>
        </header>

        <div className="performanceStrip">
          <div>
            <small>KNOWN MACRO-F1</small>
            <strong>{confidence(FINAL_METRICS.knownMacroF1)}</strong>
            <span>56 known relations</span>
          </div>
          <div>
            <small>OOD AUROC</small>
            <strong>{confidence(FINAL_METRICS.oodAuRoc)}</strong>
            <span>16 unseen relations</span>
          </div>
          <div>
            <small>UNKNOWN F1</small>
            <strong>{confidence(FINAL_METRICS.unknownF1)}</strong>
            <span>balanced profile</span>
          </div>
          <div>
            <small>WARM CPU P50</small>
            <strong>{FINAL_METRICS.warmCpuMedianMs.toFixed(1)} ms</strong>
            <span>Cloud Run · 2 vCPU</span>
          </div>
        </div>

        {error && <div className="errorBanner">{error}</div>}

        {tab === "extract" && (
          <div className="workspace">
            <section className="panel inputPanel">
              <div className="panelHeader">
                <div>
                  <span className="step">01</span>
                  <h2>Define the relation context</h2>
                </div>
                <Braces size={20} />
              </div>

              <label>
                Sentence
                <textarea
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  rows={6}
                />
              </label>

              <div className="entityGrid">
                <label>
                  Head entity
                  <input
                    value={headText}
                    onChange={(event) => setHeadText(event.target.value)}
                  />
                </label>
                <label>
                  Tail entity
                  <input
                    value={tailText}
                    onChange={(event) => setTailText(event.target.value)}
                  />
                </label>
              </div>

              <div className="profileBlock">
                <div className="profileBlockHeader">
                  <div>
                    <small>OPERATING PROFILE</small>
                    <strong>{PROFILE_METRICS[profile].label}</strong>
                  </div>
                  <span>
                    threshold {confidence(PROFILE_METRICS[profile].threshold)}
                  </span>
                </div>
                <div className="profileSelector">
                  {(model?.open_set_profiles?.length
                    ? model.open_set_profiles
                    : PROFILE_ORDER
                  ).map((item) => (
                    <button
                      key={item}
                      type="button"
                      className={item === profile ? "profileActive" : ""}
                      onClick={() => setProfile(item)}
                    >
                      <strong>{PROFILE_METRICS[item].shortLabel}</strong>
                      <span>{confidence(PROFILE_METRICS[item].threshold)}</span>
                    </button>
                  ))}
                </div>
                <p>{PROFILE_METRICS[profile].description}</p>
              </div>

              <button className="primaryButton" onClick={runExtraction} disabled={busy}>
                <Sparkles size={17} />
                {busy ? "Analysing…" : "Extract relation"}
              </button>
            </section>

            <section className="panel resultPanel">
              <div className="panelHeader">
                <div>
                  <span className="step">02</span>
                  <h2>Model decision</h2>
                </div>
                <Activity size={20} />
              </div>

              {!prediction ? (
                <div className="emptyState">
                  <Network size={44} />
                  <strong>No prediction yet</strong>
                  <p>
                    Run the model to inspect the relation, selective decision and
                    word-level influence map.
                  </p>
                </div>
              ) : (
                <>
                  <div className="heroPrediction">
                    <p>TOP RELATION</p>
                    <h3>{prediction.predictions[0]?.relation_name}</h3>
                    <div className="heroMeta">
                      <strong>
                        {confidence(prediction.predictions[0]?.confidence ?? 0)}
                      </strong>
                      <span>{prediction.latency_ms.toFixed(1)} ms</span>
                    </div>
                  </div>

                  {openSet && (
                    <div
                      className={
                        openSet.accepted
                          ? "decisionCard accepted"
                          : "decisionCard rejected"
                      }
                    >
                      <ShieldCheck size={18} />
                      <div>
                        <small>SELECTIVE PREDICTION</small>
                        <strong>
                          {openSet.accepted
                            ? "KNOWN RELATION"
                            : "UNKNOWN RELATION"}
                        </strong>
                      </div>
                      <span>
                        {PROFILE_METRICS[openSet.profile].label} · threshold{" "}
                        {confidence(openSet.threshold)}
                      </span>
                    </div>
                  )}

                  {openSet && (
                    <div className="decisionTelemetry">
                      <div>
                        <small>CONFIDENCE</small>
                        <strong>
                          {confidence(openSet.predictions[0]?.confidence ?? 0)}
                        </strong>
                      </div>
                      <div>
                        <small>THRESHOLD</small>
                        <strong>{confidence(openSet.threshold)}</strong>
                      </div>
                      <div>
                        <small>MARGIN</small>
                        <strong>
                          {(
                            ((openSet.predictions[0]?.confidence ?? 0) -
                              openSet.threshold) *
                            100
                          ).toFixed(1)}
                          pp
                        </strong>
                      </div>
                    </div>
                  )}

                  <div className="ranking">
                    {prediction.predictions.map((item, index) => (
                      <div className="rankRow" key={item.relation_id}>
                        <span>{String(index + 1).padStart(2, "0")}</span>
                        <div>
                          <div className="rankLabel">
                            <strong>{item.relation_name}</strong>
                            <em>{confidence(item.confidence)}</em>
                          </div>
                          <div className="bar">
                            <i style={{ width: confidence(item.confidence) }} />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </section>

            <section className="panel evidencePanel v2EvidencePanel">
              <div className="panelHeader">
                <div>
                  <span className="step">03</span>
                  <h2>Word influence map</h2>
                </div>
                <BrainCircuit size={20} />
              </div>

              {!explanation ? (
                <p className="muted">
                  V2 masks each complete word and re-runs the target relation to
                  measure how its calibrated confidence changes.
                </p>
              ) : (
                <>
                  <p className="muted">
                    Darker supporting words reduce the relation confidence most
                    when removed. Opposing words increase confidence when removed.
                    This is perturbation-based evidence, not a causal claim.
                  </p>
                  <div className="influenceLegend">
                    <span><i className="supportSwatch" /> supports relation</span>
                    <span><i className="opposeSwatch" /> opposes relation</span>
                  </div>
                  <div className="wordInfluenceMap">
                    {explanation.influences.map((item) => (
                      <span
                        key={`${item.start}-${item.end}-${item.word}`}
                        className={`influenceWord ${item.direction}`}
                        style={{
                          ["--strength" as string]: Math.max(
                            0.08,
                            Math.abs(item.normalized_influence),
                          ),
                        }}
                        title={`confidence effect: ${item.confidence_drop >= 0 ? "+" : ""}${(
                          item.confidence_drop * 100
                        ).toFixed(2)} pp`}
                      >
                        {item.word}
                      </span>
                    ))}
                  </div>
                  {topSupporting.length > 0 && (
                    <div className="topCues">
                      <small>STRONGEST SUPPORTING CUES</small>
                      <div>
                        {topSupporting.map((item) => (
                          <span key={`${item.start}-cue`}>
                            {item.word} · +{(item.confidence_drop * 100).toFixed(2)} pp
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="explainMeta">
                    <span>{explanation.method}</span>
                    <span>{explanation.latency_ms.toFixed(1)} ms</span>
                    <span>{explanation.model_id}</span>
                  </div>
                </>
              )}
            </section>

            {openSet && openSet.predictions[0] && (
              <section className="panel feedbackPanel">
                <div className="panelHeader">
                  <div>
                    <span className="step">04</span>
                    <h2>Human feedback loop</h2>
                  </div>
                  <MessageSquareText size={20} />
                </div>
                <p className="muted">
                  Confirm the model, flag the pair as unknown, or provide a
                  correction. Raw text is never persisted unless you explicitly
                  opt in below.
                </p>
                <div className="feedbackActions">
                  <button
                    className="secondaryButton"
                    onClick={() => void sendFeedback("confirm")}
                  >
                    <CheckCircle2 size={16} /> Confirm
                  </button>
                  <button
                    className="secondaryButton"
                    onClick={() => void sendFeedback("unknown")}
                  >
                    Mark unknown
                  </button>
                </div>
                <div className="correctionRow">
                  <input
                    list="relation-catalog"
                    placeholder="Correct relation name"
                    value={correction}
                    onChange={(event) => setCorrection(event.target.value)}
                  />
                  <datalist id="relation-catalog">
                    {relations.map((item) => (
                      <option
                        key={item.relation_id}
                        value={item.relation_name}
                      />
                    ))}
                  </datalist>
                  <button
                    className="secondaryButton"
                    onClick={() => void sendFeedback("correct")}
                  >
                    Save correction
                  </button>
                </div>
                <label className="consentRow">
                  <input
                    type="checkbox"
                    checked={feedbackConsent}
                    onChange={(event) => setFeedbackConsent(event.target.checked)}
                  />
                  Store this sentence and entity spans for future reviewed retraining
                </label>
                {feedbackStatus && (
                  <div className="feedbackStatus">{feedbackStatus}</div>
                )}
              </section>
            )}
          </div>
        )}

        {tab === "batch" && (
          <div className="workspace singleColumn">
            <section className="panel">
              <div className="panelHeader">
                <div>
                  <span className="step">ASYNC BATCH</span>
                  <h2>Durable knowledge extraction jobs</h2>
                </div>
                <FileUp size={20} />
              </div>

              <div className="uploadBox">
                <input
                  id="csv"
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void handleCsv(file);
                  }}
                />
                <label htmlFor="csv">
                  <FileUp size={28} />
                  <strong>Import entity-pair CSV</strong>
                  <a className="sampleLink" href="/sample-relations.csv" download>
                    Download sample CSV
                  </a>
                  <span>
                    id, text, head_text, head_start, head_end, tail_text,
                    tail_start, tail_end
                  </span>
                </label>
              </div>

              <div className="batchActions">
                <span>{batchRows.length} rows ready · max 2,000</span>
                <button
                  className="primaryButton compact"
                  disabled={!batchRows.length || busy}
                  onClick={runBatch}
                >
                  {busy ? "Processing…" : "Queue batch job"}
                </button>
                {batchResults.length > 0 && (
                  <button
                    className="secondaryButton"
                    onClick={() =>
                      downloadText(
                        "relationscope-v2-results.csv",
                        exportBatchResults(batchResults),
                      )
                    }
                  >
                    Export results
                  </button>
                )}
              </div>

              {batchJob && (
                <div className="jobStatus">
                  <div>
                    <small>JOB</small>
                    <strong>{batchJob.job_id}</strong>
                  </div>
                  <div>
                    <small>STATUS</small>
                    <strong>{batchJob.status.toUpperCase()}</strong>
                  </div>
                  <div>
                    <small>PROGRESS</small>
                    <strong>
                      {batchJob.processed}/{batchJob.total}
                    </strong>
                  </div>
                  <div>
                    <small>MODEL</small>
                    <strong>{batchJob.model_id ?? "pending"}</strong>
                  </div>
                </div>
              )}

              {batchResults.length > 0 && (
                <KnowledgeGraph results={batchResults} />
              )}

              {batchResults.length > 0 && (
                <div className="tableWrap">
                  <table>
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>HEAD</th>
                        <th>RELATION</th>
                        <th>TAIL</th>
                        <th>CONF.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.slice(0, 100).map((row) => (
                        <tr key={row.id}>
                          <td>{row.id}</td>
                          <td>{row.head.text}</td>
                          <td>{row.relation?.relation_name ?? "—"}</td>
                          <td>{row.tail.text}</td>
                          <td>
                            {row.relation
                              ? confidence(row.relation.confidence)
                              : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </div>
        )}

        {tab === "ops" && (
          <div className="labGrid">
            <section className="panel opsHero">
              <div className="panelHeader">
                <div>
                  <span className="step">V2 OBSERVABILITY</span>
                  <h2>Privacy-preserving production telemetry</h2>
                </div>
                <button
                  className="iconButton"
                  onClick={() => void loadOps()}
                  disabled={opsBusy}
                  title="Refresh"
                >
                  <RefreshCw size={18} />
                </button>
              </div>
              <p className="muted">
                Operational events persist model version, profile, confidence,
                decision and latency. Raw sentences are not written to telemetry.
              </p>
              <div className="opsMetricGrid">
                <div>
                  <small>PREDICTIONS</small>
                  <strong>{metrics?.total_predictions ?? 0}</strong>
                </div>
                <div>
                  <small>KNOWN RATE</small>
                  <strong>{confidence(metrics?.known_rate ?? 0)}</strong>
                </div>
                <div>
                  <small>UNKNOWN RATE</small>
                  <strong>{confidence(metrics?.unknown_rate ?? 0)}</strong>
                </div>
                <div>
                  <small>P50 LATENCY</small>
                  <strong>{(metrics?.p50_latency_ms ?? 0).toFixed(1)} ms</strong>
                </div>
                <div>
                  <small>P95 LATENCY</small>
                  <strong>{(metrics?.p95_latency_ms ?? 0).toFixed(1)} ms</strong>
                </div>
                <div>
                  <small>MEAN CONF.</small>
                  <strong>{confidence(metrics?.mean_confidence ?? 0)}</strong>
                </div>
              </div>
            </section>

            <section className="panel registryCard">
              <div className="panelHeader">
                <div>
                  <span className="step">MODEL LIFECYCLE</span>
                  <h2>Registry & promotion state</h2>
                </div>
                <Network size={20} />
              </div>
              <div className="registryCurrent">
                <small>PROMOTED</small>
                <strong>{registry?.current_model_id ?? model?.model_id ?? "—"}</strong>
              </div>
              <div className="registryList">
                {(registry?.entries ?? []).map((entry) => (
                  <div key={`${entry.model_version}-${entry.gcs_prefix}`}>
                    <div>
                      <strong>{entry.model_version}</strong>
                      <span>{entry.promoted ? "PROMOTED" : "REGISTERED"}</span>
                    </div>
                    <small>{entry.gcs_prefix}</small>
                  </div>
                ))}
              </div>
            </section>

            <section className="panel architectureCard">
              <div className="panelHeader">
                <div>
                  <span className="step">V2 LOOP</span>
                  <h2>From inference to better training data</h2>
                </div>
                <Activity size={20} />
              </div>
              <div className="architectureFlow">
                <span>Inference</span>
                <b>→</b>
                <span>Telemetry</span>
                <b>→</b>
                <span>Human feedback</span>
                <b>→</b>
                <span>Consented review data</span>
                <b>→</b>
                <span>Retraining export</span>
                <b>→</b>
                <span>Immutable model release</span>
              </div>
            </section>
          </div>
        )}

        {tab === "lab" && (
          <div className="labGrid">
            <section className="panel labSummaryCard">
              <div>
                <span className="step">PROMOTED MODEL</span>
                <h2>ModernBERT with entity-aware selective prediction</h2>
                <p>
                  Frozen evaluation on 56 known relations plus 16 final unseen
                  relations. V2 adds word-level perturbation explanations without
                  changing the frozen v1 benchmark.
                </p>
              </div>
              <div className="labSummaryMetrics">
                <div>
                  <small>KNOWN TOP-3</small>
                  <strong>{confidence(FINAL_METRICS.knownTop3)}</strong>
                </div>
                <div>
                  <small>KNOWN ECE</small>
                  <strong>{confidence(FINAL_METRICS.knownEce)}</strong>
                </div>
                <div>
                  <small>UNKNOWN REJECTION</small>
                  <strong>{confidence(FINAL_METRICS.unknownRejection)}</strong>
                </div>
                <div>
                  <small>CLIENT P50</small>
                  <strong>{FINAL_METRICS.warmClientMedianMs.toFixed(0)} ms</strong>
                </div>
              </div>
            </section>

            <section className="panel chartPanel">
              <RiskCoverageChart />
            </section>

            <section className="panel profileLabCard">
              <div className="panelHeader">
                <div>
                  <span className="step">PROFILES</span>
                  <h2>Frozen operating points</h2>
                </div>
                <ShieldCheck size={20} />
              </div>
              <div className="profileMatrix">
                {PROFILE_ORDER.map((item) => {
                  const metricsForProfile = PROFILE_METRICS[item];
                  return (
                    <article key={item}>
                      <div className="profileMatrixHead">
                        <strong>{metricsForProfile.label}</strong>
                        <span>{confidence(metricsForProfile.threshold)}</span>
                      </div>
                      <dl>
                        <div>
                          <dt>Known coverage</dt>
                          <dd>{confidence(metricsForProfile.knownAcceptance)}</dd>
                        </div>
                        <div>
                          <dt>Accepted accuracy</dt>
                          <dd>
                            {confidence(metricsForProfile.acceptedKnownAccuracy)}
                          </dd>
                        </div>
                        <div>
                          <dt>Unknown rejection</dt>
                          <dd>{confidence(metricsForProfile.unknownRejection)}</dd>
                        </div>
                        <div>
                          <dt>Unknown F1</dt>
                          <dd>{confidence(metricsForProfile.unknownF1)}</dd>
                        </div>
                      </dl>
                    </article>
                  );
                })}
              </div>
            </section>

            <section className="panel protocolCard">
              <div className="panelHeader">
                <div>
                  <span className="step">EVALUATION</span>
                  <h2>Open-set protocol</h2>
                </div>
                <FlaskConical size={20} />
              </div>
              <p className="protocolName">{EVALUATION_PROTOCOL.name}</p>
              <div className="protocolGrid">
                <div>
                  <small>KNOWN RELATIONS</small>
                  <strong>{EVALUATION_PROTOCOL.knownRelations}</strong>
                </div>
                <div>
                  <small>CALIBRATION UNKNOWN</small>
                  <strong>{EVALUATION_PROTOCOL.calibrationUnknownRelations}</strong>
                </div>
                <div>
                  <small>FINAL UNSEEN</small>
                  <strong>{EVALUATION_PROTOCOL.finalUnknownRelations}</strong>
                </div>
                <div>
                  <small>TRAIN KNOWN</small>
                  <strong>{EVALUATION_PROTOCOL.trainKnown.toLocaleString()}</strong>
                </div>
                <div>
                  <small>TEST KNOWN</small>
                  <strong>{EVALUATION_PROTOCOL.testKnown.toLocaleString()}</strong>
                </div>
                <div>
                  <small>TEST UNKNOWN</small>
                  <strong>{EVALUATION_PROTOCOL.testUnknown.toLocaleString()}</strong>
                </div>
              </div>
            </section>

            <section className="panel architectureCard">
              <div className="panelHeader">
                <div>
                  <span className="step">SYSTEM</span>
                  <h2>Production architecture V2</h2>
                </div>
                <Network size={20} />
              </div>
              <div className="architectureFlow">
                <span>Text + entity pair</span>
                <b>→</b>
                <span>HEAD / TAIL markers</span>
                <b>→</b>
                <span>ModernBERT</span>
                <b>→</b>
                <span>Entity-pair pooling</span>
                <b>→</b>
                <span>Calibration</span>
                <b>→</b>
                <span>Relation / UNKNOWN</span>
              </div>
              <div className="deploymentLine">
                <span>Word occlusion</span>
                <span>GCS ops store</span>
                <span>Cloud Tasks batch</span>
                <span>Feedback loop</span>
                <span>Model registry</span>
              </div>
            </section>
          </div>
        )}
      </section>
    </main>
  );
}
