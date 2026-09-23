"use client";

import {
  Activity,
  Braces,
  BrainCircuit,
  FileUp,
  FlaskConical,
  Network,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  explainRelation,
  getModelInfo,
  predictBatch,
  predictOpenSet,
  predictRelation,
} from "@/lib/api";
import { KnowledgeGraph } from "@/components/knowledge-graph";
import { RiskCoverageChart } from "@/components/risk-coverage-chart";
import { exportBatchResults, parseBatchCsv } from "@/lib/csv";
import {
  EVALUATION_PROTOCOL,
  FINAL_METRICS,
  PROFILE_METRICS,
} from "@/lib/model-report";
import type {
  BatchInput,
  BatchResult,
  ExplainResponse,
  ModelInfo,
  OpenSetPredictResponse,
  OpenSetProfile,
  PredictResponse,
} from "@/lib/types";

type Tab = "extract" | "batch" | "lab";

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

export function RelationScopeStudio() {
  const [tab, setTab] = useState<Tab>("extract");
  const [text, setText] = useState(DEMO_TEXT);
  const [headText, setHeadText] = useState("Marie Curie");
  const [tailText, setTailText] = useState("Warsaw");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [openSet, setOpenSet] = useState<OpenSetPredictResponse | null>(null);
  const [profile, setProfile] = useState<OpenSetProfile>("balanced");
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [batchRows, setBatchRows] = useState<BatchInput[]>([]);
  const [batchResults, setBatchResults] = useState<BatchResult[]>([]);

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
  }, []);

  const evidenceMax = useMemo(
    () =>
      Math.max(
        0.000001,
        ...(explanation?.evidence.map((item) =>
          Math.max(item.confidence_drop, 0),
        ) ?? []),
      ),
    [explanation],
  );

  async function runExtraction() {
    setBusy(true);
    setError(null);
    setPrediction(null);
    setOpenSet(null);
    setExplanation(null);

    try {
      const head = resolveUniqueSpan(text, headText);
      const tail = resolveUniqueSpan(text, tailText);
      const base = await predictRelation(text, head, tail, 5);
      setPrediction(base);

      if (model?.open_set_enabled) {
        setOpenSet(await predictOpenSet(text, head, tail, 5));
      }

      setExplanation(await explainRelation(text, head, tail, 12));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleCsv(file: File) {
    setError(null);
    setBatchResults([]);
    try {
      const rows = parseBatchCsv(await file.text());
      if (rows.length > 128) {
        throw new Error("The demo accepts at most 128 rows per batch.");
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

    try {
      const responses = await predictBatch(batchRows, 1);
      setBatchResults(
        batchRows.map((row, index) => ({
          ...row,
          relation: responses[index]?.predictions[0],
        })),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Batch failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">R</div>
          <div>
            <strong>RelationScope</strong>
            <span>NLP Studio</span>
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
            <FileUp size={18} /> Batch
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
            <p className="eyebrow">RELATION EXTRACTION SYSTEM</p>
            <h1>
              From text to <span>structured knowledge.</span>
            </h1>
          </div>
          <div className="headerPills">
            <span><BrainCircuit size={15} /> entity-aware</span>
            <span><ShieldCheck size={15} /> calibrated</span>
          </div>
        </header>

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
                    Run the model to inspect the predicted relation, confidence,
                    open-set decision and token evidence.
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
                        threshold {confidence(openSet.threshold)}
                      </span>
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

            <section className="panel evidencePanel">
              <div className="panelHeader">
                <div>
                  <span className="step">03</span>
                  <h2>Evidence map</h2>
                </div>
                <BrainCircuit size={20} />
              </div>

              {!explanation ? (
                <p className="muted">
                  Token-occlusion evidence will appear here after inference.
                </p>
              ) : (
                <>
                  <p className="muted">
                    Tokens are ranked by the drop in prediction confidence when
                    individually masked. This is evidence by occlusion, not a
                    causal explanation.
                  </p>
                  <div className="evidenceTokens">
                    {explanation.evidence.map((item) => (
                      <span
                        key={`${item.position}-${item.token}`}
                        style={{
                          ["--impact" as string]: Math.max(
                            item.confidence_drop,
                            0,
                          ) / evidenceMax,
                        }}
                        title={`confidence drop: ${item.confidence_drop.toFixed(4)}`}
                      >
                        {item.token}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </section>
          </div>
        )}

        {tab === "batch" && (
          <div className="workspace singleColumn">
            <section className="panel">
              <div className="panelHeader">
                <div>
                  <span className="step">BATCH</span>
                  <h2>Knowledge extraction at scale</h2>
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
                <span>{batchRows.length} rows ready</span>
                <button
                  className="primaryButton compact"
                  disabled={!batchRows.length || busy}
                  onClick={runBatch}
                >
                  Run batch
                </button>
                {batchResults.length > 0 && (
                  <button
                    className="secondaryButton"
                    onClick={() =>
                      downloadText(
                        "relationscope-results.csv",
                        exportBatchResults(batchResults),
                      )
                    }
                  >
                    Export results
                  </button>
                )}
              </div>

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

        {tab === "lab" && (
          <div className="labGrid">
            <article className="panel metricCard">
              <small>EXPERIMENTAL LINEAGE</small>
              <h3>LSTM → Attention → T5 → Transformer</h3>
              <p>
                The production system keeps the ENSAI notebook as an explicit
                research lineage rather than replacing it.
              </p>
            </article>
            <article className="panel metricCard">
              <small>ABLATION</small>
              <h3>Do entity markers actually help?</h3>
              <p>
                Raw CLS, marked CLS and entity-pair pooling are trained under
                matched conditions and compared on Macro-F1 and calibration.
              </p>
            </article>
            <article className="panel metricCard">
              <small>OPEN SET</small>
              <h3>Never-seen relation detection</h3>
              <p>
                Relation-level holdouts evaluate whether confidence can support
                a defensible UNKNOWN_RELATION decision.
              </p>
            </article>
            <article className="panel metricCard">
              <small>QUALITY</small>
              <h3>No metric without reproduction</h3>
              <p>
                README and CV claims are intentionally blocked until committed
                scripts reproduce the result on a frozen protocol.
              </p>
            </article>

            <section className="panel architectureCard">
              <div className="panelHeader">
                <div>
                  <span className="step">SYSTEM</span>
                  <h2>Production architecture</h2>
                </div>
                <Network size={20} />
              </div>
              <div className="architectureFlow">
                <span>Text + entity pair</span>
                <b>→</b>
                <span>HEAD / TAIL markers</span>
                <b>→</b>
                <span>Transformer encoder</span>
                <b>→</b>
                <span>Entity-aware pooling</span>
                <b>→</b>
                <span>Calibration</span>
                <b>→</b>
                <span>Relation / UNKNOWN</span>
              </div>
            </section>
          </div>
        )}
      </section>
    </main>
  );
}
