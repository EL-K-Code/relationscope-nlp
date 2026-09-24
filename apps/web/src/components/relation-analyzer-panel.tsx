"use client";

import {
  Activity,
  CheckCircle2,
  MessageSquareText,
  Network,
  RotateCcw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useMemo, useState } from "react";

import {
  explainRelation,
  predictOpenSet,
  predictRelation,
  submitFeedback,
} from "@/lib/api";
import { PROFILE_METRICS } from "@/lib/model-report";
import type {
  InfluenceExplainResponse,
  ModelInfo,
  OpenSetPredictResponse,
  OpenSetProfile,
  PredictResponse,
  RelationCatalogItem,
} from "@/lib/types";

const PROFILE_ORDER: OpenSetProfile[] = [
  "high_coverage",
  "balanced",
  "high_safety",
];

const DEMO_TEXT = "Albert Einstein worked at Princeton University.";

function resolveUniqueSpan(text: string, mention: string) {
  const start = text.indexOf(mention);
  if (start < 0) throw new Error(`Entity "${mention}" was not found in the sentence.`);
  if (text.indexOf(mention, start + 1) >= 0) {
    throw new Error(`Entity "${mention}" occurs more than once.`);
  }
  return { text: mention, start, end: start + mention.length };
}

function confidence(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function RelationAnalyzerPanel({
  model,
  relations,
}: {
  model: ModelInfo | null;
  relations: RelationCatalogItem[];
}) {
  const [text, setText] = useState(DEMO_TEXT);
  const [headText, setHeadText] = useState("Albert Einstein");
  const [tailText, setTailText] = useState("Princeton University");
  const [profile, setProfile] = useState<OpenSetProfile>(
    model?.default_open_set_profile ?? "balanced",
  );
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [openSet, setOpenSet] = useState<OpenSetPredictResponse | null>(null);
  const [explanation, setExplanation] = useState<InfluenceExplainResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feedbackConsent, setFeedbackConsent] = useState(false);
  const [correction, setCorrection] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);

  const topSupporting = useMemo(
    () =>
      [...(explanation?.influences ?? [])]
        .filter((item) => item.confidence_drop > 0)
        .sort((a, b) => b.confidence_drop - a.confidence_drop)
        .slice(0, 4),
    [explanation],
  );

  const matchedCorrection = useMemo(() => {
    const normalized = correction.trim().toLowerCase();
    if (!normalized) return null;
    return (
      relations.find(
        (item) => item.relation_name.toLowerCase() === normalized,
      ) ?? null
    );
  }, [correction, relations]);

  function clearAnalysisState() {
    setPrediction(null);
    setOpenSet(null);
    setExplanation(null);
    setError(null);
    setCorrection("");
    setFeedbackStatus(null);
  }

  function startNewAnalysis() {
    clearAnalysisState();
    setText("");
    setHeadText("");
    setTailText("");
  }

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

  async function sendFeedback(verdict: "confirm" | "correct" | "unknown") {
    if (!openSet || !openSet.predictions[0]) return;
    if (verdict === "correct" && !correction.trim()) {
      setFeedbackStatus("Choose or enter the corrected relation first.");
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
          verdict === "correct" ? matchedCorrection?.relation_id ?? null : null,
        corrected_relation_name: verdict === "correct" ? correction.trim() : null,
        consent_to_store_text: feedbackConsent,
      });
      if (verdict === "correct" && !matchedCorrection) {
        setFeedbackStatus(
          response.stored_text
            ? "New relation suggestion saved with consented text for ontology review. It is not added to the current model automatically."
            : "New relation suggestion saved for ontology review without raw text. It is not added to the current model automatically.",
        );
      } else if (verdict === "correct") {
        setFeedbackStatus(
          response.stored_text
            ? "Correction saved with consented text. This known relation can enter the reviewed retraining workflow."
            : "Correction saved without raw text. The label is recorded, but the example cannot be reused as training text.",
        );
      } else {
        setFeedbackStatus(
          response.stored_text
            ? "Feedback saved with training-data consent."
            : "Feedback saved without storing raw text.",
        );
      }
    } catch (reason) {
      setFeedbackStatus(
        reason instanceof Error ? reason.message : "Feedback could not be saved.",
      );
    }
  }

  const top = prediction?.predictions[0];

  return (
    <div className="relationPage">
      <div className="relationWorkbench">
        <section className="surface relationInput">
          <div className="surfaceHeader">
            <div>
              <span className="sectionKicker">PAIR ANALYSIS</span>
              <h2>Define a relation pair</h2>
            </div>
            <Network size={19} />
          </div>

          <label className="field">
            <span>Sentence</span>
            <textarea
              value={text}
              onChange={(event) => {
                clearAnalysisState();
                setText(event.target.value);
              }}
              rows={7}
            />
          </label>

          <div className="fieldGrid">
            <label className="field">
              <span>Head entity</span>
              <input
                value={headText}
                onChange={(event) => {
                  clearAnalysisState();
                  setHeadText(event.target.value);
                }}
              />
            </label>
            <label className="field">
              <span>Tail entity</span>
              <input
                value={tailText}
                onChange={(event) => {
                  clearAnalysisState();
                  setTailText(event.target.value);
                }}
              />
            </label>
          </div>

          <div className="profilePicker">
            <div className="profilePickerHead">
              <span>Operating profile</span>
              <strong>{PROFILE_METRICS[profile].label}</strong>
            </div>
            <div className="segmentedControl three">
              {(model?.open_set_profiles?.length
                ? model.open_set_profiles
                : PROFILE_ORDER
              ).map((item) => (
                <button
                  key={item}
                  className={profile === item ? "active" : ""}
                  onClick={() => {
                    clearAnalysisState();
                    setProfile(item);
                  }}
                  type="button"
                >
                  {PROFILE_METRICS[item].shortLabel}
                </button>
              ))}
            </div>
            <small>
              Threshold {confidence(PROFILE_METRICS[profile].threshold)} ·{" "}
              {PROFILE_METRICS[profile].description}
            </small>
          </div>

          {error && <div className="notice noticeError">{error}</div>}

          <button
            className="button buttonPrimary buttonFull"
            onClick={() => void runExtraction()}
            disabled={busy || !text.trim() || !headText.trim() || !tailText.trim()}
          >
            <Sparkles size={17} />
            {busy ? "Analyzing…" : "Run relation analysis"}
          </button>
        </section>

        <section className="surface relationResult">
          <div className="surfaceHeader">
            <div>
              <span className="sectionKicker">DECISION</span>
              <h2>Model output</h2>
            </div>
            {prediction ? (
              <button
                className="textButton resetAction"
                type="button"
                onClick={startNewAnalysis}
              >
                <RotateCcw size={14} /> New analysis
              </button>
            ) : (
              <Activity size={19} />
            )}
          </div>

          {!prediction || !top ? (
            <div className="emptyState compactEmpty">
              <div className="emptyIcon"><Sparkles size={22} /></div>
              <strong>Ready for a relation pair</strong>
              <p>Run an analysis to see the ranked relation and abstention decision.</p>
            </div>
          ) : (
            <div className="predictionStack">
              <div className="predictionHero">
                <div>
                  <span>Top relation</span>
                  <h3>{top.relation_name}</h3>
                  <small>{top.relation_id}</small>
                </div>
                <strong>{confidence(top.confidence)}</strong>
              </div>

              {openSet && (
                <div className={openSet.accepted ? "decisionPill known" : "decisionPill unknown"}>
                  <ShieldCheck size={16} />
                  <strong>{openSet.accepted ? "Known relation" : "Unknown relation"}</strong>
                  <span>{confidence(openSet.threshold)} threshold</span>
                </div>
              )}

              <div className="rankingList">
                {prediction.predictions.map((item, index) => (
                  <div className="rankingItem" key={item.relation_id}>
                    <span className="rankIndex">{index + 1}</span>
                    <div>
                      <div className="rankingTopline">
                        <strong>{item.relation_name}</strong>
                        <span>{confidence(item.confidence)}</span>
                      </div>
                      <div className="progressTrack">
                        <i style={{ width: confidence(item.confidence) }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="microMeta">
                <span>{prediction.model_id}</span>
                <span>{prediction.latency_ms.toFixed(0)} ms</span>
              </div>
            </div>
          )}
        </section>
      </div>

      {prediction && (
        <div className="relationLowerGrid">
          <section className="surface explanationCard">
            <div className="surfaceHeader">
              <div>
                <span className="sectionKicker">EVIDENCE</span>
                <h2>Word influence</h2>
              </div>
            </div>
            {!explanation ? (
              <div className="skeletonRows">
                <span /><span /><span />
              </div>
            ) : (
              <>
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
                      title={`${(item.confidence_drop * 100).toFixed(2)} pp`}
                    >
                      {item.word}
                    </span>
                  ))}
                </div>
                {topSupporting.length > 0 && (
                  <div className="cueRow">
                    {topSupporting.map((item) => (
                      <span key={`${item.start}-cue`}>
                        {item.word} <b>+{(item.confidence_drop * 100).toFixed(1)} pp</b>
                      </span>
                    ))}
                  </div>
                )}
              </>
            )}
          </section>

          {openSet && top && (
            <section className="surface feedbackCard">
              <div className="surfaceHeader">
                <div>
                  <span className="sectionKicker">FEEDBACK</span>
                  <h2>Review this prediction</h2>
                </div>
                <MessageSquareText size={19} />
              </div>
              <div className="feedbackQuickActions">
                <button className="button buttonSecondary" onClick={() => void sendFeedback("confirm")}>
                  <CheckCircle2 size={15} /> Confirm
                </button>
                <button className="button buttonSecondary" onClick={() => void sendFeedback("unknown")}>
                  Mark unknown
                </button>
              </div>
              <p className="feedbackHelper">
                Correct the relation label here — the head and tail entities stay unchanged.
              </p>
              <div className="correctionInline">
                <input
                  list="relation-catalog"
                  placeholder="Correct relation or suggest a new one"
                  value={correction}
                  onChange={(event) => {
                    setCorrection(event.target.value);
                    setFeedbackStatus(null);
                  }}
                />
                <datalist id="relation-catalog">
                  {relations.map((item) => (
                    <option key={item.relation_id} value={item.relation_name} />
                  ))}
                </datalist>
                <button className="button buttonSecondary" onClick={() => void sendFeedback("correct")}>
                  Save
                </button>
              </div>
              {correction.trim() && (
                <div
                  className={
                    matchedCorrection
                      ? "correctionHint known"
                      : "correctionHint suggestion"
                  }
                >
                  {matchedCorrection
                    ? `Known catalog relation · ${matchedCorrection.relation_id} · eligible for reviewed retraining when text consent is enabled.`
                    : "New relation suggestion · recorded for ontology review, but not trainable by the current 56-class model."}
                </div>
              )}
              <label className="privacyToggle">
                <input
                  type="checkbox"
                  checked={feedbackConsent}
                  onChange={(event) => setFeedbackConsent(event.target.checked)}
                />
                <span>Allow this text to enter the reviewed training-data workflow</span>
              </label>
              {feedbackStatus && <div className="notice">{feedbackStatus}</div>}
            </section>
          )}
        </div>
      )}
    </div>
  );
}
