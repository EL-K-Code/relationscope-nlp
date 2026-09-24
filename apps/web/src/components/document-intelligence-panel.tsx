"use client";

import {
  CheckCircle2,
  FileText,
  Network,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { useMemo, useState } from "react";

import { DocumentKnowledgeGraph } from "@/components/document-knowledge-graph";
import { analyzeDocument, uploadDocument } from "@/lib/api";
import type {
  DocumentAnalyzeResponse,
  DocumentRelation,
  OpenSetProfile,
} from "@/lib/types";

const SAMPLE =
  "Albert Einstein worked at Princeton University. Marie Curie studied at the University of Paris. OpenAI is based in San Francisco.";

type InputMode = "text" | "file";

function confidence(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function DocumentIntelligencePanel() {
  const [mode, setMode] = useState<InputMode>("text");
  const [text, setText] = useState(SAMPLE);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [profile, setProfile] = useState<OpenSetProfile>("balanced");
  const [includeUnknown, setIncludeUnknown] = useState(true);
  const [consent, setConsent] = useState(false);
  const [result, setResult] = useState<DocumentAnalyzeResponse | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedEdge = useMemo<DocumentRelation | null>(() => {
    if (!result?.relations.length) return null;
    return (
      result.relations.find((edge) => edge.edge_id === selectedEdgeId) ??
      result.relations[0]
    );
  }, [result, selectedEdgeId]);

  async function runAnalysis() {
    setBusy(true);
    setError(null);
    setSelectedEdgeId(null);
    try {
      const nextResult =
        mode === "file"
          ? selectedFile
            ? await uploadDocument(selectedFile, {
                profile,
                includeUnknown,
                maxCandidatePairs: 256,
                consentToStoreReviewData: consent,
              })
            : null
          : await analyzeDocument({
              text,
              title: "Pasted document",
              profile,
              include_unknown: includeUnknown,
              max_candidate_pairs: 256,
              consent_to_store_review_data: consent,
            });

      if (!nextResult) {
        throw new Error("Choose a PDF or TXT file first.");
      }
      setResult(nextResult);
      setSelectedEdgeId(nextResult.relations[0]?.edge_id ?? null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Document analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  const canRun = mode === "text" ? Boolean(text.trim()) : Boolean(selectedFile);

  return (
    <div className="documentPage">
      <section className="surface documentComposer">
        <div className="documentComposerTop">
          <div>
            <span className="sectionKicker">DOCUMENT ANALYZER</span>
            <h2>Extract a relation graph</h2>
          </div>
          <div className="segmentedControl">
            <button
              type="button"
              className={mode === "text" ? "active" : ""}
              onClick={() => setMode("text")}
            >
              Paste text
            </button>
            <button
              type="button"
              className={mode === "file" ? "active" : ""}
              onClick={() => setMode("file")}
            >
              Upload file
            </button>
          </div>
        </div>

        {mode === "text" ? (
          <label className="field documentTextField">
            <span>Document text</span>
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={8}
              placeholder="Paste text to analyze…"
            />
          </label>
        ) : (
          <div className="dropzone documentDropzone">
            <input
              id="document-file"
              type="file"
              accept=".pdf,.txt,text/plain,application/pdf"
              onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            />
            <label htmlFor="document-file">
              <span className="dropzoneIcon"><UploadCloud size={25} /></span>
              <strong>{selectedFile?.name ?? "Choose a PDF or TXT"}</strong>
              <p>
                {selectedFile
                  ? `${(selectedFile.size / 1024).toFixed(0)} KB ready for analysis`
                  : "Up to 10 MiB · text-based PDFs work best"}
              </p>
              <span className="dropzoneAction">Browse files</span>
            </label>
          </div>
        )}

        <div className="documentToolbar">
          <div className="toolbarGroup">
            <span className="toolbarLabel">Profile</span>
            <div className="segmentedControl three compactSegments">
              {(["high_coverage", "balanced", "high_safety"] as OpenSetProfile[]).map(
                (item) => (
                  <button
                    key={item}
                    type="button"
                    className={profile === item ? "active" : ""}
                    onClick={() => setProfile(item)}
                  >
                    {item === "high_coverage"
                      ? "Coverage"
                      : item === "high_safety"
                        ? "Safety"
                        : "Balanced"}
                  </button>
                ),
              )}
            </div>
          </div>

          <label className="compactToggle">
            <input
              type="checkbox"
              checked={includeUnknown}
              onChange={(event) => setIncludeUnknown(event.target.checked)}
            />
            <span>Show unknown relations</span>
          </label>

          <label className="compactToggle">
            <input
              type="checkbox"
              checked={consent}
              onChange={(event) => setConsent(event.target.checked)}
            />
            <span>Allow review queue</span>
          </label>

          <button
            className="button buttonPrimary analyzeButton"
            disabled={busy || !canRun}
            onClick={() => void runAnalysis()}
          >
            <Sparkles size={17} />
            {busy ? "Analyzing…" : "Analyze"}
          </button>
        </div>

        {error && <div className="notice noticeError">{error}</div>}
      </section>

      {!result ? (
        <section className="analysisEmpty">
          <div className="analysisEmptyGraphic">
            <div className="graphOrb orbOne" />
            <div className="graphOrb orbTwo" />
            <div className="graphOrb orbThree" />
            <Network size={28} />
          </div>
          <strong>Your knowledge graph will appear here.</strong>
          <p>Upload a document or paste text to detect entities and relation candidates.</p>
        </section>
      ) : (
        <>
          <section className="analysisSummary">
            <div className="summaryIdentity">
              <span className="analysisStatus"><CheckCircle2 size={14} /> Analysis complete</span>
              <strong>{result.title ?? "Analyzed document"}</strong>
              <small>{result.document_id}</small>
            </div>
            <div className="summaryMetrics">
              <div><span>Entities</span><strong>{result.entities.length}</strong></div>
              <div><span>Relations</span><strong>{result.relations.length}</strong></div>
              <div><span>Pairs</span><strong>{result.processed_pairs}</strong></div>
              <div><span>Review</span><strong>{result.review_candidates}</strong></div>
              <div><span>Latency</span><strong>{result.latency_ms.toFixed(0)} ms</strong></div>
            </div>
          </section>

          <section className="resultWorkbench">
            <div className="resultColumn entityColumn">
              <div className="columnHeader">
                <span>Entities</span>
                <small>{result.entities.length}</small>
              </div>
              <div className="entityList">
                {result.entities.map((entity) => (
                  <div className="entityListItem" key={entity.entity_id}>
                    <span className="entityAvatar">{entity.text.slice(0, 1).toUpperCase()}</span>
                    <div>
                      <strong>{entity.text}</strong>
                      <span>{entity.label} · {entity.mentions.length} mention{entity.mentions.length === 1 ? "" : "s"}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="resultColumn relationColumn">
              <div className="columnHeader">
                <span>Relations</span>
                <small>{result.relations.length}</small>
              </div>
              <div className="relationList">
                {result.relations.length ? (
                  result.relations.map((edge) => (
                    <button
                      type="button"
                      key={edge.edge_id}
                      className={`relationListItem ${selectedEdge?.edge_id === edge.edge_id ? "selected" : ""}`}
                      onClick={() => setSelectedEdgeId(edge.edge_id)}
                    >
                      <div className="relationLine">
                        <strong>{edge.head_text}</strong>
                        <span>→</span>
                        <strong>{edge.tail_text}</strong>
                      </div>
                      <div className="relationListMeta">
                        <span>{edge.relation.relation_name}</span>
                        <em>{confidence(edge.relation.confidence)}</em>
                      </div>
                    </button>
                  ))
                ) : (
                  <div className="emptyInline">No relation passed the selected output filter.</div>
                )}
              </div>
            </div>

            <aside className="resultColumn detailColumn">
              <div className="columnHeader">
                <span>Relation detail</span>
                {selectedEdge && (
                  <span className={selectedEdge.accepted ? "tinyStatus known" : "tinyStatus unknown"}>
                    {selectedEdge.accepted ? "KNOWN" : "UNKNOWN"}
                  </span>
                )}
              </div>

              {selectedEdge ? (
                <div className="relationDetail">
                  <div className="relationTriple">
                    <span>{selectedEdge.head_text}</span>
                    <div>
                      <i />
                      <strong>{selectedEdge.relation.relation_name}</strong>
                    </div>
                    <span>{selectedEdge.tail_text}</span>
                  </div>

                  <div className="detailConfidence">
                    <div>
                      <span>Confidence</span>
                      <strong>{confidence(selectedEdge.relation.confidence)}</strong>
                    </div>
                    <div className="confidenceTrack">
                      <i style={{ width: confidence(selectedEdge.relation.confidence) }} />
                      <b style={{ left: confidence(selectedEdge.threshold) }} />
                    </div>
                    <small>threshold {confidence(selectedEdge.threshold)}</small>
                  </div>

                  <div className="evidenceBox">
                    <span>Evidence</span>
                    <p>{selectedEdge.evidence_text}</p>
                  </div>

                  <div className="detailMetaGrid">
                    <div><span>Relation ID</span><strong>{selectedEdge.relation.relation_id}</strong></div>
                    <div><span>Profile</span><strong>{selectedEdge.profile}</strong></div>
                    <div><span>Sentence</span><strong>#{selectedEdge.sentence_index + 1}</strong></div>
                    <div><span>Decision</span><strong>{selectedEdge.decision.replace("_", " ")}</strong></div>
                  </div>
                </div>
              ) : (
                <div className="emptyInline">Select a relation to inspect evidence.</div>
              )}
            </aside>
          </section>

          <details className="graphDisclosure">
            <summary>
              <span><Network size={16} /> Knowledge graph</span>
              <small>{result.entities.length} nodes · {result.relations.length} edges</small>
            </summary>
            <DocumentKnowledgeGraph result={result} />
          </details>

          <section className="analysisMetaBar">
            <span><FileText size={14} /> RE {result.model_id}</span>
            <span><ShieldCheck size={14} /> NER {result.ner_model_id}</span>
            <span>Queued for review {result.queued_for_review}</span>
          </section>
        </>
      )}
    </div>
  );
}
