"use client";

import {
  FileText,
  Network,
  ScanSearch,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { useState } from "react";

import { DocumentKnowledgeGraph } from "@/components/document-knowledge-graph";
import { analyzeDocument, uploadDocument } from "@/lib/api";
import type {
  DocumentAnalyzeResponse,
  OpenSetProfile,
} from "@/lib/types";

const SAMPLE = `Marie Curie was born in Warsaw. Marie Curie studied at the University of Paris. Pierre Curie worked with Marie Curie in Paris.`;

function confidence(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function DocumentIntelligencePanel() {
  const [text, setText] = useState(SAMPLE);
  const [profile, setProfile] = useState<OpenSetProfile>("balanced");
  const [includeUnknown, setIncludeUnknown] = useState(false);
  const [consent, setConsent] = useState(false);
  const [result, setResult] = useState<DocumentAnalyzeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runTextAnalysis() {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await analyzeDocument({
          text,
          title: "Pasted document",
          profile,
          include_unknown: includeUnknown,
          max_candidate_pairs: 256,
          consent_to_store_review_data: consent,
        }),
      );
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Document analysis failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function handleFile(file: File) {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await uploadDocument(file, {
          profile,
          includeUnknown,
          maxCandidatePairs: 256,
          consentToStoreReviewData: consent,
        }),
      );
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Document upload failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="documentWorkspace">
      <section className="panel documentInputPanel">
        <div className="panelHeader">
          <div>
            <span className="step">V3 DOCUMENT INTELLIGENCE</span>
            <h2>Start from raw text, not entity spans</h2>
          </div>
          <ScanSearch size={20} />
        </div>

        <p className="muted">
          V3 discovers entities automatically, generates same-sentence entity
          pairs, runs calibrated relation extraction, and assembles a document
          knowledge graph.
        </p>

        <label>
          Document text
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={10}
          />
        </label>

        <div className="documentControls">
          <label>
            Operating profile
            <select
              value={profile}
              onChange={(event) =>
                setProfile(event.target.value as OpenSetProfile)
              }
            >
              <option value="high_coverage">High coverage</option>
              <option value="balanced">Balanced</option>
              <option value="high_safety">High safety</option>
            </select>
          </label>

          <label className="inlineCheck">
            <input
              type="checkbox"
              checked={includeUnknown}
              onChange={(event) => setIncludeUnknown(event.target.checked)}
            />
            Include UNKNOWN edges
          </label>

          <label className="inlineCheck">
            <input
              type="checkbox"
              checked={consent}
              onChange={(event) => setConsent(event.target.checked)}
            />
            Allow uncertain sentence pairs into the review queue
          </label>
        </div>

        <div className="documentActions">
          <button
            className="primaryButton"
            disabled={busy || !text.trim()}
            onClick={() => void runTextAnalysis()}
          >
            <Network size={17} />
            {busy ? "Analysing document…" : "Build knowledge graph"}
          </button>

          <label className="secondaryButton uploadDocumentButton">
            <Upload size={16} />
            Upload PDF / TXT
            <input
              type="file"
              accept=".pdf,.txt,text/plain,application/pdf"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleFile(file);
              }}
            />
          </label>
        </div>

        {error && <div className="errorBanner">{error}</div>}
      </section>

      {!result ? (
        <section className="panel documentEmpty">
          <FileText size={44} />
          <strong>No document graph yet</strong>
          <p>
            Paste text or upload a PDF/TXT to let RelationScope discover entities
            and relations automatically.
          </p>
        </section>
      ) : (
        <>
          <section className="panel documentSummary">
            <div className="panelHeader">
              <div>
                <span className="step">DOCUMENT RESULT</span>
                <h2>{result.title ?? "Analyzed document"}</h2>
              </div>
              <ShieldCheck size={20} />
            </div>
            <div className="documentMetricGrid">
              <div>
                <small>ENTITIES</small>
                <strong>{result.entities.length}</strong>
              </div>
              <div>
                <small>RELATIONS</small>
                <strong>{result.relations.length}</strong>
              </div>
              <div>
                <small>CANDIDATE PAIRS</small>
                <strong>{result.candidate_pairs}</strong>
              </div>
              <div>
                <small>REVIEW CANDIDATES</small>
                <strong>{result.review_candidates}</strong>
              </div>
              <div>
                <small>QUEUED</small>
                <strong>{result.queued_for_review}</strong>
              </div>
              <div>
                <small>LATENCY</small>
                <strong>{result.latency_ms.toFixed(0)} ms</strong>
              </div>
            </div>
            <div className="documentMeta">
              <span>RE: {result.model_id}</span>
              <span>NER: {result.ner_model_id}</span>
              <span>ID: {result.document_id}</span>
            </div>
          </section>

          <DocumentKnowledgeGraph result={result} />

          <section className="panel documentTables">
            <div className="documentTableBlock">
              <div className="panelHeader">
                <div>
                  <span className="step">ENTITIES</span>
                  <h2>Canonical entity inventory</h2>
                </div>
              </div>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th>ENTITY</th>
                      <th>TYPE</th>
                      <th>MENTIONS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.entities.slice(0, 100).map((entity) => (
                      <tr key={entity.entity_id}>
                        <td>{entity.text}</td>
                        <td>{entity.label}</td>
                        <td>{entity.mentions.length}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="documentTableBlock">
              <div className="panelHeader">
                <div>
                  <span className="step">RELATIONS</span>
                  <h2>Document relation evidence</h2>
                </div>
              </div>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      <th>HEAD</th>
                      <th>RELATION</th>
                      <th>TAIL</th>
                      <th>CONF.</th>
                      <th>DECISION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.relations.slice(0, 150).map((edge) => (
                      <tr key={edge.edge_id}>
                        <td>{edge.head_text}</td>
                        <td>{edge.relation.relation_name}</td>
                        <td>{edge.tail_text}</td>
                        <td>{confidence(edge.relation.confidence)}</td>
                        <td>{edge.decision}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
