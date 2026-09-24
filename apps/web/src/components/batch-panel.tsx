"use client";

import { Download, FileUp, Layers3, UploadCloud } from "lucide-react";
import { useState } from "react";

import { KnowledgeGraph } from "@/components/knowledge-graph";
import { createBatchJob, getBatchJob } from "@/lib/api";
import { exportBatchResults, parseBatchCsv } from "@/lib/csv";
import type {
  BatchInput,
  BatchJobSummary,
  BatchResult,
  OpenSetProfile,
} from "@/lib/types";

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

export function BatchPanel() {
  const [profile, setProfile] = useState<OpenSetProfile>("balanced");
  const [rows, setRows] = useState<BatchInput[]>([]);
  const [results, setResults] = useState<BatchResult[]>([]);
  const [job, setJob] = useState<BatchJobSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleCsv(file: File) {
    setError(null);
    setResults([]);
    setJob(null);
    try {
      const parsed = parseBatchCsv(await file.text());
      if (parsed.length > 2000) {
        throw new Error("A batch can contain at most 2,000 rows.");
      }
      setRows(parsed);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Invalid CSV file.");
    }
  }

  async function runBatch() {
    if (!rows.length) return;
    setBusy(true);
    setError(null);
    setResults([]);

    try {
      const created = await createBatchJob(rows, profile, 1);
      setJob(created);
      let current = await getBatchJob(created.job_id);
      setJob(current);

      for (let attempt = 0; attempt < 120; attempt += 1) {
        if (current.status === "succeeded" || current.status === "failed") break;
        await wait(2000);
        current = await getBatchJob(created.job_id);
        setJob(current);
      }

      if (current.status === "failed") {
        throw new Error(current.error ?? "The batch job failed.");
      }
      if (current.status !== "succeeded") {
        throw new Error("The job is still running. Keep the job ID and check again later.");
      }

      setResults(
        rows.map((row, index) => ({
          ...row,
          relation: current.results?.[index]?.relation,
        })),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Batch processing failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="batchPage">
      <section className="surface batchComposer">
        <div className="surfaceHeader">
          <div>
            <span className="sectionKicker">ASYNC BATCH</span>
            <h2>Process relation pairs at scale</h2>
          </div>
          <Layers3 size={19} />
        </div>

        <div className="dropzone">
          <input
            id="batch-csv"
            type="file"
            accept=".csv,text/csv"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void handleCsv(file);
            }}
          />
          <label htmlFor="batch-csv">
            <span className="dropzoneIcon"><UploadCloud size={24} /></span>
            <strong>{rows.length ? `${rows.length} rows ready` : "Drop or choose a CSV"}</strong>
            <p>Entity-pair jobs · up to 2,000 rows</p>
            <span className="dropzoneAction">Choose file</span>
          </label>
        </div>

        <div className="batchToolbar">
          <div className="segmentedControl three compactSegments">
            {(["high_coverage", "balanced", "high_safety"] as OpenSetProfile[]).map((item) => (
              <button
                key={item}
                type="button"
                className={profile === item ? "active" : ""}
                onClick={() => setProfile(item)}
              >
                {item.replace("_", " ")}
              </button>
            ))}
          </div>
          <a href="/sample-relations.csv" download className="textButton">
            <FileUp size={14} /> Sample CSV
          </a>
          <button
            className="button buttonPrimary"
            disabled={!rows.length || busy}
            onClick={() => void runBatch()}
          >
            {busy ? "Processing…" : "Queue batch"}
          </button>
        </div>

        {error && <div className="notice noticeError">{error}</div>}
      </section>

      {job && (
        <section className="surface jobCard">
          <div className="jobProgressTop">
            <div>
              <span className="sectionKicker">JOB</span>
              <strong>{job.job_id}</strong>
            </div>
            <span className={`statusBadge ${job.status}`}>{job.status}</span>
          </div>
          <div className="jobProgressBar">
            <i style={{ width: `${job.total ? (job.processed / job.total) * 100 : 0}%` }} />
          </div>
          <div className="jobMeta">
            <span>{job.processed}/{job.total} processed</span>
            <span>{job.model_id ?? "Waiting for worker"}</span>
          </div>
        </section>
      )}

      {results.length > 0 && (
        <>
          <section className="surface batchResults">
            <div className="surfaceHeader">
              <div>
                <span className="sectionKicker">RESULTS</span>
                <h2>{results.length} predictions</h2>
              </div>
              <button
                className="button buttonSecondary"
                onClick={() =>
                  downloadText(
                    "relationscope-results.csv",
                    exportBatchResults(results),
                  )
                }
              >
                <Download size={15} /> Export CSV
              </button>
            </div>
            <div className="dataTableWrap">
              <table className="dataTable">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>HEAD</th>
                    <th>RELATION</th>
                    <th>TAIL</th>
                    <th>CONFIDENCE</th>
                  </tr>
                </thead>
                <tbody>
                  {results.slice(0, 100).map((row) => (
                    <tr key={row.id}>
                      <td className="monoCell">{row.id}</td>
                      <td>{row.head.text}</td>
                      <td>{row.relation?.relation_name ?? "—"}</td>
                      <td>{row.tail.text}</td>
                      <td>{row.relation ? confidence(row.relation.confidence) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <KnowledgeGraph results={results} />
        </>
      )}
    </div>
  );
}
