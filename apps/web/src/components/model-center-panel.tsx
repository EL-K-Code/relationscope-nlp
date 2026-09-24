"use client";

import { Search, ShieldCheck } from "lucide-react";
import { useMemo, useState } from "react";

import { RiskCoverageChart } from "@/components/risk-coverage-chart";
import {
  EVALUATION_PROTOCOL,
  FINAL_METRICS,
  PROFILE_METRICS,
} from "@/lib/model-report";
import type {
  ModelInfo,
  OpenSetProfile,
  RelationCatalogItem,
} from "@/lib/types";

const PROFILE_ORDER: OpenSetProfile[] = [
  "high_coverage",
  "balanced",
  "high_safety",
];

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function ModelCenterPanel({
  model,
  relations,
}: {
  model: ModelInfo | null;
  relations: RelationCatalogItem[];
}) {
  const [query, setQuery] = useState("");
  const filteredRelations = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return relations;
    return relations.filter(
      (item) =>
        item.relation_id.toLowerCase().includes(normalized) ||
        item.relation_name.toLowerCase().includes(normalized),
    );
  }, [query, relations]);

  return (
    <div className="modelPage">
      <section className="modelHero surface">
        <div>
          <span className="sectionKicker">CHAMPION MODEL</span>
          <h2>{model?.model_id ?? "ModernBERT relation classifier"}</h2>
          <p>
            Entity-aware relation classification with calibrated open-set abstention.
          </p>
        </div>
        <div className="modelHeroMeta">
          <div><span>Relations</span><strong>{model?.num_relations ?? 56}</strong></div>
          <div><span>Pooling</span><strong>{model?.pooling_mode ?? "entity_pair"}</strong></div>
          <div><span>Input</span><strong>{model?.input_view ?? "marked"}</strong></div>
          <div><span>Runtime</span><strong>{model?.device ?? "cpu"}</strong></div>
        </div>
      </section>

      <section className="metricGrid modelMetrics">
        <article className="metricTile">
          <span>Known accuracy</span>
          <strong>{percent(FINAL_METRICS.knownAccuracy)}</strong>
          <small>{EVALUATION_PROTOCOL.testKnown.toLocaleString()} test examples</small>
        </article>
        <article className="metricTile">
          <span>Known macro-F1</span>
          <strong>{percent(FINAL_METRICS.knownMacroF1)}</strong>
          <small>{EVALUATION_PROTOCOL.knownRelations} known relations</small>
        </article>
        <article className="metricTile">
          <span>OOD AUROC</span>
          <strong>{percent(FINAL_METRICS.oodAuRoc)}</strong>
          <small>{EVALUATION_PROTOCOL.finalUnknownRelations} unseen relations</small>
        </article>
        <article className="metricTile">
          <span>Unknown F1</span>
          <strong>{percent(FINAL_METRICS.unknownF1)}</strong>
          <small>Balanced profile</small>
        </article>
      </section>

      <div className="modelGrid">
        <section className="surface chartSurface">
          <div className="surfaceHeader">
            <div>
              <span className="sectionKicker">SELECTIVE RISK</span>
              <h2>Risk / coverage</h2>
            </div>
          </div>
          <RiskCoverageChart />
        </section>

        <section className="surface profileSurface">
          <div className="surfaceHeader">
            <div>
              <span className="sectionKicker">OPERATING PROFILES</span>
              <h2>Choose the abstention policy</h2>
            </div>
            <ShieldCheck size={18} />
          </div>
          <div className="profileCards">
            {PROFILE_ORDER.map((profile) => {
              const item = PROFILE_METRICS[profile];
              return (
                <article key={profile}>
                  <div className="profileTitle">
                    <strong>{item.label}</strong>
                    <span>{percent(item.threshold)}</span>
                  </div>
                  <dl>
                    <div><dt>Known coverage</dt><dd>{percent(item.knownAcceptance)}</dd></div>
                    <div><dt>Accepted accuracy</dt><dd>{percent(item.acceptedKnownAccuracy)}</dd></div>
                    <div><dt>Unknown rejection</dt><dd>{percent(item.unknownRejection)}</dd></div>
                  </dl>
                </article>
              );
            })}
          </div>
        </section>
      </div>

      <section className="surface catalogSurface">
        <div className="catalogHeader">
          <div>
            <span className="sectionKicker">RELATION CATALOG</span>
            <h2>{relations.length} known relation classes</h2>
          </div>
          <label className="searchField">
            <Search size={16} />
            <input
              placeholder="Search relation or ID"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
        </div>
        <div className="relationCatalogGrid">
          {filteredRelations.map((item) => (
            <article key={item.relation_id}>
              <span>{item.relation_id}</span>
              <strong>{item.relation_name}</strong>
            </article>
          ))}
        </div>
        {!filteredRelations.length && (
          <div className="emptyInline">No relation matches “{query}”.</div>
        )}
      </section>
    </div>
  );
}
