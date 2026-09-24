"use client";

import {
  Activity,
  BarChart3,
  BrainCircuit,
  ChevronRight,
  FileSearch,
  GitBranch,
  Home,
  Layers3,
  Menu,
  Network,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { BatchPanel } from "@/components/batch-panel";
import { ContinuousLearningPanel } from "@/components/continuous-learning-panel";
import { DashboardPanel } from "@/components/dashboard-panel";
import { DocumentIntelligencePanel } from "@/components/document-intelligence-panel";
import { ModelCenterPanel } from "@/components/model-center-panel";
import { MonitoringPanel } from "@/components/monitoring-panel";
import { RelationAnalyzerPanel } from "@/components/relation-analyzer-panel";
import { getModelInfo, getRelationCatalog } from "@/lib/api";
import type { ModelInfo, RelationCatalogItem } from "@/lib/types";

type Tab =
  | "dashboard"
  | "analyze"
  | "relation"
  | "batch"
  | "learning"
  | "monitoring"
  | "model";

const NAVIGATION: {
  id: Tab;
  label: string;
  description: string;
  icon: typeof Home;
}[] = [
  { id: "dashboard", label: "Overview", description: "Workspace home", icon: Home },
  { id: "analyze", label: "Analyze", description: "Text & PDF", icon: FileSearch },
  { id: "relation", label: "Relation test", description: "Pair inspection", icon: Network },
  { id: "batch", label: "Batch", description: "CSV jobs", icon: Layers3 },
  { id: "learning", label: "Learning", description: "Review & drift", icon: GitBranch },
  { id: "monitoring", label: "Monitoring", description: "Production telemetry", icon: Activity },
  { id: "model", label: "Model", description: "Metrics & catalog", icon: BrainCircuit },
];

const PAGE_COPY: Record<Tab, { eyebrow: string; title: string; description: string }> = {
  dashboard: {
    eyebrow: "WORKSPACE",
    title: "Overview",
    description: "Your relation intelligence workspace.",
  },
  analyze: {
    eyebrow: "DOCUMENT INTELLIGENCE",
    title: "Analyze",
    description: "Extract entities and relations from raw text or PDF.",
  },
  relation: {
    eyebrow: "PAIR INSPECTION",
    title: "Relation test",
    description: "Inspect one entity pair with confidence and evidence.",
  },
  batch: {
    eyebrow: "DURABLE JOBS",
    title: "Batch processing",
    description: "Run relation extraction across CSV datasets.",
  },
  learning: {
    eyebrow: "CONTINUOUS LEARNING",
    title: "Learning",
    description: "Review model signals, drift and promotion state.",
  },
  monitoring: {
    eyebrow: "PRODUCTION",
    title: "Monitoring",
    description: "Track runtime behavior without storing raw text.",
  },
  model: {
    eyebrow: "MODEL CENTER",
    title: "Model",
    description: "Understand the champion model and its relation catalog.",
  },
};

export function RelationScopeStudio() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [relations, setRelations] = useState<RelationCatalogItem[]>([]);
  const [modelError, setModelError] = useState<string | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [navQuery, setNavQuery] = useState("");

  useEffect(() => {
    getModelInfo()
      .then((info) => setModel(info))
      .catch((reason: unknown) =>
        setModelError(reason instanceof Error ? reason.message : "Model unavailable"),
      );

    getRelationCatalog()
      .then((catalog) => setRelations(catalog.items))
      .catch(() => setRelations([]));
  }, []);

  const current = PAGE_COPY[tab];
  const filteredNavigation = useMemo(() => {
    const query = navQuery.trim().toLowerCase();
    if (!query) return NAVIGATION;
    return NAVIGATION.filter(
      (item) =>
        item.label.toLowerCase().includes(query) ||
        item.description.toLowerCase().includes(query),
    );
  }, [navQuery]);

  function navigate(next: Tab) {
    setTab(next);
    setMobileNavOpen(false);
    setSearchOpen(false);
    setNavQuery("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <main className="appShell">
      <aside className={mobileNavOpen ? "appSidebar mobileOpen" : "appSidebar"}>
        <div className="brandRow">
          <button
            className="brandButton"
            onClick={() => navigate("dashboard")}
            aria-label="RelationScope home"
          >
            <span className="brandGlyph">
              <Network size={17} />
            </span>
            <span>
              <strong>RelationScope</strong>
              <small>Intelligence platform</small>
            </span>
          </button>
          <button
            className="mobileClose"
            onClick={() => setMobileNavOpen(false)}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        <nav className="productNav" aria-label="Primary navigation">
          <span className="navSectionLabel">Workspace</span>
          {NAVIGATION.slice(0, 4).map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={tab === item.id ? "navItem active" : "navItem"}
                onClick={() => navigate(item.id)}
              >
                <Icon size={17} />
                <span>
                  <strong>{item.label}</strong>
                  <small>{item.description}</small>
                </span>
                <ChevronRight size={14} className="navChevron" />
              </button>
            );
          })}

          <span className="navSectionLabel secondary">System</span>
          {NAVIGATION.slice(4).map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={tab === item.id ? "navItem active" : "navItem"}
                onClick={() => navigate(item.id)}
              >
                <Icon size={17} />
                <span>
                  <strong>{item.label}</strong>
                  <small>{item.description}</small>
                </span>
                <ChevronRight size={14} className="navChevron" />
              </button>
            );
          })}
        </nav>

        <div className="sidebarFooter">
          <div className="modelMiniCard">
            <span className={model ? "liveDot live" : "liveDot"} />
            <div>
              <small>{model ? "Production" : "Connecting"}</small>
              <strong>{model?.model_id ?? "Model service"}</strong>
              <span>
                {model
                  ? model.num_relations + " relations · " + model.device
                  : modelError ?? "Waiting for API"}
              </span>
            </div>
          </div>
        </div>
      </aside>

      {mobileNavOpen && (
        <button
          className="mobileBackdrop"
          aria-label="Close navigation"
          onClick={() => setMobileNavOpen(false)}
        />
      )}

      <section className="appMain">
        <header className="appTopbar">
          <div className="topbarLeft">
            <button
              className="mobileMenu"
              onClick={() => setMobileNavOpen(true)}
              aria-label="Open navigation"
            >
              <Menu size={19} />
            </button>
            <div className="pageIdentity">
              <span>{current.eyebrow}</span>
              <strong>{current.title}</strong>
            </div>
          </div>

          <div className="topbarActions">
            <div className={searchOpen ? "commandSearch open" : "commandSearch"}>
              <Search size={15} />
              <input
                aria-label="Navigate workspace"
                placeholder="Jump to…"
                value={navQuery}
                onFocus={() => setSearchOpen(true)}
                onChange={(event) => {
                  setNavQuery(event.target.value);
                  setSearchOpen(true);
                }}
              />
              <kbd>⌘ K</kbd>
              {searchOpen && (
                <div className="commandMenu">
                  {filteredNavigation.map((item) => {
                    const Icon = item.icon;
                    return (
                      <button key={item.id} onClick={() => navigate(item.id)}>
                        <Icon size={16} />
                        <span>
                          <strong>{item.label}</strong>
                          <small>{item.description}</small>
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
            <button className="topbarPrimary" onClick={() => navigate("analyze")}>
              <Sparkles size={15} />
              New analysis
            </button>
            <span className="avatar">AL</span>
          </div>
        </header>

        <div className="pageHeader">
          <div>
            <span className="sectionKicker">{current.eyebrow}</span>
            <h1>{current.title}</h1>
            <p>{current.description}</p>
          </div>
          {tab === "monitoring" && (
            <div className="headerStatus">
              <span className={model ? "liveDot live" : "liveDot"} />
              {model ? "All core services reachable" : "Checking services"}
            </div>
          )}
        </div>

        <div className="pageCanvas">
          {tab === "dashboard" && (
            <DashboardPanel model={model} onNavigate={navigate} />
          )}
          {tab === "analyze" && <DocumentIntelligencePanel />}
          {tab === "relation" && (
            <RelationAnalyzerPanel model={model} relations={relations} />
          )}
          {tab === "batch" && <BatchPanel />}
          {tab === "learning" && <ContinuousLearningPanel />}
          {tab === "monitoring" && <MonitoringPanel model={model} />}
          {tab === "model" && (
            <ModelCenterPanel model={model} relations={relations} />
          )}
        </div>

        <footer className="productFooter">
          <span>RelationScope V3</span>
          <span>Private API · Cloud Run</span>
          <span>Open-set relation intelligence</span>
          <span className="footerMark">
            <BarChart3 size={13} /> Production workspace
          </span>
        </footer>
      </section>
    </main>
  );
}
