"use client";

import type { BatchResult } from "@/lib/types";

type Node = {
  id: string;
  label: string;
  x: number;
  y: number;
};

type Edge = {
  head: string;
  tail: string;
  label: string;
  confidence: number;
};

function short(text: string, max = 18) {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

export function KnowledgeGraph({ results }: { results: BatchResult[] }) {
  const triples = results
    .filter((row) => row.relation)
    .slice(0, 20)
    .map((row) => ({
      head: row.head.text,
      tail: row.tail.text,
      label: row.relation!.relation_name,
      confidence: row.relation!.confidence,
    }));

  const labels = Array.from(
    new Set(triples.flatMap((triple) => [triple.head, triple.tail])),
  ).slice(0, 18);

  const width = 900;
  const height = 430;
  const centerX = width / 2;
  const centerY = height / 2;
  const radiusX = 330;
  const radiusY = 150;

  const nodes: Node[] = labels.map((label, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(labels.length, 1) - Math.PI / 2;
    return {
      id: label,
      label,
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY,
    };
  });

  const byId = new Map(nodes.map((node) => [node.id, node]));
  const edges: Edge[] = triples.filter(
    (triple) => byId.has(triple.head) && byId.has(triple.tail),
  );

  if (!edges.length) return null;

  return (
    <section className="knowledgeGraph">
      <div className="graphHeader">
        <div>
          <small>KNOWLEDGE GRAPH</small>
          <strong>Extracted triples</strong>
        </div>
        <span>{edges.length} relations visualized</span>
      </div>

      <div className="graphCanvas">
        <svg viewBox={`0 0 ${width} ${height}`} role="img">
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" />
            </marker>
          </defs>

          {edges.map((edge, index) => {
            const source = byId.get(edge.head)!;
            const target = byId.get(edge.tail)!;
            const midX = (source.x + target.x) / 2;
            const midY = (source.y + target.y) / 2;

            return (
              <g key={`${edge.head}-${edge.tail}-${index}`}>
                <line
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  className="graphEdge"
                  markerEnd="url(#arrow)"
                />
                <text x={midX} y={midY - 6} className="edgeLabel">
                  {short(edge.label, 22)}
                </text>
              </g>
            );
          })}

          {nodes.map((node) => (
            <g key={node.id}>
              <circle cx={node.x} cy={node.y} r="34" className="graphNode" />
              <text x={node.x} y={node.y + 3} className="nodeLabel">
                {short(node.label)}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </section>
  );
}
