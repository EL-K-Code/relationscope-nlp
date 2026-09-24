"use client";

import type { DocumentAnalyzeResponse } from "@/lib/types";

type Node = {
  id: string;
  label: string;
  entityType: string;
  x: number;
  y: number;
};

function short(text: string, max = 20) {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

export function DocumentKnowledgeGraph({
  result,
}: {
  result: DocumentAnalyzeResponse;
}) {
  const relations = result.relations.slice(0, 32);
  if (!relations.length) return null;

  const referenced = new Set(
    relations.flatMap((edge) => [
      edge.head_entity_id,
      edge.tail_entity_id,
    ]),
  );
  const entities = result.entities
    .filter((entity) => referenced.has(entity.entity_id))
    .slice(0, 24);

  const width = 960;
  const height = 520;
  const centerX = width / 2;
  const centerY = height / 2;
  const radiusX = 360;
  const radiusY = 190;

  const nodes: Node[] = entities.map((entity, index) => {
    const angle =
      (Math.PI * 2 * index) / Math.max(entities.length, 1) - Math.PI / 2;
    return {
      id: entity.entity_id,
      label: entity.text,
      entityType: entity.label,
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY,
    };
  });

  const byId = new Map(nodes.map((node) => [node.id, node]));
  const visibleRelations = relations.filter(
    (edge) => byId.has(edge.head_entity_id) && byId.has(edge.tail_entity_id),
  );

  return (
    <section className="knowledgeGraph documentGraph">
      <div className="graphHeader">
        <div>
          <small>DOCUMENT KNOWLEDGE GRAPH</small>
          <strong>Canonical entities and extracted relations</strong>
        </div>
        <span>
          {nodes.length} entities · {visibleRelations.length} edges
        </span>
      </div>

      <div className="graphCanvas">
        <svg viewBox={`0 0 ${width} ${height}`} role="img">
          <defs>
            <marker
              id="doc-arrow"
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

          {visibleRelations.map((edge) => {
            const source = byId.get(edge.head_entity_id)!;
            const target = byId.get(edge.tail_entity_id)!;
            const midX = (source.x + target.x) / 2;
            const midY = (source.y + target.y) / 2;

            return (
              <g key={edge.edge_id}>
                <line
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  className={
                    edge.accepted
                      ? "graphEdge"
                      : "graphEdge documentUnknownEdge"
                  }
                  markerEnd="url(#doc-arrow)"
                />
                <text x={midX} y={midY - 7} className="edgeLabel">
                  {short(edge.relation.relation_name, 24)}
                </text>
              </g>
            );
          })}

          {nodes.map((node) => (
            <g key={node.id}>
              <circle cx={node.x} cy={node.y} r="38" className="graphNode" />
              <text x={node.x} y={node.y - 1} className="nodeLabel">
                {short(node.label)}
              </text>
              <text x={node.x} y={node.y + 12} className="nodeTypeLabel">
                {node.entityType}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </section>
  );
}
