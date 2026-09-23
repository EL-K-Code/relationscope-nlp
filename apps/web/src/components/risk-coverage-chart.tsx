"use client";

import { RISK_COVERAGE } from "@/lib/model-report";

function pct(value: number) {
  return `${(value * 100).toFixed(0)}%`;
}

export function RiskCoverageChart() {
  const width = 520;
  const height = 220;
  const padLeft = 44;
  const padRight = 20;
  const padTop = 18;
  const padBottom = 34;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  const minCoverage = 0.5;
  const maxCoverage = 1.0;
  const maxRisk = 0.12;

  const points = RISK_COVERAGE.map((item) => {
    const x =
      padLeft +
      ((item.coverage - minCoverage) / (maxCoverage - minCoverage)) * plotWidth;
    const y = padTop + (1 - item.risk / maxRisk) * plotHeight;
    return { ...item, x, y };
  });

  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

  return (
    <div className="riskChart">
      <div className="chartHeader">
        <div>
          <small>SELECTIVE CLASSIFICATION</small>
          <strong>Risk–coverage curve</strong>
        </div>
        <span>known test split</span>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Risk versus coverage on the known relation test set"
      >
        {[0, 0.03, 0.06, 0.09, 0.12].map((risk) => {
          const y = padTop + (1 - risk / maxRisk) * plotHeight;
          return (
            <g key={risk}>
              <line
                className="chartGrid"
                x1={padLeft}
                x2={width - padRight}
                y1={y}
                y2={y}
              />
              <text className="chartAxisLabel" x={8} y={y + 4}>
                {pct(risk)}
              </text>
            </g>
          );
        })}

        {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((coverage) => {
          const x =
            padLeft +
            ((coverage - minCoverage) / (maxCoverage - minCoverage)) *
              plotWidth;
          return (
            <text
              className="chartAxisLabel"
              key={coverage}
              x={x}
              y={height - 10}
              textAnchor="middle"
            >
              {pct(coverage)}
            </text>
          );
        })}

        <path className="riskLine" d={path} fill="none" />

        {points.map((point) => (
          <g key={point.coverage}>
            <circle
              className="riskPoint"
              cx={point.x}
              cy={point.y}
              r={4.5}
            />
            <title>
              {pct(point.coverage)} coverage · {pct(point.risk)} risk
            </title>
          </g>
        ))}

        <text
          className="chartAxisTitle"
          x={padLeft + plotWidth / 2}
          y={height - 1}
          textAnchor="middle"
        >
          coverage
        </text>
        <text
          className="chartAxisTitle"
          x={12}
          y={padTop + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${padTop + plotHeight / 2})`}
        >
          error risk
        </text>
      </svg>
    </div>
  );
}
