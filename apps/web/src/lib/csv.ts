import type { BatchInput, BatchResult } from "@/lib/types";

const REQUIRED_COLUMNS = [
  "id",
  "text",
  "head_text",
  "head_start",
  "head_end",
  "tail_text",
  "tail_start",
  "tail_end",
] as const;

function splitCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current);
  return values;
}

export function parseBatchCsv(content: string): BatchInput[] {
  const lines = content
    .replace(/\r/g, "")
    .split("\n")
    .filter((line) => line.trim().length > 0);

  if (lines.length < 2) {
    throw new Error("CSV must contain a header and at least one data row.");
  }

  const header = splitCsvLine(lines[0]).map((value) => value.trim());
  for (const column of REQUIRED_COLUMNS) {
    if (!header.includes(column)) {
      throw new Error(`Missing required CSV column: ${column}`);
    }
  }

  const index = Object.fromEntries(
    header.map((name, position) => [name, position]),
  ) as Record<string, number>;

  return lines.slice(1).map((line, rowIndex) => {
    const values = splitCsvLine(line);
    const number = (name: string): number => {
      const value = Number(values[index[name]]);
      if (!Number.isInteger(value) || value < 0) {
        throw new Error(`Invalid ${name} on CSV row ${rowIndex + 2}`);
      }
      return value;
    };

    return {
      id: values[index.id] || String(rowIndex + 1),
      text: values[index.text],
      head: {
        text: values[index.head_text],
        start: number("head_start"),
        end: number("head_end"),
      },
      tail: {
        text: values[index.tail_text],
        start: number("tail_start"),
        end: number("tail_end"),
      },
    };
  });
}

function escapeCsv(value: string | number): string {
  const text = String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

export function exportBatchResults(results: BatchResult[]): string {
  const header = [
    "id",
    "text",
    "head",
    "relation_id",
    "relation_name",
    "tail",
    "confidence",
    "error",
  ];

  const rows = results.map((result) => [
    result.id,
    result.text,
    result.head.text,
    result.relation?.relation_id ?? "",
    result.relation?.relation_name ?? "",
    result.tail.text,
    result.relation?.confidence ?? "",
    result.error ?? "",
  ]);

  return [header, ...rows]
    .map((row) => row.map(escapeCsv).join(","))
    .join("\n");
}
