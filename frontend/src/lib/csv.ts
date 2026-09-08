/*
 * Minimal client-side CSV parsing — just enough to preview an uploaded file
 * and populate the target-column dropdown before it is sent to the backend.
 * The backend (pandas) remains the source of truth for validation and the
 * real DatasetProfile.
 */

/** Parse one CSV line, handling simple double-quote escaping. */
function parseLine(line: string): string[] {
  const out: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          quoted = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      out.push(field);
      field = "";
    } else {
      field += ch;
    }
  }
  out.push(field);
  return out.map((f) => f.trim());
}

/** Read a File/Blob as text. Uses FileReader for the widest runtime support. */
export function readFileText(file: Blob): Promise<string> {
  if (typeof (file as Blob & { text?: unknown }).text === "function") {
    return (file as Blob).text();
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read file"));
    reader.readAsText(file);
  });
}

export interface ParsedCsv {
  headers: string[];
  rows: string[][];
  rowCount: number;
}

export function parseCsv(text: string): ParsedCsv {
  const lines = text
    .split(/\r\n|\n|\r/)
    .filter((l) => l.trim().length > 0);
  if (lines.length === 0) {
    return { headers: [], rows: [], rowCount: 0 };
  }
  const headers = parseLine(lines[0]);
  const rows = lines.slice(1).map(parseLine);
  return { headers, rows, rowCount: rows.length };
}

/**
 * Mirror of backend `_infer_task_type`: a non-numeric target, or a numeric
 * target with ≤20 distinct integer-like values, is classification. This is a
 * preview hint only — `task_type_source` on the returned profile records what
 * the backend actually decided.
 */
export function guessTaskType(values: string[]): "classification" | "regression" {
  const clean = values.filter((v) => v !== "" && v != null);
  if (clean.length === 0) return "regression";
  const nums = clean.map(Number);
  const allNumeric = nums.every((n) => !Number.isNaN(n));
  if (!allNumeric) return "classification";
  const distinct = new Set(clean);
  if (distinct.size <= 20 && nums.every((n) => Number.isInteger(n))) {
    return "classification";
  }
  return "regression";
}

export function columnValues(parsed: ParsedCsv, column: string): string[] {
  const idx = parsed.headers.indexOf(column);
  if (idx === -1) return [];
  return parsed.rows.map((r) => r[idx] ?? "");
}
