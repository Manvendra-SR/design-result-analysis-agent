import { useCallback, useMemo, useRef, useState } from "react";

import { useDatasets, useDeleteDataset, useIngestDataset } from "../hooks/queries";
import {
  columnValues,
  guessTaskType,
  parseCsv,
  readFileText,
  type ParsedCsv,
} from "../lib/csv";
import { apiErrorMessage, errorStatus } from "../services/api";
import type { DatasetProfile } from "../types/api";
import { Badge, Card, ErrorBox, SectionEyebrow, Spinner } from "./ui";

const PREVIEW_ROWS = 5;
const MIN_ROWS = 20; // matches backend ingest_csv._MIN_ROWS

/**
 * DatasetManager — the upload → preview → ingest flow.
 *
 * The backend's POST /api/datasets takes the CSV as text (no multipart), so
 * the browser reads the chosen file (FileReader) and posts its contents.
 * No backend change. Previously-uploaded datasets are not shown here; they
 * live in the "Start investigation" dataset picker where they are actually
 * needed.
 */
export function DatasetManager({
  onIngested,
}: {
  onIngested?: (profile: DatasetProfile) => void;
}) {
  const ingest = useIngestDataset();
  const datasets = useDatasets();
  const delDataset = useDeleteDataset();
  const inputRef = useRef<HTMLInputElement>(null);

  async function removeDataset(id: string, name: string) {
    if (!window.confirm(`Delete dataset "${name}"?`)) return;
    delDataset.reset();
    try {
      await delDataset.mutateAsync({ id });
    } catch (err) {
      if (errorStatus(err) === 409) {
        if (
          window.confirm(
            `${apiErrorMessage(err)}\n\nDelete the dataset AND those investigations?`,
          )
        ) {
          await delDataset.mutateAsync({ id, cascade: true }).catch(() => {});
        } else {
          delDataset.reset();
        }
      }
    }
  }

  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [parsed, setParsed] = useState<ParsedCsv | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [target, setTarget] = useState("");
  const [override, setOverride] = useState<"" | "classification" | "regression">("");
  const [ingested, setIngested] = useState<DatasetProfile | null>(null);

  const acceptFile = useCallback(async (f: File) => {
    setParseError(null);
    setIngested(null);
    ingest.reset();
    if (!f.name.toLowerCase().endsWith(".csv")) {
      setParseError("Please choose a .csv file.");
      return;
    }
    const content = await readFileText(f);
    const p = parseCsv(content);
    if (p.headers.length < 2) {
      setParseError("This file needs at least two columns (features + a target).");
      return;
    }
    if (p.rowCount < MIN_ROWS) {
      setParseError(
        `This file has ${p.rowCount} data row${p.rowCount === 1 ? "" : "s"}; ` +
          `the backend needs at least ${MIN_ROWS}.`,
      );
      return;
    }
    setFile(f);
    setText(content);
    setParsed(p);
    setTarget(p.headers[p.headers.length - 1]); // sensible default; user can change
  }, [ingest]);

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) void acceptFile(f);
  }

  function reset() {
    setFile(null);
    setText("");
    setParsed(null);
    setTarget("");
    setOverride("");
    setParseError(null);
    setIngested(null);
    ingest.reset();
  }

  const taskGuess = useMemo(() => {
    if (!parsed || !target) return null;
    return guessTaskType(columnValues(parsed, target));
  }, [parsed, target]);

  function submit() {
    if (!file || !target) return;
    ingest.mutate(
      {
        filename: file.name,
        csv_content: text,
        target_column: target,
        task_type_override: override || null,
      },
      {
        onSuccess: (profile) => {
          setIngested(profile);
          onIngested?.(profile);
        },
      },
    );
  }

  return (
    <div className="section">
      <SectionEyebrow step={1}>Upload dataset</SectionEyebrow>
      <Card
        title="Your CSV dataset"
        hint="A small tabular CSV. The backend validates it, drops rows with missing values, and profiles it."
      >
        {!parsed ? (
          <>
            <div
              className={dragging ? "upload upload--drag" : "upload"}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <div className="upload__icon" aria-hidden="true">
                ⬆
              </div>
              <div className="upload__title">Drop your CSV here</div>
              <div className="upload__sub">or</div>
              <button
                type="button"
                className="btn btn--primary"
                onClick={() => inputRef.current?.click()}
              >
                Browse files
              </button>
              <input
                ref={inputRef}
                type="file"
                accept=".csv,text/csv"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void acceptFile(f);
                  e.target.value = "";
                }}
              />
              <div className="field__help" style={{ marginTop: 12 }}>
                Accepted format: CSV
              </div>
            </div>
            {parseError && (
              <div style={{ marginTop: 12 }}>
                <ErrorBox>{parseError}</ErrorBox>
              </div>
            )}
          </>
        ) : (
          <div className="stack" style={{ gap: 16 }}>
            <div className="file-pill">
              <span aria-hidden="true">📄</span>
              <span className="file-pill__name">{file?.name}</span>
              <span className="file-pill__meta">
                {parsed.rowCount.toLocaleString()} rows · {parsed.headers.length}{" "}
                columns
              </span>
              <button
                type="button"
                className="link-btn"
                style={{ marginLeft: "auto" }}
                onClick={reset}
              >
                Choose a different file
              </button>
            </div>

            <div>
              <div className="field__help" style={{ marginBottom: 6 }}>
                Preview — first {Math.min(PREVIEW_ROWS, parsed.rowCount)} rows
              </div>
              <div className="table-wrap">
                <table className="preview-table">
                  <thead>
                    <tr>
                      {parsed.headers.map((h) => (
                        <th key={h}>
                          {h}
                          {h === target ? " ⓣ" : ""}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {parsed.rows.slice(0, PREVIEW_ROWS).map((r, i) => (
                      <tr key={i}>
                        {parsed.headers.map((_, j) => (
                          <td key={j}>{r[j] ?? ""}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {ingested ? (
              <div className="concluded-banner">
                <span aria-hidden="true">✓</span>
                <span>
                  Ingested as <strong>{ingested.original_filename}</strong> —{" "}
                  {ingested.task_type}
                  {ingested.task_type === "classification" && ingested.n_classes
                    ? ` (${ingested.n_classes} classes)`
                    : ""}
                  , {ingested.n_rows.toLocaleString()} usable rows,{" "}
                  {ingested.n_features} features. Now start an investigation
                  below.
                </span>
              </div>
            ) : (
              <>
                <div className="row">
                  <div className="field" style={{ flex: "1 1 220px", margin: 0 }}>
                    <label htmlFor="ds-target">Target column</label>
                    <select
                      id="ds-target"
                      value={target}
                      onChange={(e) => setTarget(e.target.value)}
                    >
                      {parsed.headers.map((h) => (
                        <option key={h} value={h}>
                          {h}
                        </option>
                      ))}
                    </select>
                    <div className="field__help">The column to predict.</div>
                  </div>
                  <div className="field" style={{ flex: "1 1 180px", margin: 0 }}>
                    <label>Task type</label>
                    <div
                      className="btn"
                      style={{
                        cursor: "default",
                        justifyContent: "flex-start",
                        width: "100%",
                      }}
                    >
                      <Badge variant="gradient">
                        {override || taskGuess || "—"}
                      </Badge>
                      <span
                        className="field__help"
                        style={{ margin: 0, marginLeft: 6 }}
                      >
                        {override ? "overridden" : "auto-detected"}
                      </span>
                    </div>
                  </div>
                </div>

                <details className="advanced-toggle">
                  <summary>Advanced</summary>
                  <div className="field" style={{ marginTop: 10 }}>
                    <label htmlFor="ds-override">Force task type</label>
                    <select
                      id="ds-override"
                      value={override}
                      onChange={(e) =>
                        setOverride(
                          e.target.value as "" | "classification" | "regression",
                        )
                      }
                    >
                      <option value="">
                        use auto-detection ({taskGuess ?? "—"})
                      </option>
                      <option value="classification">classification</option>
                      <option value="regression">regression</option>
                    </select>
                  </div>
                </details>

                {ingest.error != null && (
                  <ErrorBox>{apiErrorMessage(ingest.error)}</ErrorBox>
                )}

                <div>
                  <button
                    type="button"
                    className="btn btn--primary"
                    onClick={submit}
                    disabled={!target || ingest.isPending}
                  >
                    {ingest.isPending && <Spinner onPrimary />}
                    {ingest.isPending ? "Ingesting…" : "Ingest dataset"}
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </Card>

      {(datasets.data?.length ?? 0) > 0 && (
        <details className="advanced-toggle" style={{ marginTop: 12 }}>
          <summary>Manage uploaded datasets ({datasets.data!.length})</summary>
          <div className="card card--plain" style={{ marginTop: 10 }}>
            {delDataset.error != null && (
              <div style={{ marginBottom: 12 }}>
                <ErrorBox>{apiErrorMessage(delDataset.error)}</ErrorBox>
              </div>
            )}
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>File</th>
                    <th>Task</th>
                    <th className="num">Rows</th>
                    <th aria-label="actions" />
                  </tr>
                </thead>
                <tbody>
                  {datasets.data!.map((d) => (
                    <tr key={d.dataset_id}>
                      <td>{d.original_filename}</td>
                      <td>{d.task_type}</td>
                      <td className="num">{d.n_rows.toLocaleString()}</td>
                      <td>
                        <button
                          type="button"
                          className="icon-btn icon-btn--danger"
                          title="Delete this dataset"
                          aria-label={`Delete dataset: ${d.original_filename}`}
                          disabled={delDataset.isPending}
                          onClick={() =>
                            removeDataset(d.dataset_id, d.original_filename)
                          }
                        >
                          🗑
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="field__help" style={{ marginTop: 8 }}>
              A dataset in use by an investigation can't be deleted until you
              confirm removing those investigations too.
            </p>
          </div>
        </details>
      )}
    </div>
  );
}
