import { useEffect, useState } from "react";

import {
  useCreateSession,
  useDatasets,
  useDeleteSession,
  useSessionList,
} from "../hooks/queries";
import { formatTimestamp } from "../lib/format";
import { apiErrorMessage } from "../services/api";
import { Badge, Card, Empty, ErrorBox, QueryState, SectionEyebrow } from "./ui";

const MAX_QUESTION = 500;

/**
 * SessionManager — "Start investigation" form + the list of past
 * investigations. The dataset picker is populated from GET /api/datasets and
 * defaults to whatever was just uploaded (`preselectedDatasetId`); earlier
 * uploads are simply other options in the same dropdown, not a prominent
 * table.
 */
export function SessionManager({
  onOpenSession,
  preselectedDatasetId,
}: {
  onOpenSession: (sessionId: string) => void;
  preselectedDatasetId?: string;
}) {
  const sessions = useSessionList();
  const datasets = useDatasets();
  const create = useCreateSession();
  const del = useDeleteSession();

  function deleteInvestigation(id: string, question: string) {
    if (
      !window.confirm(
        `Delete this investigation and all its experiments?\n\n"${question}"\n\nThis cannot be undone.`,
      )
    ) {
      return;
    }
    del.mutate(id);
  }

  const [question, setQuestion] = useState("");
  const [datasetId, setDatasetId] = useState("");

  useEffect(() => {
    if (preselectedDatasetId) setDatasetId(preselectedDatasetId);
  }, [preselectedDatasetId]);

  const tooLong = question.length > MAX_QUESTION;
  const canSubmit =
    question.trim().length > 0 && !tooLong && !!datasetId && !create.isPending;

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    create.mutate(
      { research_question: question.trim(), dataset_id: datasetId },
      {
        onSuccess: (res) => {
          setQuestion("");
          onOpenSession(res.session_id);
        },
      },
    );
  }

  const hasDatasets = (datasets.data ?? []).length > 0;

  return (
    <div className="section">
      <SectionEyebrow step={2}>Start an investigation</SectionEyebrow>
      <Card
        title="Ask a question about your dataset"
        hint="One investigation pairs a research question with a dataset. The agent then plans, runs, and analyses experiments until it can answer."
      >
        <form onSubmit={submit}>
          <div className="field">
            <label htmlFor="rq">Research question</label>
            <textarea
              id="rq"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Does dropout improve performance on this dataset?"
            />
            <div
              className="field__help"
              style={{ color: tooLong ? "var(--danger)" : undefined }}
            >
              {question.length}/{MAX_QUESTION}
            </div>
          </div>

          <div className="field">
            <label htmlFor="rq-dataset">Dataset</label>
            <select
              id="rq-dataset"
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              disabled={!hasDatasets}
            >
              <option value="">
                {hasDatasets ? "select a dataset…" : "upload a dataset first"}
              </option>
              {(datasets.data ?? []).map((d) => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.original_filename} · {d.task_type} ·{" "}
                  {d.n_rows.toLocaleString()} rows
                  {d.dataset_id === preselectedDatasetId ? "  (just uploaded)" : ""}
                </option>
              ))}
            </select>
            {!hasDatasets && (
              <div className="field__help">
                Upload a CSV in step 1 to enable this.
              </div>
            )}
          </div>

          {create.error != null && (
            <div className="field">
              <ErrorBox>{apiErrorMessage(create.error)}</ErrorBox>
            </div>
          )}

          <button
            type="submit"
            className="btn btn--primary"
            disabled={!canSubmit}
          >
            {create.isPending ? "Starting…" : "Start investigation"}
          </button>
        </form>
      </Card>

      <div style={{ marginTop: 20 }}>
        <Card title="Your investigations" plain>
          <QueryState isLoading={sessions.isLoading} error={sessions.error}>
            {del.error != null && (
              <div style={{ marginBottom: 12 }}>
                <ErrorBox>{apiErrorMessage(del.error)}</ErrorBox>
              </div>
            )}
            {sessions.data && sessions.data.length > 0 ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Research question</th>
                      <th>Status</th>
                      <th className="num">Cycles</th>
                      <th className="num">Experiments</th>
                      <th>Created</th>
                      <th aria-label="actions" />
                    </tr>
                  </thead>
                  <tbody>
                    {sessions.data.map((s) => (
                      <tr
                        key={s.session_id}
                        className="row-clickable investigation-list-row"
                        onClick={() => onOpenSession(s.session_id)}
                      >
                        <td>{s.research_question}</td>
                        <td>
                          {s.status === "concluded" ? (
                            <Badge variant="ok">concluded</Badge>
                          ) : s.run_phase === "running" ? (
                            <Badge variant="warn">running</Badge>
                          ) : s.run_phase === "failed" ? (
                            <Badge variant="danger">failed</Badge>
                          ) : (
                            <Badge variant="muted">active</Badge>
                          )}
                        </td>
                        <td className="num">{s.cycle_count}</td>
                        <td className="num">{s.experiment_count}</td>
                        <td>{formatTimestamp(s.created_at)}</td>
                        <td>
                          <button
                            type="button"
                            className="icon-btn icon-btn--danger"
                            title="Delete this investigation"
                            aria-label={`Delete investigation: ${s.research_question}`}
                            disabled={del.isPending}
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteInvestigation(
                                s.session_id,
                                s.research_question,
                              );
                            }}
                          >
                            🗑
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty>No investigations yet — start one above.</Empty>
            )}
          </QueryState>
        </Card>
      </div>
    </div>
  );
}
