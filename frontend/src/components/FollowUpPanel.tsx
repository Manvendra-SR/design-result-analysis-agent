import { useState } from "react";

import { useCreateSession } from "../hooks/queries";
import { apiErrorMessage } from "../services/api";
import type { SessionDetail } from "../types/api";
import { Badge, Card, ErrorBox } from "./ui";

const MAX_QUESTION = 500;

/**
 * FollowUpPanel — asking a new research question once an investigation is
 * finished.
 *
 * A concluded investigation is immutable: its evidence answers the question it
 * was created with, and its research question is write-once. Continuing
 * therefore starts a NEW investigation on the same dataset, linked to this one
 * for provenance. That linkage is deliberately provenance-only — the new
 * investigation begins with no experiments, anomalies or cycle history of its
 * own, so evidence gathered for one question can never be mistaken for
 * evidence about another.
 */
export function FollowUpPanel({
  session,
  stoppedAtLimit,
  onOpenSession,
}: {
  session: SessionDetail;
  stoppedAtLimit: boolean;
  onOpenSession: (sessionId: string) => void;
}) {
  const create = useCreateSession();
  const [question, setQuestion] = useState("");

  const tooLong = question.length > MAX_QUESTION;
  const canSubmit = question.trim().length > 0 && !tooLong && !create.isPending;

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    create.mutate(
      {
        research_question: question.trim(),
        dataset_id: session.dataset_id,
        parent_session_id: session.session_id,
      },
      {
        onSuccess: (res) => {
          setQuestion("");
          onOpenSession(res.session_id);
        },
      },
    );
  }

  return (
    <Card
      title="Ask a follow-up question"
      badge={<Badge variant="gradient">Next investigation</Badge>}
      hint={
        stoppedAtLimit
          ? "This investigation stopped at its cycle limit without settling the question. A follow-up starts a fresh investigation on the same dataset — narrow the question, or ask about a different factor."
          : "This investigation is finished. A follow-up starts a fresh investigation on the same dataset, so its evidence stays separate from the answer above."
      }
    >
      <form onSubmit={submit}>
        <div className="field">
          <label htmlFor="followup-rq">New research question</label>
          <textarea
            id="followup-rq"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Does a wider hidden layer beat the linear baseline?"
          />
          <div
            className="field__help"
            style={{ color: tooLong ? "var(--danger)" : undefined }}
          >
            {question.length}/{MAX_QUESTION}
          </div>
        </div>

        <p className="card__hint" style={{ marginTop: 0 }}>
          Runs on the same dataset. The new investigation starts from scratch —
          it does not reuse this one&apos;s experiments, so the two questions
          are never answered from mixed evidence.
        </p>

        {create.error != null && (
          <div className="field">
            <ErrorBox>{apiErrorMessage(create.error)}</ErrorBox>
          </div>
        )}

        <button type="submit" className="btn btn--primary" disabled={!canSubmit}>
          {create.isPending ? "Starting…" : "Start follow-up investigation"}
        </button>
      </form>
    </Card>
  );
}
