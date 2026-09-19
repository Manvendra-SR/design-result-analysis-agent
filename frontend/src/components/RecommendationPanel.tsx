import { describeConfig } from "../lib/format";
import type { Recommendation, TerminationReason } from "../types/api";
import { Badge, Card, Empty, InterpretationTag } from "./ui";

/**
 * RecommendationPanel (task 7.9 / 7.13). `explanation` and `evidence_summary`
 * are LLM prose — italic, under an "LLM Interpretation" tag. There is no
 * per-cycle approval button: the backend already ran every cycle. This shows
 * the FINAL recommendation (GET /sessions/{id}/recommendation).
 *
 * The recommendation is stored exactly as the agent produced it, so on a run
 * stopped by the safety cap its action is still `run_more_experiments`. That
 * combination is rendered as "stopped at the limit", never as a conclusion —
 * previously the backend rewrote the action to "conclude" and left prose
 * arguing for more experiments underneath it.
 */
export function RecommendationPanel({
  recommendation,
  terminationReason = null,
  maxCycles,
}: {
  recommendation: Recommendation | null;
  terminationReason?: TerminationReason | null;
  maxCycles?: number;
}) {
  const stoppedAtLimit = terminationReason === "cycle_limit";
  return (
    <Card
      title="Recommendation"
      badge={<InterpretationTag />}
      right={
        recommendation && (
          <Badge
            variant={
              stoppedAtLimit
                ? "danger"
                : recommendation.action === "conclude"
                  ? "ok"
                  : "warn"
            }
          >
            {stoppedAtLimit
              ? "Stopped at limit"
              : recommendation.action === "conclude"
                ? "Concluded"
                : "More experiments"}
          </Badge>
        )
      }
    >
      {!recommendation ? (
        <Empty>
          No recommendation yet. It appears once the first run completes.
        </Empty>
      ) : (
        <div className="stack" style={{ gap: 16 }}>
          {stoppedAtLimit && (
            <div className="error-box" role="status">
              <strong>This is not a settled answer.</strong> The investigation
              hit its {maxCycles ?? ""}-cycle safety limit while the agent still
              wanted more experiments. What follows is the agent&apos;s last
              reasoning — the case for continuing, not a conclusion. Ask a
              follow-up question below to keep investigating.
            </div>
          )}

          {!stoppedAtLimit && recommendation.action === "conclude" && (
            <div className="concluded-banner">
              <span aria-hidden="true">✓</span>
              <span>
                The Recommender judged the evidence sufficient to answer the
                research question.
              </span>
            </div>
          )}

          <div>
            <h3 className="card__hint" style={{ margin: "0 0 4px" }}>
              Explanation
            </h3>
            <p className="llm-text">{recommendation.explanation}</p>
          </div>

          <div>
            <h3 className="card__hint" style={{ margin: "0 0 4px" }}>
              Evidence summary
            </h3>
            <p className="llm-text">{recommendation.evidence_summary}</p>
          </div>

          {recommendation.recommended_experiments.length > 0 && (
            <div>
              <h3 className="card__hint" style={{ margin: "0 0 4px" }}>
                {stoppedAtLimit
                  ? "Experiments it wanted next (never run — the limit stopped the loop)"
                  : "Experiments it queued next"}
              </h3>
              <ul className="list-plain mono">
                {recommendation.recommended_experiments.map((cfg, i) => (
                  <li key={i}>{describeConfig(cfg)}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
