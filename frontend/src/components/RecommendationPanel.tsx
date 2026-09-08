import { describeConfig } from "../lib/format";
import type { Recommendation } from "../types/api";
import { Badge, Card, Empty, InterpretationTag } from "./ui";

/**
 * RecommendationPanel (task 7.9 / 7.13). `explanation` and `evidence_summary`
 * are LLM prose — italic, under an "LLM Interpretation" tag. There is no
 * per-cycle approval button: the backend already ran every cycle. This shows
 * the FINAL recommendation (GET /sessions/{id}/recommendation).
 */
export function RecommendationPanel({
  recommendation,
}: {
  recommendation: Recommendation | null;
}) {
  return (
    <Card
      title="Recommendation"
      badge={<InterpretationTag />}
      right={
        recommendation && (
          <Badge
            variant={
              recommendation.action === "conclude" ? "ok" : "warn"
            }
          >
            {recommendation.action === "conclude"
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
          {recommendation.action === "conclude" && (
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
                Experiments it queued next
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
