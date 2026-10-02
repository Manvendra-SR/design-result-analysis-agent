import { metricDelta, metricValue, pct } from "../lib/format";
import type { Report } from "../types/api";
import { CiBar, ciScale, VerdictLabel } from "./ResultsPanel";
import { Card, InterpretationTag, StatisticalTag } from "./ui";

/**
 * The answer: one comparison - the best validation level vs the reference -
 * evaluated once on the held-out test split, which the loop never looked at.
 */
export function ReportPanel({ report }: { report: Report }) {
  const { metric, comparison } = report;
  return (
    <Card title="Result (held-out test set)" badge={<StatisticalTag />}>
      <p className="headline">{report.headline}</p>

      <div className="stat-tiles" style={{ marginTop: 16 }}>
        <div className="stat-tile">
          <div className="stat-tile__label">{report.challenger}</div>
          <div className="stat-tile__value">{metricValue(metric, report.challenger_score)}</div>
        </div>
        <div className="stat-tile">
          <div className="stat-tile__label">{report.reference} (reference)</div>
          <div className="stat-tile__value">{metricValue(metric, report.reference_score)}</div>
        </div>
        <div className="stat-tile">
          <div className="stat-tile__label">Difference · 95% CI</div>
          <div className="stat-tile__value">{metricDelta(metric, comparison.diff)}</div>
          <div className="card__hint" style={{ margin: "4px 0 0" }}>
            {metricDelta(metric, comparison.ci_low)} to {metricDelta(metric, comparison.ci_high)} ·{" "}
            <VerdictLabel verdict={comparison.verdict} />
          </div>
          <div style={{ marginTop: 8 }}>
            <CiBar comparison={comparison} scale={ciScale([comparison])} />
          </div>
        </div>
      </div>

      <p className="card__hint" style={{ marginTop: 14 }}>
        {report.n_test_rows.toLocaleString()} test rows · {report.rounds_run} round
        {report.rounds_run === 1 ? "" : "s"} ·{" "}
        {report.stopped_by === "budget" ? "stopped by the round budget" : "concluded by the agent"}
        {report.majority_rate !== null && ` · majority-class floor ${pct(report.majority_rate)}`}
      </p>

      {report.interpretation && (
        <div style={{ marginTop: 16 }}>
          <InterpretationTag />
          <p className="llm-text">{report.interpretation}</p>
        </div>
      )}
    </Card>
  );
}
