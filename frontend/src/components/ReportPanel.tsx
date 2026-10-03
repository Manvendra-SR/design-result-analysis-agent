import { confidenceLabel, metricDelta, metricValue, pct } from "../lib/format";
import type { Comparison, Metric, Report } from "../types/api";
import { CiBar, ciScale, VerdictLabel } from "./ResultsPanel";
import { Card, InterpretationTag, StatisticalTag } from "./ui";

const STOPPED_BY: Record<Report["stopped_by"], string> = {
  agent: "concluded by the agent",
  budget: "stopped by the round budget",
  settled: "stopped because every contender was one family",
};

function ComparisonTile({ title, comparison, metric, level }: {
  title: string;
  comparison: Comparison;
  metric: Metric;
  level: string;
}) {
  return (
    <div className="stat-tile">
      <div className="stat-tile__label">{title} · {level} CI</div>
      <div className="stat-tile__value">{metricDelta(metric, comparison.diff)}</div>
      <div className="card__hint" style={{ margin: "4px 0 0" }}>
        {metricDelta(metric, comparison.ci_low)} to {metricDelta(metric, comparison.ci_high)} ·{" "}
        <VerdictLabel verdict={comparison.verdict} />
      </div>
      <div style={{ marginTop: 8 }}>
        <CiBar comparison={comparison} scale={ciScale([comparison])} />
      </div>
    </div>
  );
}

/**
 * The answer: the winner chosen on validation, compared on the held-out test
 * split - which the loop never looked at - with the reference and, in
 * selection mode, with the best candidate of another family.
 */
export function ReportPanel({ report }: { report: Report }) {
  const { metric } = report;
  const level = confidenceLabel(report.confidence);
  const effort = Object.entries(report.effort)
    .map(([family, n]) => `${family} ${n}`)
    .join(", ");
  return (
    <Card title="Result (held-out test set)" badge={<StatisticalTag />}>
      <p className="headline">{report.headline}</p>

      <div className="stat-tiles" style={{ marginTop: 16 }}>
        <div className="stat-tile">
          <div className="stat-tile__label">{report.winner} (challenger)</div>
          <div className="stat-tile__value">{metricValue(metric, report.winner_score.value)}</div>
        </div>
        <div className="stat-tile">
          <div className="stat-tile__label">{report.reference} (reference)</div>
          <div className="stat-tile__value">{metricValue(metric, report.reference_score.value)}</div>
        </div>
        <ComparisonTile title="vs reference" comparison={report.primary} metric={metric} level={level} />
        {report.secondary && (
          <ComparisonTile title={`vs ${report.runner_up}`} comparison={report.secondary} metric={metric} level={level} />
        )}
      </div>

      <p className="card__hint" style={{ marginTop: 14 }}>
        {report.n_test_rows.toLocaleString()} test rows · {report.rounds_run} round
        {report.rounds_run === 1 ? "" : "s"} · {STOPPED_BY[report.stopped_by]} · {report.candidates_tried} candidates
        tried ({effort}) · validation → test drop for the winner {metricDelta(metric, report.val_to_test_drop)}
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
