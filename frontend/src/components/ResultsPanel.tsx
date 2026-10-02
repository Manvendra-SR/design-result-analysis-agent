import { metricDelta, metricValue, pct } from "../lib/format";
import type { Analysis, Comparison, Verdict } from "../types/api";
import { Badge, Card, Empty, StatisticalTag } from "./ui";

/** The 95% CI drawn against zero, on a scale shared by every row. */
export function CiBar({ comparison, scale }: { comparison: Comparison; scale: number }) {
  const at = (v: number) => `${((v + scale) / (2 * scale)) * 100}%`;
  return (
    <div className="ci-bar" aria-hidden="true">
      <div className="ci-bar__zero" style={{ left: "50%" }} />
      <div
        className={`ci-bar__range ci-bar__range--${comparison.verdict}`}
        style={{ left: at(comparison.ci_low), right: `calc(100% - ${at(comparison.ci_high)})` }}
      />
      <div className="ci-bar__point" style={{ left: at(comparison.diff) }} />
    </div>
  );
}

export function VerdictLabel({ verdict }: { verdict: Verdict }) {
  return <span className={`verdict--${verdict}`}>{verdict}</span>;
}

export function ciScale(comparisons: Comparison[]): number {
  return Math.max(1e-9, ...comparisons.flatMap((c) => [Math.abs(c.ci_low), Math.abs(c.ci_high)])) * 1.1;
}

/**
 * Validation-split results, recomputed by the backend on every poll. These
 * drive the agent's decisions and are exploratory; the confirmatory number is
 * the single test-split comparison in the report.
 */
export function ResultsPanel({ analysis }: { analysis: Analysis | null }) {
  if (!analysis) {
    return (
      <Card title="Results (validation)" badge={<StatisticalTag />}>
        <Empty>Results appear as soon as the first runs finish.</Empty>
      </Card>
    );
  }
  const { metric } = analysis;
  const byLabel = new Map(analysis.comparisons.map((c) => [c.label, c]));
  const scale = ciScale(analysis.comparisons);

  return (
    <Card
      title="Results (validation)"
      badge={<StatisticalTag />}
      hint={`Each level vs the reference on the same ${analysis.n_rows.toLocaleString()} validation rows: the difference in ${metric} (${analysis.higher_is_better ? "higher" : "lower"} is better) with a 95% paired-bootstrap confidence interval. "Seed spread" is how much a level varies between training runs - stability, not uncertainty.`}
    >
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Level</th>
              <th className="num">Runs</th>
              <th className="num">{metric}</th>
              <th className="num">Seed spread</th>
              <th className="num">Δ vs reference</th>
              <th className="num">95% CI</th>
              <th aria-label="confidence interval chart" />
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {analysis.conditions.map((c) => {
              const cmp = byLabel.get(c.label);
              return (
                <tr key={c.label}>
                  <td className="mono">
                    {c.label} {c.is_reference && <Badge variant="gradient">reference</Badge>}
                  </td>
                  <td className="num">
                    {c.n_ok}
                    {c.n_failed > 0 && <span className="verdict--worse"> (+{c.n_failed} failed)</span>}
                  </td>
                  <td className="num">{metricValue(metric, c.mean)}</td>
                  <td className="num">{c.seed_std === null ? "—" : `±${metricValue(metric, c.seed_std)}`}</td>
                  <td className="num">{cmp ? metricDelta(metric, cmp.diff) : "—"}</td>
                  <td className="num">
                    {cmp ? `${metricDelta(metric, cmp.ci_low)} to ${metricDelta(metric, cmp.ci_high)}` : "—"}
                  </td>
                  <td>{cmp && <CiBar comparison={cmp} scale={scale} />}</td>
                  <td>{cmp ? <VerdictLabel verdict={cmp.verdict} /> : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {analysis.majority_rate !== null && (
        <p className="card__hint" style={{ marginTop: 12 }}>
          Sanity floor: always predicting the most common class scores {pct(analysis.majority_rate)}.
        </p>
      )}
    </Card>
  );
}
