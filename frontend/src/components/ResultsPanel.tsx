import { metricDelta, metricValue, pct } from "../lib/format";
import type { Analysis, Comparison, Verdict } from "../types/api";
import { Badge, Card, Empty, StatisticalTag } from "./ui";

/** The CI drawn against zero, on a scale shared by every row. */
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
 * drive the agent's decisions and are exploratory; the confirmatory numbers
 * are the test-split comparisons in the report.
 *
 * Effect mode shows each level vs the reference. Selection mode shows each
 * candidate vs the current leader (contenders are those not clearly worse),
 * and a refined candidate also vs its parent.
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
  const selection = analysis.mode === "selection";
  const anchor = selection ? "leader" : "reference";
  const main = new Map(analysis.comparisons.filter((c) => c.anchor === anchor).map((c) => [c.a, c]));
  const parent = new Map(analysis.comparisons.filter((c) => c.anchor === "parent").map((c) => [c.a, c]));
  const scale = ciScale([...main.values()]);

  return (
    <Card
      title="Results (validation)"
      badge={<StatisticalTag />}
      hint={`Each candidate vs the ${selection ? "current leader" : "reference"} on the same ${analysis.n_rows.toLocaleString()} validation rows: the difference in ${metric} (${analysis.higher_is_better ? "higher" : "lower"} is better) with a 95% paired-bootstrap confidence interval. "Seed spread" is how much a candidate varies between training runs - stability, not uncertainty. "Train gap" is how much better it scores on the training rows (large = overfitting).`}
    >
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Candidate</th>
              <th className="num">Runs</th>
              <th className="num">{metric}</th>
              <th className="num">Seed spread</th>
              <th className="num">Train gap</th>
              <th className="num">Δ vs {anchor}</th>
              <th className="num">95% CI</th>
              <th aria-label="confidence interval chart" />
              <th>Verdict</th>
              {selection && <th className="num">Δ vs parent</th>}
            </tr>
          </thead>
          <tbody>
            {analysis.conditions.map((c) => {
              const cmp = main.get(c.id);
              const vsParent = parent.get(c.id);
              return (
                <tr key={c.id}>
                  <td className="mono">
                    {c.label} {c.is_reference && <Badge variant="gradient">reference</Badge>}{" "}
                    {c.id === analysis.leader && <Badge variant="ok">leader</Badge>}{" "}
                    {selection && c.contender && c.id !== analysis.leader && <Badge variant="warn">contender</Badge>}
                  </td>
                  <td className="num">
                    {c.n_ok}
                    {c.n_failed > 0 && <span className="verdict--worse"> (+{c.n_failed} failed)</span>}
                  </td>
                  <td className="num">{metricValue(metric, c.mean)}</td>
                  <td className="num">{c.seed_std === null ? "—" : `±${metricValue(metric, c.seed_std)}`}</td>
                  <td className="num">{c.gap === null ? "—" : metricDelta(metric, c.gap)}</td>
                  <td className="num">{cmp ? metricDelta(metric, cmp.diff) : "—"}</td>
                  <td className="num">
                    {cmp ? `${metricDelta(metric, cmp.ci_low)} to ${metricDelta(metric, cmp.ci_high)}` : "—"}
                  </td>
                  <td>{cmp && <CiBar comparison={cmp} scale={scale} />}</td>
                  <td>{cmp ? <VerdictLabel verdict={cmp.verdict} /> : "—"}</td>
                  {selection && (
                    <td className="num">
                      {vsParent ? `${metricDelta(metric, vsParent.diff)} (${c.change})` : "—"}
                    </td>
                  )}
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
