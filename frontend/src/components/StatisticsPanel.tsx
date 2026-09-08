import {
  comparisonLabel,
  effectSizeLabel,
  isSignificant,
  num,
} from "../lib/format";
import type { StatisticalComparison } from "../types/api";
import { Card, Empty, StatisticalTag } from "./ui";

/**
 * StatisticsPanel (task 7.8 / 7.13). Everything here is scipy output from the
 * analysis node — rendered in monospace under a "Statistical Results" tag,
 * never reinterpreted.
 */
export function StatisticsPanel({
  comparisons,
}: {
  comparisons: StatisticalComparison[];
}) {
  return (
    <Card title="Statistical Analysis" badge={<StatisticalTag />}>
      {comparisons.length === 0 ? (
        <Empty>
          No comparisons yet. The analysis node needs at least two conditions
          with ≥2 successful experiments each.
        </Empty>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Comparison</th>
                <th>Metric</th>
                <th className="num">t</th>
                <th className="num">p-value</th>
                <th className="num">Cohen&apos;s d</th>
                <th>Effect</th>
                <th className="num">95% CI</th>
                <th className="num">n</th>
                <th>Significant?</th>
              </tr>
            </thead>
            <tbody>
              {comparisons.map((c) => {
                const sig = isSignificant(c.p_value);
                return (
                  <tr key={c.comparison_id}>
                    <td className="mono">{comparisonLabel(c)}</td>
                    <td className="mono">{c.metric}</td>
                    <td className="num">{num(c.t_statistic, 2)}</td>
                    <td className="num">{num(c.p_value, 4)}</td>
                    <td className="num">{num(c.effect_size, 2)}</td>
                    <td>{effectSizeLabel(c.effect_size)}</td>
                    <td className="num">
                      [{num(c.confidence_interval[0], 3)},{" "}
                      {num(c.confidence_interval[1], 3)}]
                    </td>
                    <td className="num">
                      {c.sample_sizes[0]}/{c.sample_sizes[1]}
                      {c.warning ? " ⚠" : ""}
                    </td>
                    <td className={sig ? "sig-yes" : "sig-no"}>
                      {sig ? "✓ yes" : "no"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {comparisons.some((c) => c.warning) && (
        <p className="card__hint" style={{ marginTop: 12 }}>
          ⚠ underpowered: fewer than 5 successful experiments per condition —
          treat the p-value with caution.
        </p>
      )}
    </Card>
  );
}
