import {
  comparisonLabel,
  effectSizeLabel,
  isSignificant,
  num,
  testTypeLabel,
} from "../lib/format";
import type {
  ComparisonSkip,
  ConditionSummary,
  StatisticalComparison,
} from "../types/api";
import { Badge, Card, Empty, StatisticalTag } from "./ui";

/**
 * StatisticsPanel (task 7.8 / 7.13). Everything here is scipy output from the
 * analysis node — rendered in monospace under a "Statistical Results" tag,
 * never reinterpreted.
 *
 * The empty state used to assert a fixed reason ("needs at least two
 * conditions with ≥2 successful experiments each") that was often simply
 * untrue — in one real run both conditions had 16 and 6 successful
 * replicates, and the actual blocker was that one of them was deterministic.
 * The backend now records why each pair was skipped and this renders that,
 * rather than guessing.
 */
export function StatisticsPanel({
  comparisons,
  skipped = [],
  conditionSummaries = [],
}: {
  comparisons: StatisticalComparison[];
  skipped?: ComparisonSkip[];
  conditionSummaries?: ConditionSummary[];
}) {
  const hasOneSample = comparisons.some((c) => c.test_type === "one_sample_t");

  return (
    <Card
      title="Statistical Analysis"
      badge={<StatisticalTag />}
      hint="Computed with scipy over every experiment run so far — cumulative, not per cycle."
    >
      {comparisons.length === 0 && skipped.length === 0 ? (
        <Empty>
          No comparisons yet — the analysis node runs after the first cycle of
          experiments completes.
        </Empty>
      ) : null}

      {comparisons.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Comparison</th>
                <th>Metric</th>
                <th>Test</th>
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
                    <td className="mono">
                      {c.test_type === "one_sample_t" ? "1-sample" : "2-sample"}
                    </td>
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
          ⚠ underpowered: fewer than 5 successful replicates in a varying
          condition — treat the p-value with caution.
        </p>
      )}

      {hasOneSample && (
        <p className="card__hint" style={{ marginTop: 8 }}>
          {testTypeLabel(
            comparisons.find((c) => c.test_type === "one_sample_t")!,
          )}
          : a condition whose result is identical for every random seed (as
          linear_baseline is) is treated as a known constant and tested
          against, rather than the comparison being skipped.
        </p>
      )}

      {skipped.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3 className="card__hint" style={{ margin: "0 0 6px" }}>
            Pairs that could not be compared
          </h3>
          <ul className="list-plain">
            {skipped.map((s, i) => (
              <li key={`${s.condition_a_name}|${s.condition_b_name}|${i}`}>
                <Badge variant="warn">{s.reason_code.replace("_", " ")}</Badge>{" "}
                <span className="mono">
                  {s.condition_a_name} vs {s.condition_b_name}
                </span>{" "}
                — {s.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {conditionSummaries.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3 className="card__hint" style={{ margin: "0 0 6px" }}>
            Per condition{" "}
            <span style={{ fontWeight: 400 }}>
              (a condition is a configuration without its random seed; each
              replicate differs only by seed)
            </span>
          </h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Condition</th>
                  <th className="num">replicates</th>
                  <th className="num">mean</th>
                  <th className="num">std</th>
                  <th className="num">min</th>
                  <th className="num">max</th>
                </tr>
              </thead>
              <tbody>
                {conditionSummaries.map((s) => (
                  <tr key={s.condition_name}>
                    <td className="mono">
                      {s.condition_name}
                      {s.deterministic && (
                        <>
                          {" "}
                          <Badge variant="muted">deterministic</Badge>
                        </>
                      )}
                    </td>
                    <td className="num">
                      {s.n_successful}
                      {s.n_anomalous > 0 ? ` (+${s.n_anomalous} flagged)` : ""}
                      {s.n_failed > 0 ? ` (+${s.n_failed} failed)` : ""}
                    </td>
                    <td className="num">{num(s.mean, 4)}</td>
                    <td className="num">{num(s.std, 4)}</td>
                    <td className="num">{num(s.min, 4)}</td>
                    <td className="num">{num(s.max, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {conditionSummaries.some((s) => s.deterministic) && (
            <p className="card__hint" style={{ marginTop: 8 }}>
              A <strong>deterministic</strong> condition returns the same
              result for every random seed, so extra replicates of it add no
              information.
            </p>
          )}
        </div>
      )}
    </Card>
  );
}
