import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatisticsPanel } from "../StatisticsPanel";
import { makeComparison } from "../../test/fixtures";

describe("StatisticsPanel", () => {
  it("flags p < 0.05 as significant and labels the effect size", () => {
    render(
      <StatisticsPanel
        comparisons={[makeComparison({ p_value: 0.03, effect_size: 0.9 })]}
      />,
    );
    expect(screen.getByText(/✓ yes/)).toBeInTheDocument();
    expect(screen.getByText("large")).toBeInTheDocument();
  });

  it("marks p >= 0.05 as not significant", () => {
    render(
      <StatisticsPanel
        comparisons={[makeComparison({ p_value: 0.2, effect_size: 0.1 })]}
      />,
    );
    expect(screen.getByText("no")).toBeInTheDocument();
    expect(screen.getByText("negligible")).toBeInTheDocument();
  });

  it("surfaces the underpowered warning", () => {
    render(
      <StatisticsPanel
        comparisons={[makeComparison({ warning: "underpowered", sample_sizes: [3, 3] })]}
      />,
    );
    expect(screen.getByText(/underpowered/i)).toBeInTheDocument();
  });

  it("shows an empty state with no comparisons", () => {
    render(<StatisticsPanel comparisons={[]} />);
    expect(screen.getByText(/No comparisons yet/i)).toBeInTheDocument();
  });
});
