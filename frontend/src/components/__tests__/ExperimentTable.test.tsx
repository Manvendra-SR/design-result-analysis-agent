import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ExperimentTable } from "../ExperimentTable";
import { makeConfig, makeExperiment } from "../../test/fixtures";

describe("ExperimentTable", () => {
  it("renders a row per experiment with status colour classes", () => {
    render(
      <ExperimentTable
        experiments={[
          makeExperiment({ status: "success" }),
          makeExperiment({ status: "anomalous" }),
          makeExperiment({ status: "failed", metrics: null, error: "boom" }),
        ]}
      />,
    );

    expect(document.querySelector(".status-pill--success")).toBeInTheDocument();
    expect(document.querySelector(".status-pill--anomalous")).toBeInTheDocument();
    expect(document.querySelector(".status-pill--failed")).toBeInTheDocument();
  });

  it("expands a row to show the raw config / error JSON", async () => {
    const user = userEvent.setup();
    render(
      <ExperimentTable
        experiments={[
          makeExperiment({
            status: "failed",
            metrics: null,
            error: "shape mismatch",
            config: makeConfig({ random_seed: 7 }),
          }),
        ]}
      />,
    );

    expect(screen.queryByText(/shape mismatch/)).not.toBeInTheDocument();
    await user.click(screen.getByText(/mlp ·/));
    expect(screen.getByText(/shape mismatch/)).toBeInTheDocument();
  });

  it("summarises status counts", () => {
    render(
      <ExperimentTable
        experiments={[
          makeExperiment({ status: "success" }),
          makeExperiment({ status: "success" }),
          makeExperiment({ status: "failed", metrics: null }),
        ]}
      />,
    );
    const header = document.querySelector(".card__header") as HTMLElement;
    expect(within(header).getByText("2 ok")).toBeInTheDocument();
    expect(within(header).getByText("1 failed")).toBeInTheDocument();
  });
});
