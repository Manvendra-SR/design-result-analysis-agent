import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DatasetManager } from "../DatasetManager";
import { renderWithClient } from "../../test/utils";
import { api } from "../../services/api";

vi.mock("../../services/api", () => ({
  apiErrorMessage: (e: unknown) => String(e),
  isNotFound: () => false,
  errorStatus: (e: unknown) =>
    (e as { __status?: number })?.__status ?? 0,
  api: {
    ingestDataset: vi.fn(),
    listDatasets: vi.fn(),
    deleteDataset: vi.fn(),
  },
}));

const mockApi = api as unknown as {
  ingestDataset: ReturnType<typeof vi.fn>;
  listDatasets: ReturnType<typeof vi.fn>;
  deleteDataset: ReturnType<typeof vi.fn>;
};

function csvFile(rows: number) {
  const lines = ["age,glucose,churned"];
  for (let i = 0; i < rows; i++) {
    lines.push(`${20 + i},${90 + i},${i % 2 === 0 ? "yes" : "no"}`);
  }
  return new File([lines.join("\n")], "diabetes_risk.csv", { type: "text/csv" });
}

beforeEach(() => {
  vi.clearAllMocks();
  mockApi.listDatasets.mockResolvedValue([]);
});

describe("DatasetManager", () => {
  it("previews an uploaded CSV and ingests it as text (no multipart)", async () => {
    mockApi.ingestDataset.mockResolvedValue({
      dataset_id: "ds-x",
      original_filename: "diabetes_risk.csv",
      task_type: "classification",
      n_classes: 2,
      n_rows: 30,
      n_features: 2,
    });
    const onIngested = vi.fn();
    const user = userEvent.setup();
    renderWithClient(<DatasetManager onIngested={onIngested} />);

    await user.upload(
      document.querySelector('input[type="file"]') as HTMLInputElement,
      csvFile(30),
    );

    // preview appears
    expect(await screen.findByText(/first 5 rows/i)).toBeInTheDocument();
    expect(screen.getByText(/30 rows · 3 columns/)).toBeInTheDocument();

    // target column defaulted to the last column and is selectable
    const target = screen.getByLabelText(/target column/i) as HTMLSelectElement;
    expect(target.value).toBe("churned");

    await user.click(screen.getByRole("button", { name: /ingest dataset/i }));

    await waitFor(() => expect(mockApi.ingestDataset).toHaveBeenCalled());
    const body = mockApi.ingestDataset.mock.calls[0][0];
    expect(body.filename).toBe("diabetes_risk.csv");
    expect(body.target_column).toBe("churned");
    expect(typeof body.csv_content).toBe("string");
    expect(body.csv_content).toContain("age,glucose,churned");
    expect(onIngested).toHaveBeenCalled();

    expect(await screen.findByText(/ingested as/i)).toBeInTheDocument();
  });

  it("rejects a file with too few rows before hitting the backend", async () => {
    const user = userEvent.setup();
    renderWithClient(<DatasetManager />);

    await user.upload(
      document.querySelector('input[type="file"]')!,
      csvFile(5),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(/at least 20/i);
    expect(mockApi.ingestDataset).not.toHaveBeenCalled();
  });

  it("lists previously uploaded datasets and deletes one after confirmation", async () => {
    mockApi.listDatasets.mockResolvedValue([
      {
        dataset_id: "ds-old",
        original_filename: "old.csv",
        task_type: "classification",
        n_rows: 200,
      },
    ]);
    mockApi.deleteDataset.mockResolvedValue({
      deleted: "dataset",
      id: "ds-old",
      sessions_deleted: 0,
      experiments_deleted: 0,
    });
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const user = userEvent.setup();
    renderWithClient(<DatasetManager />);

    await user.click(
      await screen.findByText(/manage uploaded datasets \(1\)/i),
    );
    await user.click(
      await screen.findByRole("button", { name: /delete dataset: old\.csv/i }),
    );

    await waitFor(() =>
      expect(mockApi.deleteDataset).toHaveBeenCalledWith("ds-old", undefined),
    );
    confirmSpy.mockRestore();
  });
});
