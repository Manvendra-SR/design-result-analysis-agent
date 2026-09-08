import { useState } from "react";

import { DatasetManager } from "./components/DatasetManager";
import { SessionManager } from "./components/SessionManager";
import { SessionView } from "./components/SessionView";

const STEPS = [
  "Upload a CSV dataset",
  "Ask a research question",
  "Watch the agent investigate",
];

/**
 * App shell: a guided two-step landing flow (upload → start investigation),
 * or one investigation's live dashboard.
 */
export function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [lastDatasetId, setLastDatasetId] = useState<string | undefined>();

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-mark" aria-hidden="true">
          A
        </div>
        <h1 className="app-title">Adaptive ML Experiment Agent</h1>
      </header>
      <p className="app-tagline">
        Upload a dataset, ask a question about it, and let an LLM-driven agent
        plan, run, and statistically analyse experiments — cycling until it can
        answer.
      </p>

      {sessionId === null ? (
        <>
          <div className="steps-strip">
            {STEPS.map((label, i) => (
              <span className="step-chip" key={label}>
                <span className="step-num">{i + 1}</span>
                {label}
              </span>
            ))}
          </div>
          <div className="stack">
            <DatasetManager
              onIngested={(profile) => setLastDatasetId(profile.dataset_id)}
            />
            <SessionManager
              onOpenSession={setSessionId}
              preselectedDatasetId={lastDatasetId}
            />
          </div>
        </>
      ) : (
        <SessionView sessionId={sessionId} onBack={() => setSessionId(null)} />
      )}
    </div>
  );
}
