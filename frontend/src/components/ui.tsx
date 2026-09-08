import type { ReactNode } from "react";

export function Card({
  title,
  badge,
  hint,
  right,
  plain,
  children,
}: {
  title?: ReactNode;
  badge?: ReactNode;
  hint?: ReactNode;
  right?: ReactNode;
  plain?: boolean;
  children: ReactNode;
}) {
  return (
    <section className={plain ? "card card--plain" : "card"}>
      {(title || badge || right) && (
        <div className="card__header">
          {title && <h2 className="card__title">{title}</h2>}
          {badge}
          {right && <div style={{ marginLeft: "auto" }}>{right}</div>}
        </div>
      )}
      {hint && <p className="card__hint">{hint}</p>}
      {children}
    </section>
  );
}

export function SectionEyebrow({
  step,
  children,
}: {
  step?: number;
  children: ReactNode;
}) {
  return (
    <div className="section__eyebrow">
      {step !== undefined && <span className="step-num">{step}</span>}
      {children}
    </div>
  );
}

type BadgeVariant = "gradient" | "muted" | "ok" | "warn" | "danger";

export function Badge({
  variant = "muted",
  children,
}: {
  variant?: BadgeVariant;
  children: ReactNode;
}) {
  return <span className={`badge badge--${variant}`}>{children}</span>;
}

/** Requirement 12.5 / 12.6: label LLM prose distinctly from computed stats. */
export function InterpretationTag() {
  return <span className="badge tag-llm">LLM Interpretation</span>;
}

export function StatisticalTag() {
  return <span className="badge tag-stat">Statistical Results</span>;
}

export function Spinner({ onPrimary }: { onPrimary?: boolean }) {
  return (
    <span
      className={onPrimary ? "spinner spinner--on-primary" : "spinner"}
      aria-label="loading"
    />
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>;
}

export function ErrorBox({ children }: { children: ReactNode }) {
  return (
    <div className="error-box" role="alert">
      {children}
    </div>
  );
}

export function QueryState({
  isLoading,
  error,
  children,
}: {
  isLoading: boolean;
  error: unknown;
  children: ReactNode;
}) {
  if (isLoading) return <Empty>Loading…</Empty>;
  if (error) return <ErrorBox>{String(error)}</ErrorBox>;
  return <>{children}</>;
}
