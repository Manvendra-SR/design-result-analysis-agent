/*
 * react-query hooks. The polling cadence here is the ONLY thing that makes
 * the dashboard "live" — every value shown is a fresh read of backend state,
 * never a locally advanced timer.
 */
import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { api, isNotFound } from "../services/api";
import type { Recommendation, SessionDetail } from "../types/api";

export const qk = {
  datasets: ["datasets"] as const,
  dataset: (id: string) => ["dataset", id] as const,
  sessions: ["sessions"] as const,
  session: (id: string) => ["session", id] as const,
  experiments: (id: string) => ["experiments", id] as const,
  recommendation: (id: string) => ["recommendation", id] as const,
  cycles: (id: string) => ["cycles", id] as const,
};

/**
 * Poll cadence for the session detail:
 * - stop entirely once concluded;
 * - fast while a run is actually in progress (backend run_phase, or this
 *   tab's own mutation via `activeRun`) - fastest during `executing`;
 * - a slow keep-alive otherwise, so a run started elsewhere is noticed.
 */
function sessionPollInterval(
  data: SessionDetail | undefined,
  activeRun: boolean,
): number | false {
  if (!data) return 2000;
  if (data.status === "concluded" || data.current_node === "concluded") {
    return false;
  }
  const running = activeRun || data.run_phase === "running";
  if (running) return data.current_node === "executing" ? 1500 : 2500;
  return 8000;
}

export function useDatasets() {
  return useQuery({ queryKey: qk.datasets, queryFn: api.listDatasets });
}

export function useSessionList() {
  return useQuery({ queryKey: qk.sessions, queryFn: api.listSessions });
}

export function useSession(sessionId: string, activeRun = false) {
  return useQuery({
    queryKey: qk.session(sessionId),
    queryFn: () => api.getSession(sessionId),
    refetchInterval: (query) =>
      sessionPollInterval(query.state.data, activeRun),
    refetchIntervalInBackground: true,
  });
}

export function useExperiments(sessionId: string, live: boolean) {
  return useQuery({
    queryKey: qk.experiments(sessionId),
    queryFn: () => api.getExperiments(sessionId),
    refetchInterval: live ? 2500 : false,
  });
}

export function useCycles(sessionId: string, live: boolean) {
  return useQuery({
    queryKey: qk.cycles(sessionId),
    queryFn: () => api.getCycles(sessionId),
    refetchInterval: live ? 3000 : false,
  });
}

export function useRecommendation(sessionId: string, live: boolean) {
  return useQuery<Recommendation | null>({
    queryKey: qk.recommendation(sessionId),
    queryFn: async () => {
      try {
        return await api.getRecommendation(sessionId);
      } catch (err) {
        if (isNotFound(err)) return null; // no recommendation until the first run
        throw err;
      }
    },
    refetchInterval: live ? 3000 : false,
  });
}

export function useCreateSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.createSession,
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.sessions }),
  });
}

export function useIngestDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.ingestDataset,
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.datasets }),
  });
}

export function useDeleteSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteSession(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.sessions }),
  });
}

export function useDeleteDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (vars: { id: string; cascade?: boolean }) =>
      api.deleteDataset(vars.id, vars.cascade),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.datasets });
      qc.invalidateQueries({ queryKey: qk.sessions });
    },
  });
}

/**
 * Runs the whole adaptive investigation (POST /run-cycle). One call, many
 * cycles server-side. While it is pending, useSession keeps polling so the
 * loop visualiser tracks the backend's current_node in real time.
 */
export function useRunInvestigation(sessionId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.runCycle(sessionId),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: qk.session(sessionId) });
      qc.invalidateQueries({ queryKey: qk.experiments(sessionId) });
      qc.invalidateQueries({ queryKey: qk.cycles(sessionId) });
      qc.invalidateQueries({ queryKey: qk.recommendation(sessionId) });
      qc.invalidateQueries({ queryKey: qk.sessions });
    },
  });
}
