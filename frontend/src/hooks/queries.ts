/*
 * react-query hooks. Polling is the only thing that makes the dashboard
 * "live": every value shown is a fresh read of backend state.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../services/api";

export const qk = {
  datasets: ["datasets"] as const,
  dataset: (id: string) => ["dataset", id] as const,
  sessions: ["sessions"] as const,
  session: (id: string) => ["session", id] as const,
};

export function useDatasets() {
  return useQuery({ queryKey: qk.datasets, queryFn: api.listDatasets });
}

export function useSessionList() {
  return useQuery({ queryKey: qk.sessions, queryFn: api.listSessions });
}

/** Polls every 2 s while the investigation is running; otherwise reads once. */
export function useSession(sessionId: string) {
  return useQuery({
    queryKey: qk.session(sessionId),
    queryFn: () => api.getSession(sessionId),
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 2000 : false,
    refetchIntervalInBackground: true,
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

/** Starts the run (POST returns immediately); polling then follows it. */
export function useRunInvestigation(sessionId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.runSession(sessionId),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: qk.session(sessionId) });
      qc.invalidateQueries({ queryKey: qk.sessions });
    },
  });
}
