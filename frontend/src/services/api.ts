/*
 * Thin axios wrapper over the REST API. One function per endpoint;
 * no caching or retry logic here (react-query owns that).
 */
import axios from "axios";

import type {
  ApiError,
  DatasetIngestRequest,
  DatasetProfile,
  DeleteResult,
  Session,
  SessionDetail,
  SessionSummary,
} from "../types/api";

/** Thrown status code an axios error carries, or 0. */
export function errorStatus(err: unknown): number {
  return axios.isAxiosError(err) ? (err.response?.status ?? 0) : 0;
}

export const http = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
  timeout: 60000,
});

/** Pull the human-readable message out of an {error, message, details} body. */
export function apiErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const data = err.response?.data as ApiError | undefined;
    if (data && typeof data.message === "string") return data.message;
    if (err.message) return err.message;
  }
  if (err instanceof Error) return err.message;
  return "Unexpected error";
}

export function isNotFound(err: unknown): boolean {
  return axios.isAxiosError(err) && err.response?.status === 404;
}

export const api = {
  // datasets
  listDatasets: () =>
    http.get<DatasetProfile[]>("/datasets").then((r) => r.data),
  getDataset: (id: string) =>
    http.get<DatasetProfile>(`/datasets/${id}`).then((r) => r.data),
  ingestDataset: (body: DatasetIngestRequest) =>
    http.post<DatasetProfile>("/datasets", body).then((r) => r.data),
  deleteDataset: (id: string, cascade = false) =>
    http
      .delete<DeleteResult>(`/datasets/${id}`, {
        params: cascade ? { cascade: true } : undefined,
      })
      .then((r) => r.data),

  // sessions
  listSessions: () =>
    http.get<SessionSummary[]>("/sessions").then((r) => r.data),
  getSession: (id: string) =>
    http.get<SessionDetail>(`/sessions/${id}`).then((r) => r.data),
  createSession: (body: { research_question: string; dataset_id: string }) =>
    http.post<Session>("/sessions", body).then((r) => r.data),
  deleteSession: (id: string) =>
    http.delete<DeleteResult>(`/sessions/${id}`).then((r) => r.data),
  /** Starts the investigation server-side and returns at once (202). */
  runSession: (id: string) =>
    http.post<Session>(`/sessions/${id}/run`).then((r) => r.data),
};
