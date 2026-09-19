export type Result<T, E = DomainError> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

export type Option<T> =
  { readonly some: true; readonly value: T } | { readonly some: false };

export type ErrorComponent =
  | "camera"
  | "vision"
  | "asr"
  | "translation"
  | "tts"
  | "renderer"
  | "network"
  | "protocol";

export interface DomainError {
  component: ErrorComponent;
  code: string;
  message: string;
  recoverable: boolean;
  details?: Record<string, unknown>;
}

export function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

export function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

export function some<T>(value: T): Option<T> {
  return { some: true, value };
}

export function none(): Option<never> {
  return { some: false };
}
