export type ServiceStatus = "ok" | "degraded" | "error";

export interface HealthCheckResponse {
  status: ServiceStatus;
  service: string;
  version?: string;
  uptimeSeconds?: number;
  timestamp: string;
}
