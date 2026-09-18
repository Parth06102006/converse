# ADR-001: Repository Architecture

## Status

Accepted

## Context

The project contains TypeScript applications/backend code and Python-based
machine-learning components.

## Decision

Use a pnpm + Turborepo monorepo for JavaScript/TypeScript applications,
services, and shared packages.

Manage Python ML components independently using Python-native tooling.

## Consequences

### Positive

- Shared TypeScript contracts
- Centralized frontend/backend tooling
- Dependency-aware builds
- Independent Python ML environments
- Clear ecosystem boundaries

### Negative

- Two package-management ecosystems
- CI requires both Node and Python tooling
