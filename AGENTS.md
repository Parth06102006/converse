# Project Agent Directives

## Overview
Converse is a real-time bidirectional ASL <-> English communication engine.

## Architecture Invariants
- **Ecosystem Separation**: TypeScript applications and services (`apps/*`, `services/*`, `packages/*`) are managed with `pnpm` and Turborepo. Python machine learning components (`ml/*`) are managed independently with `uv`.
- **Single Source of Truth**: All shared types, API schemas, and data structures live in `@converse/contracts`. Never define duplicate interfaces across packages.
- **Explicit Failure**: Use `Result<T, E>` and `Option<T>` for expected pipeline degradations and cross-boundary error handling. Avoid throwing untyped exceptions across boundaries.

## Tooling & Verification
- **Package Manager**: Use `pnpm` exclusively (`pnpm build`, `pnpm check-types`, `pnpm lint`). Never invoke `npm`, `yarn`, or `bun`.
- **Python Tooling**: Use `uv` exclusively (`uv run pytest`, `uv pip`, `uv venv`). Never install packages into global Python.
- **Verification Routine**: Verify changes with `pnpm check-types && pnpm lint && pnpm build` before committing.

## Code Quality & Style
- **Zero Emoji Standard**: Strictly zero emojis across all code, comments, documentation, UI strings, and commits.
- **Strict Typing**: Zero `any` types. Enforce strict null and undefined safety on all public interfaces.
- **Git Conventions**: Use conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`) with clear technical rationale.
- **Documentation Sync**: Keep `docs/milestones/`, `docs/contracts.md`, and `docs/decisions.md` aligned whenever APIs or subsystem behaviors change.
