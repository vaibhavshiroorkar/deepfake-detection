# ADR-001: Local MLflow tracking

Status: Accepted

Date: 2026-08-25

## Context and constraints

Local experiments need persistent run metadata, metrics, configuration, and
artifacts. The normal workflow must work without a hosted account. Tracking
must remain optional so direct CLI commands do not require MLflow. Local run
state and experiment artifacts must stay out of Git.

## Decision

Use MLflow through the optional `tracking` extra. Store metadata in
`mlflow.db` through SQLite and artifacts in `mlartifacts/`. Start runs through
`ddf run` with layered configuration files and an explicit project root.

## Options considered

| Option | Assessment |
| --- | --- |
| Local MLflow | Provides local run, metric, parameter, and artifact tracking without an account. |
| Weights & Biases | Provides a hosted collaboration service, but adds account and service dependencies. |
| No tracker | Keeps fewer dependencies, but leaves run relationships spread across files. |

## Trade-offs

Local MLflow adds an optional dependency and local database maintenance. It
keeps the primary workflow private and usable offline. It does not replace
existing JSON, checkpoint, and Parquet outputs. W&B remains a candidate when a
supervisor hosts the project or the work needs active team collaboration.

## W&B review trigger

Review W&B before moving to supervisor-hosted or team collaboration.

## Consequences

- Configured runs log runtime details, resolved configuration, metrics, and
  artifacts to a local MLflow run.
- `mlflow.db`, `mlartifacts/`, and generated run outputs remain ignored.
