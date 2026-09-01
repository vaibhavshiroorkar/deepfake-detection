# Result traceability

## Traceability contract

Every reported number must resolve to an analysis command, content hashes, and
MLflow runs. The registry stores references. MLflow stores parameters, metrics,
runtime details, and artifacts.

## Result registry

| Result ID | Paper location | Analysis command | Report SHA-256 | Prediction SHA-256 | MLflow run IDs | Decision | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Acceptance rules

A result becomes `accepted` only when every hash resolves, every required seed
is present, and the analysis uses the frozen protocol. Failed and superseded
runs remain in MLflow. Smoke fixtures cannot enter this registry.
