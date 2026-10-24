# Implementation milestones

The repository implements the six planned stages:

1. A working FastAPI proxy and bounded backend.
2. Atomic Redis Token Bucket and exact Sliding Window Log, shared/global admission and client allocation.
3. Request and backend instrumentation, provisioned Grafana, seeded open-arrival workloads, calibration.
4. EWMA/AIMD feedback, bounded budgets, decision history, failure handling.
5. Isolated benchmark services, static/adaptive policy matrix, repeated randomized trials and saved reports.
6. CI, tests, architecture source/render, documented demo, methodology and limitations.

The project can be extended incrementally after collecting full-length results. Priorities are adaptive concurrency as an additional established baseline, work-conserving client allocation, mixed-cost workloads, and an independently instrumented upstream with a real database or dependency. Only consider ML after those baselines expose a repeatable gap that forecasting can plausibly address.

The 65–95 hour estimate in the original plan was a planning estimate for a student developing and studying the system. An implementation existing in the repository does not replace running experiments, inspecting failures, and understanding its tradeoffs.
