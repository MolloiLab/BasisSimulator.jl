# BasisSimulator.jl 10x Speed — Discovery Progress Log (v1 — ARCHIVED)

**STATUS: ARCHIVED.** This is the v1 discovery log. All claims here were based on
CPU-only benchmarks and static analysis. The resulting optimizations produced ZERO
speedup on Metal GPU. See `speed-progress.md` for the v2 GPU-first discovery.

---

(Original v1 content preserved for reference — see git history for full text)

## Summary of v1 findings (UNVALIDATED ON GPU)

1. **SPEED-000:** Static profiling audit — claimed forward projection = 88-95% of time. Never measured on GPU.
2. **SPEED-001:** Energy loop fusion — claimed 10-20x on forward projection. 6.5x measured on CPU only.
3. **SPEED-002:** Branchless DDA — claimed 1.1-1.3x. No measurable effect on CPU.
4. **SPEED-003:** AK.jl analysis — claimed overhead = zero. Never tested on GPU.
5. **SPEED-004:** Separable scatter — claimed 31x (measured 147x on CPU). Never tested on GPU.

## v1 build results (CPU ONLY)

| Story | CPU Speedup | GPU Speedup |
|-------|------------|------------|
| Energy loop fusion | 6.3-6.7x | **NOT MEASURED — possibly negative** |
| Separable scatter | 147x | **NOT MEASURED** |
| Branchless DDA | ~1.0x | **NOT MEASURED** |
| CartesianIndices fix | N/A | **NOT MEASURED** |
