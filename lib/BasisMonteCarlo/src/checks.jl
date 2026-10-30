# =============================================================================
# Verifiable checks: every result scored against a stated target
# =============================================================================
# A model is only as good as the checks it passes. Each check here is a number measured from the
# simulation, a target with its tolerance and source, and a pass or a fail: no result is judged by
# eye. Notebooks end with their scorecard and save it with what they make.
# =============================================================================

"""
    Check(name, value, target, passed, source)

One verifiable check: the quantity measured (`value`), what it must meet (`target`, in words),
whether it does, and where the target comes from.
"""
struct Check
    name::String
    value::Float64
    target::String
    passed::Bool
    source::String
end

"""`value` within `tolerance` (a fraction) of `reference`."""
function check_relative(name, value, reference, tolerance; source = "")
    target = "$(round(reference; sigdigits = 4)) ± $(round(100tolerance; sigdigits = 3)) %"
    return Check(name, value, target, abs(value / reference - 1) <= tolerance, source)
end

"""`value` at most `limit`."""
check_below(name, value, limit; source = "") =
    Check(name, value, "≤ $(round(limit; sigdigits = 4))", value <= limit, source)

"""`value` between `low` and `high`."""
check_range(name, value, low, high; source = "") =
    Check(name, value, "$(round(low; sigdigits = 4)) – $(round(high; sigdigits = 4))", low <= value <= high, source)

"""
    scorecard(checks) -> String

The checks as a Markdown table, with the count that pass.
"""
function scorecard(checks::AbstractVector{Check})
    rows = [
        "| $(c.passed ? "✓" : "✗ **fail**") | $(c.name) | $(round(c.value; sigdigits = 4)) | $(c.target) | $(c.source) |"
            for c in checks
    ]
    passed = count(c -> c.passed, checks)
    return """
    | | check | measured | target | source |
    |---|---|---|---|---|
    $(join(rows, "\n"))

    **$(passed) of $(length(checks)) checks pass.**
    """
end
