"""Numerical archive comparisons; hashes identify provenance only."""
import numpy as np


def compare_source_residuals(actual, archived, *, relative_tolerance=1e-10,
                             absolute_tolerance=1e-10):
    """Require both full-array relative L2 and maximum absolute errors to pass."""
    actual, archived = np.asarray(actual, float), np.asarray(archived, float)
    if (actual.ndim != 2 or actual.shape != archived.shape or not actual.size
            or not np.isfinite(actual).all() or not np.isfinite(archived).all()):
        raise ValueError("Source residual arrays must be finite, nonempty and shape-aligned")
    difference = actual - archived
    relative_error = float(np.linalg.norm(difference) / max(np.linalg.norm(archived), 1e-15))
    maximum_absolute_difference = float(np.max(np.abs(difference)))
    passed = (relative_error <= relative_tolerance
              and maximum_absolute_difference <= absolute_tolerance)
    report = dict(passed=passed, shape=list(actual.shape),
        relative_l2_error=relative_error,
        maximum_absolute_difference=maximum_absolute_difference,
        relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance,
        comparison="complete archived source OOF residual array",
        hashes_used_for_numeric_acceptance=False)
    if not passed:
        raise AssertionError("Source OOF residuals differ from archive beyond tolerance: " + str(report))
    return report
