"""Score-only inference for confirmation protocol v2 (2026-10-14).

No file, target, prediction, model, or DMSO loading occurs here. Inputs are
already-scored aligned rows. Seed families average SCORES, never predictions.
Each fixed analysis population has its own sorted LKCP_GROUP bootstrap plan;
all aligned comparisons within that population reuse its exact draw matrix.
Intervals condition on fitted models, references, galleries and readouts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Mapping

import numpy as np

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261785
LOSSES = ("mse", "center_cosine_loss")
RETRIEVAL = ("reference_whiten32_mrr", "reference_whiten32_cosine_margin")
CONDITIONS = ("CP24|task0", "CP24|task1", "CP48|task0", "CP48|task1")


def _labels(values, name, n=None):
    a = np.asarray(values)
    if a.ndim != 1 or not len(a) or (n is not None and len(a) != n):
        raise ValueError(f"{name} must be a nonempty aligned vector")
    if any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in a):
        raise ValueError(f"{name} contains missing identities")
    a = a.astype(str)
    if any(not v.strip() for v in a):
        raise ValueError(f"{name} contains blank identities")
    return a


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class BootstrapPlan:
    group_ids: np.ndarray
    indices: np.ndarray
    seed: int | None = BOOTSTRAP_SEED

    def __post_init__(self):
        groups = _labels(self.group_ids, "group_ids")
        if len(groups) < 2 or not np.array_equal(groups, np.unique(groups)):
            raise ValueError("group_ids must be sorted, distinct, and contain >=2 chemicals")
        idx = np.asarray(self.indices)
        if (idx.ndim != 2 or idx.shape[1] != len(groups) or not len(idx)
                or not np.issubdtype(idx.dtype, np.integer)
                or np.any(idx < 0) or np.any(idx >= len(groups))):
            raise ValueError("indices must be an in-range draws-by-chemicals integer matrix")
        groups = groups.copy(); idx = idx.astype(np.int64, copy=True)
        groups.flags.writeable = False; idx.flags.writeable = False
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "indices", idx)

    def audit(self):
        return {"groups": self.group_ids.tolist(), "n_groups": len(self.group_ids),
                "draws": len(self.indices), "seed": self.seed,
                "numpy_version": np.__version__,
                "index_sha256": hashlib.sha256(self.indices.astype("<i8").tobytes()).hexdigest(),
                "stratified_by_direction": False}


def make_bootstrap_plan(groups, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Exact protocol matrix: sorted unique groups, default_rng(seed).integers."""
    ids = np.unique(_labels(groups, "groups"))
    draws = _positive_int(draws, "draws")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return BootstrapPlan(ids, np.random.default_rng(seed).integers(0, len(ids), (draws, len(ids))), int(seed))


def chemical_means(values, groups, *, plan=None):
    """Within-chemical row means, then equal chemical weights (never row weights)."""
    groups = _labels(groups, "groups")
    a = np.asarray(values, float)
    if a.ndim not in (1, 2) or a.shape[0] != len(groups) or not np.isfinite(a).all():
        raise ValueError("values must be finite aligned scores, not missing-value filled scores")
    ids, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if plan is not None and not np.array_equal(ids, plan.group_ids):
        raise ValueError("Bootstrap plan population differs from score population")
    sums = np.zeros((len(ids),) + a.shape[1:], dtype=float)
    np.add.at(sums, inverse, a)
    return sums / counts.reshape((len(ids),) + (1,) * (a.ndim - 1))


def score_family_means(rows, seed_families=None):
    """Copy finite arm->metric->row scores; append declared seed-score means.

    All members of one family require identical metric sets and aligned rows.
    A family name cannot overwrite an existing arm or depend on another family.
    """
    if not isinstance(rows, Mapping) or not rows:
        raise ValueError("rows must be a nonempty arm-to-metrics mapping")
    result = {}; n = None
    for arm, metrics in rows.items():
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("Every arm requires metrics")
        result[arm] = {}
        for metric, value in metrics.items():
            a = np.asarray(value, float)
            if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
                raise ValueError("Score vectors must be finite and one-dimensional")
            n = len(a) if n is None else n
            if len(a) != n:
                raise ValueError("Every score must share the same row alignment")
            result[arm][metric] = a.copy()
    for family, members in (seed_families or {}).items():
        members = list(members)
        if family in result or not members or len(set(members)) != len(members) or any(m not in rows for m in members):
            raise ValueError("Family requires distinct existing seed arms and a new name")
        metrics = set(rows[members[0]])
        if any(set(rows[m]) != metrics for m in members):
            raise ValueError("Seed family members must share all metric names")
        result[family] = {m: np.mean([result[a][m] for a in members], axis=0) for m in sorted(metrics)}
    return result


def bootstrap_summary(estimate, replicates, *, alternative="two-sided", threshold=0.,
                      invalid_policy="conservative", decision=True, allow_invalid=True,
                      tie_tolerance=0.):
    """Protocol percentile bounds and Monte Carlo p, including threshold ties.

    Annex E: <=1% undefined draws count adverse in p, with the full denominator;
    >1% yields p=1 and an undetermined decision. Bounds use valid draws only.
    Descriptive distributions return intervals/invalid counts but no p/decision.
    ``allow_invalid=False`` is mandatory for finite-loss mean tests K/P/R/I/N/S1/S2.
    An explicitly enabled one-sided ``tie_tolerance`` counts near-boundary draws
    adverse for p only; it never rounds the draws, estimate, or interval. The
    approved H4 addendum is the sole production caller using nonzero tolerance.
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("Unknown alternative")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    if tie_tolerance and alternative == "two-sided":
        raise ValueError("Nonzero tie_tolerance requires a one-sided test")
    if invalid_policy not in (None, "conservative"):
        raise ValueError("Annex E requires conservative invalid-draw handling")
    invalid_policy = "conservative"
    x = np.asarray(replicates, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(threshold):
        raise ValueError("Need a nonempty replicate vector and finite threshold")
    valid = np.isfinite(x); b = len(x); bad = int((~valid).sum())
    finite_point = bool(np.isfinite(estimate))
    if np.isinf(x).any() or np.isinf(estimate) or (not allow_invalid and (bad or not finite_point)):
        raise FloatingPointError("Unexpected nonfinite statistic; implementation error, not an undefined draw")
    ci = np.quantile(x[valid], [.025, .975]).tolist() if valid.any() else [None, None]
    status = "ok"
    if not finite_point or (decision and bad > b * .01):
        status = "undetermined"
    p = 1. if decision else None
    if decision and status == "ok":
        y = x[valid]
        adverse = (np.abs(y - estimate) >= abs(estimate - threshold) if alternative == "two-sided"
                   else y >= threshold if alternative == "less" else y <= threshold)
        if tie_tolerance:
            adverse |= np.abs(y - threshold) < tie_tolerance
        numerator = int(adverse.sum()) + bad
        p = (1 + numerator) / (1 + b)
    return {"estimate": float(estimate) if finite_point else None, "ci95": ci,
            "lower97_5": ci[0], "upper97_5": ci[1], "p_value": p,
            "alternative": alternative, "threshold": float(threshold), "status": status,
            "draws": b, "invalid_draws": bad, "invalid_fraction": bad / b,
            "invalid_policy": invalid_policy,
            "direction": alternative, "null": float(threshold),
            "statistic": float(estimate) if finite_point else None,
            "margin": float(estimate-threshold) if finite_point else None,
            "p_one_sided": p if alternative != 'two-sided' else None,
            "p_two_sided": p if alternative == 'two-sided' else None,
            "reason": ('Undefined point statistic' if not finite_point else
                       'More than 1% undefined draws' if decision and bad > b*.01 else None),
            "interval_valid_for_decision": bool(decision and status == "ok"),
            "descriptive_only": not decision,
            "p_method": "centered_bootstrap" if alternative == "two-sided" else "percentile_bootstrap_tail"}


def paired_summary(differences, groups, *, plan=None, alternative="two-sided", threshold=0.,
                   tie_tolerance=0.):
    plan = make_bootstrap_plan(groups) if plan is None else plan
    grouped = chemical_means(differences, groups, plan=plan)
    if grouped.ndim != 1:
        raise ValueError("paired_summary accepts one difference vector")
    return dict(bootstrap_summary(grouped.mean(), grouped[plan.indices].mean(axis=1),
                             alternative=alternative, threshold=threshold, allow_invalid=False,
                             tie_tolerance=tie_tolerance),
                n_chemicals=len(plan.group_ids), bootstrap=plan.audit())


def unavailable_result(reason, *, status="not_executed"):
    """Retain a declared missing test; never shrink its multiplicity family."""
    if status not in ("not_executed", "undetermined") or not str(reason).strip():
        raise ValueError("Unavailable tests need a declared status and reason")
    return dict(status=status, reason=str(reason), estimate=None, ci95=[None, None],
                lower97_5=None, upper97_5=None, p_value=1., invalid_draws=None,
                interval_valid_for_decision=False)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm requires finite probabilities")
    order = np.argsort(p, kind="stable")
    result = np.empty(len(p)); result[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def classify_primary(family, tests):
    """Mutually exclusive A--E protocol classification from adjusted decisions."""
    comparisons = {"K": ("k3_ext", "mlp_same_head"), "P": ("linear", "mlp256")}
    if family not in comparisons:
        raise ValueError("Only K and P have A--E classifications")
    expected = {(c, m) for c in comparisons[family] for m in LOSSES}
    lookup = {(t["comparator"], t["metric"]): ("unresolved" if t.get("status") in
        ("not_executed", "undetermined") else t["conclusion"]) for t in tests}
    if len(tests) != 4 or set(lookup) != expected or any(v not in ("method_better", "comparator_better", "unresolved") for v in lookup.values()):
        raise ValueError("Classification needs exactly the four prespecified tests")
    if "comparator_better" in lookup.values():
        return "E"
    if all(v == "method_better" for v in lookup.values()):
        return "A"
    mse_comps = comparisons[family] if family == "K" else ("mlp256",)
    if all(lookup[c, "mse"] == "method_better" for c in mse_comps):
        return "B"
    return "C" if "method_better" in lookup.values() else "D"


def _holm_family(name, tests, *, higher_better=False):
    for t in tests:
        t['p_for_holm'] = (float(t['p_value']) if t.get('status', 'ok') == 'ok' else 1.)
    adj = holm_adjust([t["p_for_holm"] for t in tests])
    for t, p in zip(tests, adj):
        t["p_holm"] = float(p); t["reject"] = bool(p <= .05)
        sign = (t["estimate"] * (1 if higher_better else -1)
                if t.get('status', 'ok') == 'ok' and t.get('estimate') is not None else 0.)
        t["conclusion"] = ("method_better" if sign > 0 else "comparator_better") if t["reject"] and sign != 0 else "unresolved"
    out = {"family": name, "alpha": .05, "sidedness": "two-sided", "family_size": len(tests),
           "tests": tests, "all_favorable": all(t["conclusion"] == "method_better" for t in tests)}
    if name in ("K", "P"):
        out["classification"] = classify_primary(name, tests)
    return out


def confirmation_families(rows, groups, codes, arms, *, seed_families=None, plan=None,
                          requested=("K", "P", "R", "I", "N", "S1", "S2"), unavailable=None):
    """Compute all score-only v2 families without selecting models/readouts.

    ``arms`` maps protocol roles to actual scored arm/family names: k4c,
    k3_ext, mlp_same_head, memory, linear, mlp256, rowperm, no_rna, k4c_20,
    k4c_5, G. Missing required roles/metrics raise; no replacement is inferred.
    Requires exactly the four conditions per chemical. Partial-row strata use
    descriptive_strata instead, which cannot emit confirmatory classifications.
    """
    groups = _labels(groups, "groups"); codes = _labels(codes, "codes", len(groups))
    if any(c not in CONDITIONS for c in codes):
        raise ValueError("Unknown condition")
    for g in np.unique(groups):
        cs = codes[groups == g]
        if len(np.unique(cs)) != len(cs) or set(cs) != set(CONDITIONS):
            raise ValueError("Each chemical requires distinct conditions and a complete four-row main trajectory")
    requested = tuple(requested)
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= {"K", "P", "R", "I", "N", "S1", "S2"}:
        raise ValueError("Unknown or repeated requested family")
    unavailable = {} if unavailable is None else unavailable
    if not isinstance(unavailable, Mapping):
        raise ValueError('unavailable must explicitly map roles or role/metric to status/reason')
    for record in unavailable.values():
        unavailable_result(record['reason'], status=record['status'])
    scores = score_family_means(rows, seed_families) if rows else {}
    if scores and len(next(iter(next(iter(scores.values())).values()))) != len(groups):
        raise ValueError("Score rows and identities differ")
    plan = make_bootstrap_plan(groups) if plan is None else plan
    if not np.array_equal(np.unique(groups), plan.group_ids):
        raise ValueError("Bootstrap plan population differs")
    if len(plan.indices) != BOOTSTRAP_DRAWS:
        raise ValueError('Formal families require the full 10,000-draw frozen matrix')
    def value(role, metric):
        try:
            return scores[arms[role]][metric]
        except KeyError as exc:
            raise ValueError(f"Missing prespecified role/metric: {role}/{metric}") from exc
    def unavailable_for(roles, metric):
        records = [unavailable.get(role+'/'+metric, unavailable.get(role)) for role in roles]
        records = [r for r in records if r is not None]
        if not records:
            return None
        status = 'not_executed' if any(r['status'] == 'not_executed' for r in records) else 'undetermined'
        return unavailable_result('; '.join(r['reason'] for r in records), status=status)
    def comparison(left, right, metric, alternative="two-sided"):
        t = unavailable_for((left, right), metric)
        if t is None:
            t = paired_summary(value(left, metric) - value(right, metric), groups, plan=plan, alternative=alternative)
        return dict(t, left=arms.get(left, left), right=arms.get(right, right), comparator=right, metric=metric)
    definitions = {"K": ("k4c", ("k3_ext", "mlp_same_head"), LOSSES),
                   "P": ("memory", ("linear", "mlp256"), LOSSES),
                   "R": ("k4c", ("memory",), LOSSES),
                   "I": ("memory", ("linear", "mlp256"), RETRIEVAL),
                   "N": ("memory", ("rowperm", "no_rna"), LOSSES)}
    out = {}
    for name in requested:
        if name in definitions:
            left, rights, metrics = definitions[name]
            out[name] = _holm_family(name, [comparison(left, r, m) for r in rights for m in metrics], higher_better=name == "I")
        elif name == "S2":
            tests = [comparison("k4c_5", "memory", m, "less") for m in LOSSES]
            for t in tests:
                t["passed"] = t['status'] == 'ok' and t["upper97_5"] < 0
            out[name] = {"family": name, "alpha": .025, "method": "intersection_union",
                         "tests": tests, "passed": all(t["passed"] for t in tests)}
        else:
            steps = []; continuing = True
            for comp in ("linear", "k3_ext", "mlp_same_head"):
                tests = []; prerequisite = {}
                for metric in LOSSES:
                    absent = unavailable_for((comp, 'G', 'k4c_20'), metric)
                    if absent is not None:
                        prerequisite[metric] = None
                        tests.append(dict(absent, metric=metric))
                        continue
                    prerequisite[metric] = float(chemical_means(value(comp, metric) - value("G", metric), groups).mean())
                    delta = value("k4c_20", metric) - .8 * value(comp, metric) - .2 * value("G", metric)
                    tests.append(dict(paired_summary(delta, groups, plan=plan, alternative="less"), metric=metric))
                eligible = all(v is not None and v < 0 for v in prerequisite.values())
                bound_pass = all(t['status'] == 'ok' and t["upper97_5"] < 0 for t in tests)
                passed = continuing and eligible and bound_pass
                steps.append({"comparator": comp, "left": arms.get("k4c_20", "k4c_20"), "right": arms.get(comp, comp),
                              "retention": .8, "comparator_minus_G": prerequisite,
                              "prerequisite_passed": eligible, "evaluated_in_sequence": continuing,
                              "bounds_passed_descriptively": bound_pass, "passed": bool(passed), "tests": tests})
                continuing = bool(passed)
            out[name] = {"family": name, "alpha": .025, "method": "fixed_sequence_intersection_union",
                         "steps": steps, "steps_passed": sum(s["passed"] for s in steps),
                         "later_steps_after_stop_are_descriptive_only": True}
    return {"families": out, "audit": {"bootstrap": plan.audit(), "rows": len(groups),
             "pooling": "within-chemical row means, equal chemicals; no direction stratification",
             "seed_families": {k: list(v) for k, v in (seed_families or {}).items()},
             "seed_aggregation": "per-row score means, NOT prediction ensemble",
             "scope": "main",
             "unavailable": dict(unavailable),
             "models_references_readouts_fixed": True,
             "error_control": "within each family, not across the protocol"}}


def discovery_holm(results):
    """Combine supplied H statistics without fitting/redefining their endpoints.

    Each present member is a bootstrap_summary-compatible mapping. H2 must
    replace p_value by its frozen within-cell permutation p and set p_method
    to 'permutation'. Missing/undefined members are retained as p=1 for family
    size, never dropped to make Holm easier. Inference remains undetermined.
    """
    names = ("H1", "H2", "H3", "H4a", "H4b", "H5")
    if not set(results) <= set(names):
        raise ValueError("Unknown discovery hypothesis")
    out = {}
    for label, members in (("core", names[:2]), ("secondary", names[2:])):
        records = []
        for name in members:
            if name not in results or not results[name]:
                r = dict(unavailable_result('Prespecified endpoint result was not supplied'),
                    alternative='greater', threshold=.3 if name == 'H1' else 0.,
                    p_one_sided=1., p_two_sided=None,
                    p_method='permutation' if name == 'H2' else 'percentile_bootstrap_tail')
            else:
                r = dict(results[name])
            p = r.get("p_value")
            if p is not None and (not np.isfinite(p) or not 0 <= p <= 1):
                raise ValueError("Invalid discovery p value")
            supplied = p is not None and r.get('status', 'ok') not in ('not_executed', 'undetermined')
            if name == "H2" and supplied and r.get("p_method") != "permutation":
                raise ValueError("H2 rejection requires the prespecified permutation p value")
            if name != "H2" and supplied:
                threshold = .3 if name == "H1" else 0.
                if r.get("alternative") != "greater" or r.get("threshold") != threshold:
                    raise ValueError(f"{name} requires a greater-than test at threshold {threshold}")
            valid = bool(p is not None and r.get('status') != 'not_executed' and (
                r.get('permutation_status', r.get('status', 'ok')) == 'ok'
                if name == 'H2' else r.get('status', 'ok') == 'ok'))
            records.append(dict(r, name=name, available_for_rejection=valid,
                                p_for_holm=float(p) if valid else 1.))
        adj = holm_adjust([r["p_for_holm"] for r in records])
        for r, p in zip(records, adj):
            upper = r.get("upper97_5")
            negative_threshold = .3 if r["name"] == "H1" else .2 if r["name"] == "H2" else 0.
            supported = r["available_for_rejection"] and p <= .025
            # An undefined H1/S3 bootstrap is never rescued by its finite subset CI.
            interval_status = (r.get('bootstrap_status', r.get('status', 'ok'))
                               if r['name'] == 'H2' else r.get('status', 'ok'))
            valid_interval = interval_status == "ok" and upper is not None and np.isfinite(upper)
            conclusion = "supported" if supported else "clearly_not_supported" if valid_interval and upper < negative_threshold else "undetermined"
            r.update(p_holm=float(p), conclusion=conclusion,
                     conclusion_detail=('supported_strength_undetermined' if supported
                        and r['name'] == 'H2' and not valid_interval else conclusion),
                     negative_description_is_uncorrected=True, negative_threshold=negative_threshold,
                     weak_consistency=bool(r["name"] == "H2" and supported and valid_interval and upper < .2))
        out[label] = {"family_size": len(members), "alpha": .025, "tests": records,
                      "missing_members_not_removed_from_family": True}
    return out


def compare_e1_6_records(left, right, *, relation="independent"):
    """Audit one aligned result-record projection against Annex E1.6.

    Callers supply the same result fields, not implementation-specific prose
    or file paths. For independent implementations only point/interval/bound
    fields tolerate an absolute difference strictly below 1e-10. Every other
    field, including p values, invalid counts, identities, matrix hashes,
    rejection decisions and classifications, must be identical. Same-code
    floating values are compared by float64 bytes, with no tolerance.

    This function never grants a boundary exception. A p/count mismatch can
    be investigated with boundary_difference_audit and the original draws;
    the discrepancy remains explicit until the independent review accepts
    that evidence. In particular, different Holm decisions are not waived.
    """
    if relation not in ("same_code", "independent"):
        raise ValueError("E1.6 relation must be same_code or independent")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping) or not left or set(left) != set(right):
        raise ValueError("Compare nonempty records with exactly the same fields")
    numeric_fields = {"point", "estimate", "statistic", "ci95", "lower97_5", "upper97_5", "margin"}
    discrepancies, compared = [], []
    max_difference = 0.

    def scalar(value):
        return value.item() if isinstance(value, np.generic) else value

    def compare(a, b, path, tolerant):
        nonlocal max_difference
        a, b = scalar(a), scalar(b)
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if isinstance(b, np.ndarray):
            b = b.tolist()
        if isinstance(a, Mapping) or isinstance(b, Mapping):
            if not isinstance(a, Mapping) or not isinstance(b, Mapping) or set(a) != set(b):
                discrepancies.append(dict(field=path, reason="record fields differ"))
                return
            for key in sorted(a):
                compare(a[key], b[key], path+"."+str(key), tolerant)
            return
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != len(b):
                discrepancies.append(dict(field=path, reason="array shape differs"))
                return
            for i, (aa, bb) in enumerate(zip(a, b)):
                compare(aa, bb, path+f"[{i}]", tolerant)
            return
        numeric_a = isinstance(a, (int, float)) and not isinstance(a, bool)
        numeric_b = isinstance(b, (int, float)) and not isinstance(b, bool)
        if numeric_a and numeric_b:
            if not np.isfinite(a) or not np.isfinite(b):
                raise ValueError("Use documented null, not nonfinite exchange results: "+path)
            difference = abs(float(a)-float(b))
            max_difference = max(max_difference, difference) if tolerant else max_difference
            if tolerant and relation == "independent":
                equal = difference < 1e-10
            elif isinstance(a, float) or isinstance(b, float):
                equal = np.float64(a).tobytes() == np.float64(b).tobytes()
            else:
                equal = a == b
            if not equal:
                discrepancies.append(dict(field=path, left=a, right=b, absolute_difference=difference,
                    reason="strict absolute difference <1e-10 required" if tolerant and relation == "independent"
                           else "exact equality required"))
        elif type(a) is not type(b) or a != b:
            discrepancies.append(dict(field=path, left=a, right=b, reason="exact equality required"))

    for field in sorted(left):
        compared.append(field)
        compare(left[field], right[field], str(field), field in numeric_fields)
    return dict(schema="v16_annexE_E1_6_comparison_v1", relation=relation,
        passed=not discrepancies, status="matched" if not discrepancies else "mismatch",
        compared_fields=compared, maximum_point_interval_bound_difference=max_difference,
        discrepancies=discrepancies, boundary_exceptions_automatically_accepted=False,
        criterion="same-code exact; independent point/CI/bounds abs<1e-10; all p/count/decisions exact")


def boundary_difference_audit(left_values, right_values, *, left_boundary,
                              right_boundary=None, operator="le", draw_ids=None):
    """List each differing boundary decision from two complete draw vectors.

    Values must be the actual finite decision quantities from each independent
    implementation, aligned to the same frozen draw IDs. For a recentered
    two-sided p, supply abs(draw-point) against abs(point-null); for H2 invalid
    residuals supply the minimum residual norm against 1e-12. Boundaries may
    be side-specific because the independently calculated point can differ.
    NaN placeholders or summary p values cannot establish a boundary cause.

    Every changed decision is listed, not just a favorable subset. This is
    supporting evidence, never permission to accept a changed final decision
    or a substitute for the independent script and full statistical check.
    """
    if operator not in ("le", "lt", "ge", "gt"):
        raise ValueError("Unknown inclusive/strict boundary operator")
    a, b = np.asarray(left_values, float), np.asarray(right_values, float)
    if a.ndim != 1 or not len(a) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Complete aligned finite draw-level decision quantities are required")
    if right_boundary is None:
        right_boundary = left_boundary
    ba = np.broadcast_to(np.asarray(left_boundary, float), a.shape)
    bb = np.broadcast_to(np.asarray(right_boundary, float), b.shape)
    if not np.isfinite(ba).all() or not np.isfinite(bb).all():
        raise ValueError("Decision boundaries must be finite")
    ids = np.arange(len(a)) if draw_ids is None else np.asarray(draw_ids)
    if ids.shape != a.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("Supply one unique identity per aligned draw")
    operation = {"le": np.less_equal, "lt": np.less, "ge": np.greater_equal, "gt": np.greater}[operator]
    aa, ab = operation(a, ba), operation(b, bb)
    changed = np.flatnonzero(aa != ab)
    records = []
    for i in changed:
        distance = min(abs(float(a[i]-ba[i])), abs(float(b[i]-bb[i])))
        records.append(dict(draw_id=ids[i].item() if isinstance(ids[i], np.generic) else ids[i],
            left_value=float(a[i]), right_value=float(b[i]), left_boundary=float(ba[i]),
            right_boundary=float(bb[i]), left_condition=bool(aa[i]), right_condition=bool(ab[i]),
            nearest_boundary_distance=distance, within_E1_6_tolerance=bool(distance < 1e-12)))
    eligible = bool(records) and all(r["within_E1_6_tolerance"] for r in records)
    return dict(schema="v16_annexE_boundary_audit_v1", draws=len(a), operator=operator,
        left_condition_count=int(aa.sum()), right_condition_count=int(ab.sum()),
        changed_draw_count=len(records), changed_draws=records,
        all_discrepancies_within_tolerance=eligible,
        status="no_discrepancies" if not records else "boundary_review_required" if eligible else "outside_tolerance",
        automatic_acceptance=False, final_decisions_must_still_match=True)


def strata_from_metadata(contexts, codes, *, layouts=None, donor_energy=None, tas_bins=None):
    """Predefined descriptive masks; TAS bins must already be frozen by C4.

    No cutpoints are estimated here. Donor energy cutpoints are exactly the
    protocol's development-fixed 2.485 and 7.268. A chemical may contribute
    different rows to different bands; it is re-averaged within each band.
    """
    contexts = _labels(contexts, "contexts"); n = len(contexts)
    codes = _labels(codes, "codes", n)
    if any(c not in CONDITIONS for c in codes):
        raise ValueError("Unknown condition")
    masks = {f"direction:{v}": contexts == v for v in np.unique(contexts)}
    masks.update({f"condition:{v}": codes == v for v in CONDITIONS})
    masks.update({f"time:CP{t}": np.char.startswith(codes, f"CP{t}|") for t in (24, 48)})
    masks.update({f"task:{t}": np.char.endswith(codes, f"|task{t}") for t in (0, 1)})
    if layouts is not None:
        layouts = _labels(layouts, "layouts", n)
        masks.update({f"layout_condition:{l}/{c}": (layouts == l) & (codes == c)
                      for l in np.unique(layouts) for c in CONDITIONS})
    if donor_energy is not None:
        e = np.asarray(donor_energy, float)
        if e.shape != (n,) or not np.isfinite(e).all() or np.any(e < 0):
            raise ValueError("Donor energy must be finite, nonnegative and row-aligned")
        masks.update({"donor:low": e < 2.485, "donor:mid": (e >= 2.485) & (e < 7.268),
                      "donor:high": e >= 7.268})
    if tas_bins is not None:
        t = np.asarray(tas_bins)
        if t.shape != (n,) or not np.isin(t, [0, 1, 2]).all():
            raise ValueError("Supply frozen TAS bin codes 0/1/2")
        masks.update({f"TAS:{b}": t == i for i, b in enumerate(("low", "mid", "high"))})
    return masks


def descriptive_strata(differences, groups, masks, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Nominal CI for supplied prespecified contrasts, without family decisions.

    Each stratum re-averages its rows within chemical, then uses its own sorted
    population with the same RNG seed, matching existing reporting. Empty or
    singleton strata are reported unavailable, not silently omitted. No Holm,
    primary classification, rejection or causal interpretation is attached.
    """
    groups = _labels(groups, "groups"); n = len(groups)
    if not differences or not masks:
        raise ValueError("Provide contrast vectors and descriptive masks")
    for v in differences.values():
        a = np.asarray(v, float)
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError("Contrasts must be finite aligned row vectors")
    out = {}
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError("Strata must be explicit boolean row masks")
        ids = np.unique(groups[mask])
        record = {"rows": int(mask.sum()), "groups": len(ids), "descriptive_only": True}
        if len(ids) < 2:
            record.update(status="unavailable", reason="fewer_than_two_chemicals", tests={})
        else:
            p = make_bootstrap_plan(groups[mask], draws=draws, seed=seed)
            tests = {}
            for key, v in differences.items():
                s = paired_summary(np.asarray(v)[mask], groups[mask], plan=p)
                tests[key] = {"estimate": s["estimate"], "ci95": s["ci95"]}
            record.update(status="ok", tests=tests, bootstrap=p.audit())
        out[name] = record
    return out
