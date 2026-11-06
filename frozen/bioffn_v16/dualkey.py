import numpy as np
from .crosscell import _inventory, _labels, _matrix
CONDITION_ORDER = ((24, 0), (24, 1), (48, 0), (48, 1))

def _chemical_rows(ids, groups, conditions):
    """Return chemical IDs and their canonical four-row indices."""
    ids, groups = _inventory(ids, groups)
    raw = np.asarray(conditions)
    if raw.shape != (len(ids), 2):
        raise ValueError("Conditions must be aligned (time, task) pairs")
    try:
        values = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Conditions must contain numeric time/task pairs") from exc
    if not np.isfinite(values).all():
        raise ValueError("Conditions must contain finite time/task pairs")
    chemicals = tuple(sorted(set(groups)))
    order = []
    for group in chemicals:
        idx = np.flatnonzero(groups == group)
        found = [tuple(row) for row in values[idx]]
        if len(idx) != 4 or set(found) != set(CONDITION_ORDER):
            raise ValueError(f"Chemical {group} needs exactly four unique conditions: "
                             "(24, 0), (24, 1), (48, 0), (48, 1)")
        by_condition = dict(zip(found, idx))
        order.append([by_condition[condition] for condition in CONDITION_ORDER])
    return ids, groups, chemicals, np.asarray(order, dtype=int)

