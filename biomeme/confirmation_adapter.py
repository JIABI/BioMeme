import numpy as np
from bioffn_v16.dualkey import _chemical_rows
from bioffn_v16.transfer import trajectory_rows
from bioffn_v16.gated_transfer import C_GRID, select_gated_transfers

def donor_arrays(episode, donor):
    """Align an explicit matched/permuted/position donor inventory, no file I/O.

    donor={ids, groups, conditions, delta, variant}. Its mapping/provenance is
    constructed upstream with the frozen matching rules. Missing position
    partners must be resolved explicitly there, not zero-filled in this API.
    """
    if donor.get('variant') not in ('true', 'rowperm', 'position'):
        raise ValueError('Declare the donor control variant')
    di, dg, _, _ = _chemical_rows(donor['ids'], donor['groups'], donor['conditions'])
    delta = np.asarray(donor['delta'], float)
    if delta.ndim != 2 or len(delta) != len(di) or not np.isfinite(delta).all():
        raise ValueError('Invalid donor response array')
    pos = {v: i for i, v in enumerate(di)}
    result = []
    for prefix in ('', 'query_'):
        ids, groups, conditions = (episode[prefix+k] for k in ('ids', 'groups', 'conditions'))
        ix = [pos[str(i)] for i in ids]
        np.testing.assert_array_equal(dg[ix], groups)
        np.testing.assert_array_equal(np.asarray(donor['conditions'])[ix], conditions)
        d = delta[ix]
        if d.shape[1] != episode['base'].shape[1]:
            raise ValueError('Donor and full response coordinates differ')
        result.append((d, trajectory_rows(d, groups, conditions)))
    return result


def fit_k4c_head(episode, fitted_base, donor, *, callback=None):
    """Independent complete K4c search on this base's OWN OOF residuals."""
    (dt, tt), (dq, tq) = donor_arrays(episode, donor)
    family = fitted_base['family']
    oof = fitted_base['oof']
    if oof['audit'].get('own_base_family', oof['method']) != family:
        raise PermissionError('Cannot borrow another base family\'s OOF errors')
    models = select_gated_transfers(dt, {family: oof}, ids=episode['ids'],
        groups=episode['groups'], query_groups=episode['query_groups'], trajectory=tt,
        callback=callback)
    chosen = min(C_GRID, key=lambda c: models[family, c].audit['selected_cv_mse'])
    model = models[family, chosen]
    correction = model.predict_correction(dq, groups=episode['query_groups'], trajectory=tq)
    result = dict(prediction=fitted_base['prediction']+correction, model=model,
        c_candidates={c: models[family, c] for c in C_GRID}, selected_c=chosen,
        family=family, donor_variant=donor['variant'], own_oof=True)
    if 'predictions' in fitted_base:
        result['predictions'] = {s: p+correction for s, p in fitted_base['predictions'].items()}
    return result


