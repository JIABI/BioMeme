"""Tahoe plan v1.1 section 4: losses, chemical pooling, families and descriptive statistics (Claude's code).

Written from the registered text (sections 4.1-4.5, annex E E1.1-E1.6 as cited there) and my earlier
independent statistics for protocol v2 (indep_stats.py, agreed for U2OS / sci-Plex3). numpy only.
  * row losses: MSE over retained genes; 1 - cos(pred - o, y - o), cos := 0 when either norm is 0 (counted);
  * pooling: per seed -> mean over seeds -> mean over the dose-step rows -> mean over the drug's eligible targets
    in the population -> one value per drug (drugs sorted by Python sorted());
  * bootstrap: default_rng(20261785).integers(0, n, (10000, n)) per population; percentile interval with numpy's
    default linear quantile; two-sided p = (1 + #{|t* - t| >= |t|}) / 10001 with draws whose |t* - t| and |t|
    differ by < 1e-12 counted as adverse; one-sided ('less') p = (1 + #{t* >= 0 or |t*| < 1e-12}) / 10001;
  * Holm within a family, alpha 0.05; conclusion method_better / comparator_better / unresolved;
    YK, YP classes A-E (protocol v2 5.2/5.3); YS2 and YS1 decided by the one-sided 97.5% upper bound < 0 with
    |bound| < 1e-12 failing; YS1 fixed sequence with the comparator-better-than-G prerequisite;
  * ratios: ratio of pooled means per draw; undefined point when the denominator mean <= 0; draws with
    denominator <= 0 invalid; > 1% invalid -> no interval.
"""
from __future__ import annotations

import numpy as np

DRAWS = 10000
BOOT_SEED = 20261785
TOL = 1e-12
METRICS = ('mse', 'center_cosine_loss')
NEURAL = ('mlp256', 'samehead', 'k3_ext')
SEEDS = tuple(range(20261008, 20261013))


def row_losses(pred, truth, center):
    mse = np.mean((pred - truth) ** 2, axis=1)
    a, b = pred - center, truth - center
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cos = np.zeros(len(a))
    ok = (na > 0) & (nb > 0)
    cos[ok] = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return mse, 1.0 - cos, int((na == 0).sum()), int((nb == 0).sum())


class Pop:
    def __init__(self, name, drugs, target_drugs):
        self.name = name
        self.drugs = sorted(drugs)
        if list(drugs) != self.drugs:
            raise ValueError('population order must be sorted()')
        self.target_drugs = {t: sorted(v) for t, v in target_drugs.items()}
        n = len(self.drugs)
        self.idx = (np.random.default_rng(BOOT_SEED).integers(0, n, (DRAWS, n)) if n else
                    np.zeros((DRAWS, 0), np.int64))

    def means(self, values):
        return np.concatenate([values[self.idx[i:i + 500]].mean(axis=1) for i in range(0, DRAWS, 500)])


def plane(analysis, arm, version='fixed', budget='all', donor=''):
    return f'{analysis}/{arm}/{version}/{budget}/{donor or "none"}'


class Scores:
    """Row scores keyed by plane -> {(target, drug, step, seed): array([mse, cos])}."""

    def __init__(self):
        self.rows = {}
        self.cache = {}

    def add(self, pl, target, drug, step, seed, mse, cos):
        d = self.rows.setdefault(pl, {})
        key = (target, drug, int(step), int(seed))
        if key in d:
            raise ValueError('duplicate scored row ' + str((pl, key)))
        d[key] = np.array([mse, cos])

    def chem(self, pl, pop, steps=(1, 2), only_target=None):
        ck = (pl, pop.name, tuple(steps), only_target)
        if ck in self.cache:
            return self.cache[ck]
        seeds = SEEDS if pl.split('/')[1] in NEURAL else (0,)
        rows = self.rows[pl]
        out = []
        for q in pop.drugs:
            per_t = []
            for t in sorted(pop.target_drugs):
                if q not in pop.target_drugs[t] or (only_target is not None and t != only_target):
                    continue
                per_t.append(np.mean([np.mean([rows[t, q, s, sd] for sd in seeds], axis=0) for s in steps], axis=0))
            if not per_t:
                raise ValueError('drug without eligible target: ' + q)
            out.append(np.mean(per_t, axis=0))
        v = np.asarray(out, np.float64).reshape(-1, 2)
        self.cache[ck] = v
        return v


def summarize(x, pop, test=False, alternative='two-sided'):
    x = np.asarray(x, np.float64)
    est = float(x.mean())
    d = pop.means(x)
    lo, hi = np.quantile(d, [0.025, 0.975])
    out = dict(estimate=est, ci95=[float(lo), float(hi)], lower97_5=float(lo), upper97_5=float(hi),
               n_chemicals=len(x), invalid_draws=0)
    if test:
        if alternative == 'two-sided':
            resid = np.abs(d - est) - abs(est)
        else:
            resid = d
        near = np.abs(resid) < TOL
        adverse = (resid >= 0) | near
        out.update(p_value=(1 + int(adverse.sum())) / (DRAWS + 1), near_boundary_draws=int(near.sum()))
    return out


def ratio_summary(a, b, pop):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not len(a) or b.mean() <= 0:
        return dict(status='undefined', estimate=None, ci95=[None, None], n_chemicals=len(a))
    da, db = pop.means(a), pop.means(b)
    valid = db > 0
    invalid = int((~valid).sum())
    if invalid <= 0.01 * DRAWS:
        lo, hi = np.quantile(da[valid] / db[valid], [0.025, 0.975])
        ci = [float(lo), float(hi)]
        status = 'ok'
    else:
        ci, status = [None, None], 'interval_not_reported'
    return dict(status=status, estimate=float(a.mean() / b.mean()), ci95=ci, invalid_draws=invalid,
                invalid_fraction=invalid / DRAWS, n_chemicals=len(a))


def holm(tests, alpha=0.05):
    p = np.array([t['p_value'] for t in tests])
    order = np.argsort(p, kind='stable')
    running, m = 0.0, len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        tests[i]['p_holm'] = float(running)
    for t in tests:
        reject = t['p_holm'] <= alpha
        t['reject'] = bool(reject)
        t['conclusion'] = ('unresolved' if not reject else
                           'method_better' if t['estimate'] < 0 else 'comparator_better')
    return tests


def classify(tests, family):
    c = [t['conclusion'] for t in tests]
    if 'comparator_better' in c:
        return 'E'
    if all(x == 'method_better' for x in c):
        return 'A'
    if family == 'YK' and c[0] == 'method_better' and c[2] == 'method_better':
        return 'B'
    if family == 'YP' and c[2] == 'method_better':
        return 'B'
    if 'method_better' in c:
        return 'C'
    return 'D'


def bound_pass(t):
    u = t['upper97_5']
    return bool(u < 0 and abs(u) >= TOL)


class Stats:
    def __init__(self, scores, pops):
        self.s, self.p = scores, pops

    def contrast(self, left, right, pop='P_pool', test=False, alternative='two-sided', only_target=None):
        P = self.p[pop]
        a = self.s.chem(left, P, only_target=only_target)
        b = self.s.chem(right, P, only_target=only_target)
        return [dict(summarize(a[:, i] - b[:, i], P, test, alternative), metric=m, left=left, right=right)
                for i, m in enumerate(METRICS)]

    def means(self, pl, pop='P_pool', intervals=True, steps=(1, 2), only_target=None):
        P = self.p[pop]
        a = self.s.chem(pl, P, steps=steps, only_target=only_target)
        if not intervals:
            return {m: float(a[:, i].mean()) for i, m in enumerate(METRICS)}
        return {m: summarize(a[:, i], P) for i, m in enumerate(METRICS)}

    def retention(self, comparator, pop='P_pool', test=False):
        P = self.p[pop]
        a = self.s.chem(plane('main', 'full', budget='20'), P)
        b = self.s.chem(comparator, P)
        g = self.s.chem(plane('main', 'G'), P)
        tests, pre = [], {}
        for i, m in enumerate(METRICS):
            pre[m] = float((b[:, i] - g[:, i]).mean())
            tests.append(dict(summarize(a[:, i] - 0.8 * b[:, i] - 0.2 * g[:, i], P, test, 'less'), metric=m))
        return dict(comparator_plane=comparator, comparator_minus_G=pre, tests=tests)

    def families(self):
        full, mem = plane('main', 'full', 'tuned'), plane('main', 'memory', 'tuned')

        def fam(name, pairs):
            tests = [t for l, r in pairs for t in self.contrast(l, r, test=True)]
            holm(tests)
            out = dict(tests=tests)
            if name in ('YK', 'YP'):
                out['classification'] = classify(tests, name)
            return out

        res = dict(
            YR=fam('YR', [(full, mem)]),
            YK=fam('YK', [(full, plane('main', 'k3_ext')), (full, plane('main', 'samehead'))]),
            YP=fam('YP', [(mem, plane('main', 'linear', 'tuned')), (mem, plane('main', 'mlp256'))]),
            YF=fam('YF', [(plane('main', 'full'), plane('main', 'k3_ext')),
                          (plane('main', 'full'), plane('main', 'samehead')),
                          (plane('main', 'memory'), plane('main', 'mlp256'))]))
        ys2 = self.contrast(plane('main', 'full', budget='5'), plane('main', 'memory'), test=True, alternative='less')
        for t in ys2:
            t['passed'] = bound_pass(t)
        res['YS2'] = dict(tests=ys2, passed=all(t['passed'] for t in ys2))
        going, steps = True, []
        for arm in ('linear', 'k3_ext', 'samehead'):
            st = self.retention(plane('main', arm), test=True)
            pre_ok = all(v < 0 for v in st['comparator_minus_G'].values())
            b_ok = all(bound_pass(t) for t in st['tests'])
            st.update(evaluated_in_sequence=going, prerequisite_passed=pre_ok, bounds_passed_descriptively=b_ok,
                      passed=bool(going and pre_ok and b_ok))
            going = st['passed']
            steps.append(st)
        res['YS1'] = dict(steps=steps, steps_passed=int(sum(s['passed'] for s in steps)))
        return res

    def descriptive_first(self):
        full, mem, lin = (plane('main', a, 'tuned') for a in ('full', 'memory', 'linear'))
        pairs = [(full, mem), (plane('main', 'full', budget='5'), plane('main', 'memory')),
                 (full, plane('main', 'k3_ext')), (full, plane('main', 'samehead')), (mem, lin),
                 (mem, plane('main', 'mlp256')), (plane('main', 'full'), plane('main', 'k3_ext')),
                 (plane('main', 'full'), plane('main', 'samehead')), (plane('main', 'memory'), plane('main', 'mlp256'))]
        d = {}
        d['per_target'] = {name: dict(contrasts=[x for a, b in pairs for x in self.contrast(a, b, name)],
                                      retention=[self.retention(plane('main', a), pop=name)
                                                 for a in ('linear', 'k3_ext', 'samehead')])
                           for name in sorted(self.p) if name.startswith('P_t:')}
        d['budget_curves'] = {plane('main', arm, v, b): self.means(plane('main', arm, v, b), intervals=False)
                              for arm in ('full', 'linear', 'memory')
                              for v, b in [('fixed', x) for x in ('0', '10', '20', '5', 'all')] + [('tuned', 'all')]}
        d['tuning'] = dict(
            tuned_minus_fixed={a: self.contrast(plane('main', a, 'tuned'), plane('main', a))
                               for a in ('memory', 'full', 'linear')},
            YS2_tuned_comparator=self.contrast(plane('main', 'full', budget='5'), mem),
            YS1_tuned_comparators=[self.retention(plane('main', a, 'tuned' if a == 'linear' else 'fixed'))
                                   for a in ('linear', 'k3_ext', 'samehead')])
        d['zero_shot'] = dict(
            n_chemicals=len(self.p['P_ZS'].drugs),
            means={k: self.means(k, 'P_ZS') for k in (plane('main', 'ZS-ref'), plane('main', 'ZS-mean'),
                                                       plane('main', 'G'), mem, full)},
            contrasts=[x for arm in ('ZS-ref', 'ZS-mean') for x in self.contrast(full, plane('main', arm), 'P_ZS')])
        panel = {b: self.contrast(plane('main', 'full', 'tuned' if b == 'all' else 'fixed', b),
                                  plane('main', 'panel', budget=b)) for b in ('5', '10', '20', 'all')}
        panel['panel_all_minus_B1'] = self.contrast(plane('main', 'panel'), plane('main', 'B1'))
        d['panel'] = panel
        d['negative_controls'] = dict(
            permuted_minus_memory=self.contrast(plane('permuted', 'full', 'tuned'), mem),
            main_full_minus_memory=self.contrast(full, mem),
            same_plate_permuted_minus_memory=self.contrast(plane('plate_permuted', 'full', 'tuned'),
                                                           plane('plate_true', 'memory', 'tuned'), 'P_S'),
            same_plate_true_minus_memory=self.contrast(plane('plate_true', 'full', 'tuned'),
                                                       plane('plate_true', 'memory', 'tuned'), 'P_S'),
            n_P_S=len(self.p['P_S'].drugs))
        sens = {}
        for analysis, pop in (('cells200', 'P_200'), ('pc32', 'P_pool')):
            f, m, l = (plane(analysis, a, 'tuned') for a in ('full', 'memory', 'linear'))
            planes = [plane(analysis, 'G')] + [plane(analysis, a, v, b) for a in ('full', 'linear', 'memory')
                                               for v, b in (('fixed', '5'), ('fixed', 'all'), ('tuned', 'all'))]
            sens[analysis] = dict(YR=self.contrast(f, m, pop),
                                  YS2=self.contrast(plane(analysis, 'full', budget='5'), plane(analysis, 'memory'), pop),
                                  memory_minus_linear=self.contrast(m, l, pop),
                                  means={k: self.means(k, pop) for k in planes})
        sens['nn_original'] = [x for a, b in ((full, 'k3_ext'), (full, 'samehead'), (mem, 'mlp256'))
                               for x in self.contrast(a, plane('nn_original', b))]
        sens['nn_original_unchanged_memory_minus_linear'] = self.contrast(mem, lin)
        d['sensitivities'] = sens
        arm_planes = ([plane('main', a) for a in ('B1', 'G', 'full', 'k3_ext', 'linear', 'memory', 'mlp256',
                                                  'samehead', 'source')]
                      + [plane('main', a, 'tuned') for a in ('full', 'linear', 'memory')]
                      + [plane('main', 'panel', budget=b) for b in ('10', '20', '5', 'all')]
                      + [plane('permuted', 'full', 'tuned')])
        d['arm_means'] = {k: self.means(k) for k in arm_planes}
        d['subset_arm_means'] = {k: self.means(k, 'P_S') for k in (plane('plate_permuted', 'full', 'tuned'),
                                                                   plane('plate_true', 'full', 'tuned'),
                                                                   plane('plate_true', 'memory', 'tuned'))}
        comps = [x for a in ('source', 'G', 'B1') for x in self.contrast(full, plane('main', a))]
        comps += self.contrast(plane('main', 'memory'), plane('main', 'linear'))
        d['prespecified_additional_contrasts'] = comps
        return d

    def noise_floor(self, replicate, targets):
        """replicate: {'t|q|2': y14}; targets: {'t|q|2': y6 (primary plate-6 profile)}."""
        P = self.p['P_noise']
        indiv = {k: float(np.mean((targets[k] - replicate[k]) ** 2) / 2.0) for k in sorted(replicate)}
        noise = np.asarray([np.mean([indiv[f'{t}|{q}|2'] for t in sorted(P.target_drugs) if q in P.target_drugs[t]])
                            for q in P.drugs], np.float64)
        arms = [plane('main', a, 'tuned' if a in ('memory', 'full', 'linear') else 'fixed')
                for a in ('G', 'memory', 'full', 'linear', 'mlp256', 'k3_ext', 'samehead')]
        res = {}
        for pl in arms:
            sc = self.s.chem(pl, P, steps=(2,))
            res[pl] = dict(mse=summarize(sc[:, 0], P), noise_fraction=ratio_summary(noise, sc[:, 0], P))
        m = self.s.chem(plane('main', 'memory', 'tuned'), P, steps=(2,))
        f = self.s.chem(plane('main', 'full', 'tuned'), P, steps=(2,))
        return dict(noise=summarize(noise, P), arm_row2_mse=res,
                    reducible_error_reduction=ratio_summary(m[:, 0] - f[:, 0], m[:, 0] - noise, P),
                    chemical_noise=noise.tolist(), per_target_drug_noise=indiv)

    def transfer(self):
        P = self.p['P_TM']
        pairs = []
        for t in sorted(P.target_drugs):
            mem = plane('transfer', 'memory', 'tuned', donor=t)
            for r in sorted(P.target_drugs):
                if r == t:
                    continue
                full = plane('transfer', 'full', 'tuned', donor=f'{r}:{t}')
                a = self.s.chem(full, P, only_target=t)
                b = self.s.chem(mem, P, only_target=t)
                pairs.append(dict(donor=r, target=t,
                                  contrasts=self.contrast(full, mem, 'P_TM', only_target=t),
                                  relative_mse_reduction=ratio_summary(b[:, 0] - a[:, 0], b[:, 0], P),
                                  means=dict(full=self.means(full, 'P_TM', only_target=t),
                                             memory=self.means(mem, 'P_TM', only_target=t))))
        return dict(pairs=pairs, pair_count=len(pairs), n_chemicals=len(P.drugs))
