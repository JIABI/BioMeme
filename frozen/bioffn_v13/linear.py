"""Full-output linear and frozen random-feature residual maps, without data I/O."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from bioffn_v7.baselines import group_weights, _matrix


@dataclass
class FeatureMap:
    mean: np.ndarray
    scale: np.ndarray
    weight: object
    bias: object

    @classmethod
    def fit(cls, x, groups, width=None, seed=20261008):
        x = _matrix(x, 'features')
        if width is None:
            return cls(np.zeros(x.shape[1]), np.ones(x.shape[1]), None, None)
        w = group_weights(groups)
        mean = w @ x
        scale = np.sqrt(w @ (x-mean)**2)
        scale = np.where(scale > 1e-12, scale, 1.)
        rng = np.random.default_rng(seed)
        return cls(mean, scale, rng.standard_normal((x.shape[1], width))/np.sqrt(x.shape[1]),
                   rng.standard_normal(width)*.5)

    def transform(self, x):
        z = (_matrix(x, 'query features')-self.mean)/self.scale
        if self.weight is None:
            return z
        t = z @ self.weight + self.bias
        return .5*t*(1.+np.tanh(np.sqrt(2./np.pi)*(t+.044715*t**3)))


@dataclass
class RidgeMap:
    feature_map: FeatureMap
    train_features: np.ndarray
    coefficients: np.ndarray
    feature_mean: np.ndarray
    residual_mean: np.ndarray
    audit: dict

    def predict(self, x):
        z = self.feature_map.transform(x)-self.feature_mean
        return self.residual_mean + (z @ self.train_features.T) @ self.coefficients


def fit_path(x, residual, groups, lambdas, *, width=None, seed=20261008):
    """Unpenalized chemical-equal intercept; loss uses n*g sum convention."""
    x, residual = _matrix(x, 'features'), _matrix(residual, 'residual', len(x))
    lam = np.asarray(lambdas, float)
    if lam.ndim != 1 or not len(lam) or np.any(lam <= 0) or not np.isfinite(lam).all():
        raise ValueError('Positive finite ridge grid required')
    fmap = FeatureMap.fit(x, groups, width, seed)
    z = fmap.transform(x)
    w = group_weights(groups)
    zm, rm = w @ z, w @ residual
    z = z-zm
    sw = np.sqrt(len(x)*w)
    weighted = sw[:, None]*z
    values, vectors = eigh(weighted @ weighted.T)
    values = np.maximum(values, 0.)
    right = vectors.T @ (sw[:, None]*(residual-rm))
    result = {}
    for value in lam:
        alpha = sw[:, None]*(vectors @ (right/(values[:, None]+value)))
        result[float(value)] = RidgeMap(fmap, z, alpha, zm, rm, dict(lambda_value=float(value),
            width=width, seed=int(seed), full_output_features=residual.shape[1],
            output_compression=False, first_layer_trained=False,
            objective='sum_i n*g_i ||R_i - intercept - f(x_i)||^2 + lambda ||W||^2'))
    return result
