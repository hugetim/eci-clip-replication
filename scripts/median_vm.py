"""Median-centred XB with variance-matched capability noise: the improvement on XB that the post's last footnote
proposes. Used by bayes_eci.py (--median-centre, --var-match) and by the scoring scripts.

Two changes to XB's latent beta, x ~ Beta(a, b) with a + b = kappa:

1. Median-centring. Epoch's curve mu = sigmoid(slope_b (C_m - D_b)) sets the beta's median, I_mu(a, kappa - a) = 1/2
   (I = the regularized incomplete beta function), instead of its mean (a = mu kappa). So the curve is the latent score
   a model beats half the time. No extra parameter. There is no closed form for a: solve_median finds it by Newton's
   method with a bisection fallback (exact to ~1e-15), and the model uses a smooth table of those roots, herm_a
   (bicubic Hermite interpolation in logit(mu) and log(kappa); within 3e-9 of the exact roots), which PyTensor can
   evaluate and differentiate cheaply.

2. Capability noise, by variance matching. A model's capability on a benchmark varies around its ECI, normal with sd
   tau (one parameter, shared). Pushed through the curve, that adds score variance ~ (slope_b tau)^2 (mu (1 - mu))^2,
   which shrinks toward 0 and 1 faster than the beta's own 4 sigma_b^2 mu (1 - mu), and more on steeper benchmarks.
   The beta keeps its form with the summed variance: kappa + 1 = 1 / s2, s2 = 4 sigma_b^2 + (slope_b tau)^2 mu (1 - mu)
   (kappa_formula; slope_b and tau on the raw capability scale, where their product equals the ECI-scale one). s2
   is capped smoothly at 0.7, which leaves plausible values unchanged (to 2e-8 up to s2 = 0.4; fits reach ~0.2).
   Averaging the likelihood over the capability noise by quadrature instead gives essentially the same fit at ~20x
   the cost.
"""
import numpy as np

# ---- variance-matched kappa
S2_MAX, SHARP = 0.7, 50.0  # smooth cap on s2 (soft minimum): kappa >= 1 / 0.7 - 1 = 0.43


def kappa_formula(sigma, slope_raw, tau, mu, xp=np):
    s2 = 4 * sigma ** 2 + (slope_raw * tau) ** 2 * mu * (1 - mu)
    s2 = -xp.logaddexp(-SHARP * s2, -SHARP * S2_MAX) / SHARP
    return 1 / s2 - 1


# ---- median-centring: exact roots
H = 1e-5  # relative step for central differences of betainc in a and b


def solve_median(mu, kappa, n_newton=40, n_bisect=200, tol=1e-12):
    """a with I_mu(a, kappa - a) = 1/2, by Newton's method in v = logit(a / kappa) (dF/dv from central differences of
    betainc; steps capped at +-2), from the mean-centred a = mu kappa; scores not converged are finished by bisection."""
    from scipy.special import betainc, expit, logit
    mu, kappa = np.broadcast_arrays(np.asarray(mu, float), np.asarray(kappa, float))
    v = logit(mu)
    for _ in range(n_newton):
        a = kappa * expit(v); b = kappa - a
        F = betainc(a, b, mu) - 0.5
        if np.abs(F).max() < tol:
            break
        Ia = (betainc(a * (1 + H), b, mu) - betainc(a * (1 - H), b, mu)) / (2 * H * a)
        Ib = (betainc(a, b * (1 + H), mu) - betainc(a, b * (1 - H), mu)) / (2 * H * b)
        dF = (Ia - Ib) * a * b / kappa  # < 0: the median rises with a
        v = v - np.clip(F / np.where(dF < 0, dF, -1e-300), -2.0, 2.0)
    a = kappa * expit(v)
    bad = np.abs(betainc(a, kappa - a, mu) - 0.5) >= tol
    if bad.any():
        m, k = mu[bad], kappa[bad]
        lo, hi = np.full(m.shape, -40.0), np.full(m.shape, 40.0)
        for _ in range(n_bisect):
            mid = (lo + hi) / 2; f = betainc(k * expit(mid), k - k * expit(mid), m) - 0.5
            lo, hi = np.where(f > 0, mid, lo), np.where(f > 0, hi, mid)
        v = v.copy(); v[bad] = (lo + hi) / 2; a = kappa * expit(v)
    return a


# ---- median-centring: the table the model uses
TABLE_U = (-13.9, 13.9, 557)                  # logit(mu), mu in [1e-6, 1 - 1e-6] (logit +-13.82)
TABLE_W = (np.log(0.3), np.log(5000.0), 197)  # log(kappa)
_HERM = None


def _herm_tables():
    """z = logit(a / kappa) at the knots (exact roots), and z_u, z_w, z_uw from a bicubic spline through them."""
    global _HERM
    if _HERM is None:
        from scipy.interpolate import RectBivariateSpline
        from scipy.special import expit, logit
        u = np.linspace(*TABLE_U); w = np.linspace(*TABLE_W)
        U, W = np.meshgrid(u, w, indexing="ij")
        z = logit(solve_median(expit(U.ravel()), np.exp(W.ravel())).reshape(U.shape) / np.exp(W))
        s = RectBivariateSpline(u, w, z, kx=3, ky=3, s=0)
        T = [s(u, w, dx=dx, dy=dy) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))]
        _HERM = dict(u0=u[0], w0=w[0], hu=u[1] - u[0], hw=w[1] - w[0], nu=len(u), nw=len(w),
                     Z=T[0].ravel(), Zu=T[1].ravel(), Zw=T[2].ravel(), Zuw=T[3].ravel())
    return _HERM


def herm_z(u, w, xp=np):
    """z(u, w) by bicubic Hermite interpolation of the table (u, w outside it are clamped to its edges)."""
    T = _herm_tables()
    if xp is np:
        floor, clip, take, as_int = np.floor, np.clip, (lambda t, i: t[i]), (lambda x: x.astype(int))
    else:
        import pytensor.tensor as pt
        floor, clip, as_int = pt.floor, pt.clip, (lambda x: pt.cast(x, "int64"))
        take = lambda t, i: pt.as_tensor_variable(t)[i]
    hu, hw = T["hu"], T["hw"]
    u = clip(u, T["u0"], T["u0"] + hu * (T["nu"] - 1)); w = clip(w, T["w0"], T["w0"] + hw * (T["nw"] - 1))
    fi = clip(floor((u - T["u0"]) / hu), 0, T["nu"] - 2); fj = clip(floor((w - T["w0"]) / hw), 0, T["nw"] - 2)
    t = (u - T["u0"]) / hu - fi; s = (w - T["w0"]) / hw - fj
    i, j = as_int(fi), as_int(fj)
    h0 = lambda x: (2 * x - 3) * x * x + 1; h1 = lambda x: ((x - 2) * x + 1) * x  # value / slope basis at 0
    g0 = lambda x: (3 - 2 * x) * x * x; g1 = lambda x: (x - 1) * x * x           # value / slope basis at 1
    bu = ((h0(t), h1(t)), (g0(t), g1(t))); bw = ((h0(s), h1(s)), (g0(s), g1(s)))
    z = 0
    for di in (0, 1):
        for dj in (0, 1):
            k = (i + di) * T["nw"] + (j + dj)
            z = z + (take(T["Z"], k) * bu[di][0] * bw[dj][0] + hu * take(T["Zu"], k) * bu[di][1] * bw[dj][0]
                     + hw * take(T["Zw"], k) * bu[di][0] * bw[dj][1] + hu * hw * take(T["Zuw"], k) * bu[di][1] * bw[dj][1])
    return z


def herm_a(mu, kappa, xp=np):
    """Median-centred a(mu, kappa) from the table (numpy or PyTensor)."""
    if xp is np:
        from scipy.special import expit
        mu, kappa = np.asarray(mu, float), np.asarray(kappa, float)
        return kappa * expit(herm_z(np.log(mu) - np.log1p(-mu), np.log(kappa)))
    import pytensor.tensor as pt
    return kappa * pt.sigmoid(herm_z(pt.log(mu) - pt.log1p(-mu), pt.log(kappa), xp=pt))


# ---- for the scoring scripts
def beta_params(dr, diag, keep, bench_idx, sigma_s, slope, mu):
    """The latent beta's (a, b) per (draw, score), as the fit's model sets them. dr, diag: the fit's draws.npz and
    diagnostics.json; keep: the draws used; sigma_s: sigma_b per score (draws x scores); slope: ECI-scale slopes
    (draws x benchmarks); mu: the curve per score (draws x scores)."""
    kappa = 1 / (4 * sigma_s ** 2) - 1
    if diag.get("var_match"):
        b_scale = dr["b"][keep].astype(float)[:, None]  # ECI points per raw unit: raw slope = ECI slope * b
        tau = dr["hyper_tau_cap"][keep].astype(float)[:, None]
        kappa = kappa_formula(sigma_s, slope[:, bench_idx] * b_scale, tau, mu)
    a = herm_a(mu, kappa) if diag.get("median_centre") else mu * kappa
    return a, kappa - a
