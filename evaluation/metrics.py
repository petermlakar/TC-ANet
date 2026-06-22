import torch
import math

import sklearn

from time import time

from typing import List, Dict, Tuple, Optional

def diebold_mariano(s1: torch.Tensor, s2: torch.Tensor, label1: str, label2: str, alpha: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    k: int = 20*5
    
    s1 = s1.flatten(start_dim = 1)
    s2 = s2.flatten(start_dim = 1)

    assert k < s1.shape[-1] and k < s2.shape[-1]

    di: torch.Tensor = s1 - s2
    dbar: torch.Tensor = di.nanmean(dim = -1, keepdim = True)

    n: torch.Tensor = (~di.isnan()).sum(dim = -1, keepdim = True)

    def autocovariance():

        T: int = di.shape[-1]

        c: torch.Tensor = ((di - dbar)*(di - dbar)).nansum(dim = -1, keepdim = True)/n

        for j in range(1, k + 1):
            
            c += 2*((di[:, :T - j] - dbar)*(di[:, j:] - dbar)).nansum(dim = -1, keepdim = True)/n
        
        return c

    sig: torch.Tensor = autocovariance().sqrt()

    t: torch.Tensor = (n.sqrt()*dbar/sig)[:, 0]

    p: torch.Tensor = 2*(1 - 0.5*(1 + torch.special.erf(t.abs()/math.sqrt(2))))
    p_sorted: torch.Tensor = p.sort()[0]

    bh_threshold: torch.Tensor = alpha*torch.linspace(1, len(t), steps = len(t))/len(t)
    bh_alpha: torch.Tensor = p_sorted[p_sorted <= bh_threshold].max()

    i: torch.Tensor = p <= bh_alpha

    m1: torch.Tensor = (t[i] < 0).sum()/len(t)
    m2: torch.Tensor = (t[i] > 0).sum()/len(t)
    nd: torch.Tensor = (~i).sum()/len(t)

    print(f'{label1}: {m1:.3f}; {label2}: {m2:.3f}; No difference: {nd:.3f}')

    return (m1, m2, nd)


def tbrier(x: torch.Tensor, y: torch.Tensor, n: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    under: float = (x <= 0).sum()/x.numel()
    print(f'{under:.10f}')

    i: torch.Tensor = ~torch.isnan(y)
    N: int = i.sum()

    C: List[torch.Tensor] = []
    X: List[torch.Tensor] = []
    Y: List[torch.Tensor] = []

    y = y[i]
    x = x[i]

    T: torch.Tensor = y[y >= 0].unique().sort()[0]
    _n: int = len(T)//15

    for t in T[::_n]:

        score: torch.Tensor = brier_score(x, y, t.item()).mean()

        #print(f'{t} {score:.8f}')

        C.append((y >= t).sum()/N)
        X.append(t)
        Y.append(score)

    return torch.tensor(C), torch.tensor(X), torch.tensor(Y)

def twcrps(x: torch.Tensor, y: torch.Tensor, n: int = 30, title: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    i: torch.Tensor = ~torch.isnan(y)
    N: int = i.sum()

    C: List[torch.Tensor] = []
    X: List[torch.Tensor] = []
    Y: List[torch.Tensor] = []

    y = y[i]
    x = x[i]

    T: torch.Tensor = y[y >= 0].unique().sort()[0]
    #T: torch.Tensor = torch.quantile(y[y > 0.1], torch.linspace(0.01, 0.999, steps = 15))

    _n: int = len(T)//15

    for k, t in enumerate(T[::_n]):

        yt: torch.Tensor = y.clone().clamp(min = t)
        xt: torch.Tensor = x.clone().clamp(min = t)

        crps: torch.Tensor = torch.cat([crps_ensemble(_x, _y) for (_x, _y) in zip(torch.chunk(xt, max(xt.shape[0]//100, 1), 0), torch.chunk(yt, max(yt.shape[0]//100, 1), 0))], dim = 0)

        C.append((y >= t).sum()/N)
        X.append(t)
        Y.append(crps.nanmean())

        print((y >= t).sum(), t, crps.nanmean(), crps.std())

    return torch.tensor(C), torch.tensor(X), torch.tensor(Y)

def binary_reliability(x: torch.Tensor, y: torch.Tensor, thr: float) -> torch.Tensor:

    # The ensemble contained in x is treated as a quantile forecast
    # We count the number of quantiles which predict that a realization is
    # going to be bellow or equal to thr. This provides us with a probability
    # of the variable realization being bellow y.
    # 
    # Then, for each unique probabilistic forecast, we look at the actual realizations.
    # The fraction of realizations when p probability was forecast should be equal.

    p: float = 1/(x.shape[-1] + 1)
    x = (x.flatten(end_dim = -2) <= thr).sum(dim = -1, keepdim = True).clamp(min = 1)*p

    y = y.flatten()[..., None] <= thr

    # 0.11 0.22 0.33 0.44 0.55 0.66 0.77 0.88 0.99 
    # 0    0    0    0    1.0  2.0  3.0  5.0  6.0
    # 1    1    1    1    0    0    0    0    0
    # 4 - 1

    p: torch.Tensor = x.unique().sort()[0]

    observed_fractions: List[torch.Tensor] = []

    for t in p:
        
        i: torch.Tensor = x == t
        observed_fractions.append(y[i].sum()/i.sum())

    return torch.stack([p, torch.tensor(observed_fractions)], dim = 0)

def roc(x: torch.Tensor, y: torch.Tensor, K: int = 100) -> torch.Tensor:

    T: torch.Tensor = (x > 0).sum(dim = -1).type(torch.float32)/x.shape[-1]

    fpr, tpr, thr = (torch.from_numpy(_x) for _x in sklearn.metrics.roc_curve((y > 0).flatten(), T.flatten()))

    res = torch.stack([tpr, fpr])

    """
    G: torch.Tensor = (y > 0)
    _G: torch.Tensor = y == 0

    P: torch.Tensor = G.sum()
    N: torch.Tensor = _G.sum()

    res: torch.Tensor = torch.zeros((2, K + 1), dtype = torch.float32)

    for j, i in enumerate(torch.cat([torch.linspace(0, 1, steps = K), torch.ones((1,), dtype = torch.float32)*1.1])):

        t: torch.Tensor = T >= i

        TPR: torch.Tensor = torch.logical_and(t,  G).sum()/P
        FPR: torch.Tensor = torch.logical_and(t, _G).sum()/N
        
        res[0, j] = TPR
        res[1, j] = FPR
    """

    return res

def brier_score(x: torch.Tensor, y: torch.Tensor, v: float = 0.0) -> torch.Tensor:
    # Now computed as probability of exceedence of threshold

    #    [0.125 0.250 0.375 0.5 0.625 0.8 0.925]
    # x: [0.0 0.0 0.0 1.0 2.0 5.0 7.0] - [1 1 1 1 1 0 0] - probability of rain less than or equal v
    # y: [3.2] | t -> 1
    # t: [3]

    p_v: torch.Tensor = (x > v).sum(dim = -1)/(x.shape[-1] + 1)
    y_v: torch.Tensor = (y > v).type(torch.float32)

    return (p_v - y_v).pow(2)

def energy_score(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    members: int = x.shape[-1]

    # y: [...]
    # x: [..., members]

    t0: torch.Tensor = torch.linalg.vector_norm((y[..., None] - x), dim = -2).mean(dim = -1)
    t1: torch.Tensor = torch.linalg.vector_norm(x[..., None] - x[..., None, :], dim = -3).sum(dim = (-1, -2))/(2*members*members)

    return t0 - t1

def variogram_create_locality_matrix(d: int, locality: int = 1) -> torch.Tensor:

    M: torch.Tensor = torch.zeros((d, d), dtype = torch.float32)

    for i in range(0, min(d, locality) + 1):
        M = torch.diagonal_scatter(M, torch.ones((d - i - 1,), dtype = torch.float32), offset =  i + 1)
        M = torch.diagonal_scatter(M, torch.ones((d - i - 1,), dtype = torch.float32), offset = -i - 1)

    return M

def variogram_score(x: torch.Tensor, y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    
    t0: torch.Tensor = (y[..., None] - y[:, None]).abs().sqrt()
    t1: torch.Tensor = (x[:, None] - x[:, :, None]).abs().sqrt().mean(dim = -1)

    V: torch.Tensor = ((t0 - t1)*M[None]).pow(2).sum(dim = (-1, -2))
    
    return V

def crps_expectation(x: torch.Tensor, y: torch.Tensor):

    t0: torch.Tensor = (y[..., None] - x).abs().nanmean(dim = -1)
    t1: torch.Tensor = 0.5*(x[..., None] - x[..., None, :]).abs().nanmean(dim = (-1, -2))

    return t0 - t1

# Continuous ranked probability score estimation based on Hersbach 2000
def crps_quantile_decomposition(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    bins: torch.Tensor = x[:, 1:] - x[:, :-1]

    #alpha: torch.Tensor = 

    exit()

    # Quantile levels for each input quantiles,
    # treating the quantiles as ensemble members with
    # equal weights, P_2 - P_1 = P_2 - P_1 = ...
    # where P denoted the for this empirical distribution
    p = torch.linspace(1.0/x.shape[-1], 1, steps = x.shape[-1], device = y.device)[:-1]

    x = x.view(s0*s1*s2, x.shape[-1])
    y = y.view(s0*s1*s2, y.shape[-1])

    # Length of the quantile intervals
    # bin[j] =  [x_2 - x_1, x_3 - x_2, ...]
    bin = x[..., 1:] - x[..., :-1]

    # Alpha values to determine bin widths
    # 0 < i < N, where N denotes the number of forecasts
    #
    # Alpha, excluding outliers, is defined as:
    #
    # y > x_(i + 1)       => alpha_i = x_(i + 1) - x_i // y is bigger than the upper bound of this bin
    # x_(i + 1) > y > x_i => alpha_i = y - x_i         // y lies inside the bin
    # y < x_i             => alpha_i = 0               // y is smaller than the lower bound of this bin
    # 
    # Example of a: a[j] = [1,   1,   1,   0,   0,   0,   0]
    #                      [x_2, x_3, x_4, x_5, x_6, x_7, x_8]
    a = y > x[..., 1:]

    # Beta, excluding outliers, is defined as:
    #
    # y > x_(i + 1)       => beta_i = 0               // y is bigger than the upper bound of this bin
    # x_(i + 1) > y > x_i => beta_i = x_(i + 1) - y   // y lies inside the bin
    # y < x_i             => beta_i = x_(i + 1) - x_i // y is smaller than the lower bound of this bin
    # 
    # Example of a: b[j] = [0,   0,   0,   0,   1,   1,   1]
    #                      [x_1, x_2, x_3, x_4, x_5, x_6, x_7]
    b = y < x[..., :-1]

    # Find all the bins that contain y
    # ~a[j] and ~b[j] = [0,   0,   0,   1,   0,   0,   0]
    # In case of an outlier this becomes a zero vector, while either a or b contains all zeros and the remaining all ones
    j = torch.logical_and(~a, ~b)

    a = a.type(torch.float32)
    b = b.type(torch.float32)
    j = j.type(torch.float32)

    # Add all the bin contributions where y lies completelly outside the bin and is not an outlier
    #
    # c[j] = a_1*(x_2 - x_1)*p_1^2 + a_2*(x_3 - x_2)*p_2^2 + a_3*(x_4 - x_3)*p_3^2 + a_4*(x_5 - x_4)*p_4^2 + ... +
    #        b_5*(x_6 - x_5)*p_5^2 + b_6*(x_7 - x_6)*p_6^2 + b_7*(x_8 - x_7)*p_7^2
    c = (bin*a*p.pow(2) + bin*b*(1.0 - p).pow(2)).sum(dim = -1)

    # We handle the case where y lies inside an interval
    c += (j*((y - x[..., :-1])*p.pow(2) + (x[..., 1:] - y)*(1.0 - p).pow(2))).sum(dim = -1)

    # Finall, add outlier contributions
    c += (x[..., 0] - y[..., 0])*b[..., 0] + (y[..., 0] - x[..., -1])*a[..., -1]

    return c.view(s0, s1, s2)



# Continuous ranked probability score estimation based on Hersbach 2000
def crps_ensemble(x: torch.Tensor, y: torch.Tensor, t: Optional[torch.Tensor] = None):

    # Expectation formula
    #t0 = np.abs(x - y[..., None]).mean(axis = -1)
    #t1 = np.abs(0.5*(x[..., None] - x[..., None, :])).sum(axis = (-1, -2))/(x.shape[-1]*(x.shape[-1] - 1))

    y = y[..., None]

    # Quantile levels for each input quantiles,
    # treating the quantiles as ensemble members with
    # equal weights, P_2 - P_1 = P_2 - P_1 = ...
    # where P denoted the for this empirical distribution
    p = torch.linspace(1.0/x.shape[-1], 1, steps = x.shape[-1], device = y.device)[:-1]

    # Length of the quantile intervals
    # bin[j] =  [x_2 - x_1, x_3 - x_2, ...]
    bin = x[..., 1:] - x[..., :-1]

    # Alpha values to determine bin widths
    # 0 < i < N, where N denotes the number of forecasts
    #
    # Alpha, excluding outliers, is defined as:
    #
    # y > x_(i + 1)       => alpha_i = x_(i + 1) - x_i // y is bigger than the upper bound of this bin
    # x_(i + 1) > y > x_i => alpha_i = y - x_i         // y lies inside the bin
    # y < x_i             => alpha_i = 0               // y is smaller than the lower bound of this bin
    # 
    # Example of a: a[j] = [1,   1,   1,   0,   0,   0,   0]
    #                      [x_2, x_3, x_4, x_5, x_6, x_7, x_8]
    a = y > x[..., 1:]

    # Beta, excluding outliers, is defined as:
    #
    # y > x_(i + 1)       => beta_i = 0               // y is bigger than the upper bound of this bin
    # x_(i + 1) > y > x_i => beta_i = x_(i + 1) - y   // y lies inside the bin
    # y < x_i             => beta_i = x_(i + 1) - x_i // y is smaller than the lower bound of this bin
    # 
    # Example of a: b[j] = [0,   0,   0,   0,   1,   1,   1]
    #                      [x_1, x_2, x_3, x_4, x_5, x_6, x_7]
    b = y < x[..., :-1]

    # Find all the bins that contain y
    # ~a[j] and ~b[j] = [0,   0,   0,   1,   0,   0,   0]
    # In case of an outlier this becomes a zero vector, while either a or b contains all zeros and the remaining all ones
    j = torch.logical_and(~a, ~b)

    a = a.type(torch.float32)
    b = b.type(torch.float32)
    j = j.type(torch.float32)

    # Add all the bin contributions where y lies completelly outside the bin and is not an outlier
    #
    # c[j] = a_1*(x_2 - x_1)*p_1^2 + a_2*(x_3 - x_2)*p_2^2 + a_3*(x_4 - x_3)*p_3^2 + a_4*(x_5 - x_4)*p_4^2 + ... +
    #        b_5*(x_6 - x_5)*p_5^2 + b_6*(x_7 - x_6)*p_6^2 + b_7*(x_8 - x_7)*p_7^2

    #t0: torch.Tensor = bin*a*p.pow(2) + bin*b*(1.0 - p).pow(2)
    #t1: torch.Tensor = j*((y - x[..., :-1])*p.pow(2) + (x[..., 1:] - y)*(1.0 - p).pow(2))
    #t2: torch.Tensor = (x[..., 0] - y[..., 0])*b[..., 0] + (y[..., 0] - x[..., -1])*a[..., -1]

    c = (bin*a*p.pow(2) + bin*b*(1.0 - p).pow(2)).sum(dim = -1)

    # We handle the case where y lies inside an interval
    c += (j*((y - x[..., :-1])*p.pow(2) + (x[..., 1:] - y)*(1.0 - p).pow(2))).sum(dim = -1)

    # Finall, add outlier contributions
    c += (x[..., 0] - y[..., 0])*b[..., 0] + (y[..., 0] - x[..., -1])*a[..., -1]

    return c

def quantile_loss(q: torch.Tensor, y: torch.Tensor):

    y = y[..., None]

    ql = y - q

    #t = torch.linspace(0.01, 0.99, steps = q.shape[-1])
    t = torch.arange(1/(q.shape[-1] + 1), 1.0, step = 1/(q.shape[-1] + 1))
    t = torch.tile(t, (ql.shape[0], ql.shape[1], ql.shape[2], 1))

    i0 = ql <  0.0
    i1 = ql >= 0.0

    ql[i0] *= t[i0] - 1.0
    ql[i1] *= t[i1]

    return ql

def rank_histogram(r: torch.Tensor, y: torch.Tensor, edges_prob_mass: float = None, censored: bool = True):

    y_valid = ~y.isnan()

    c = torch.zeros(r.shape[-1] + 1, dtype = torch.float32)

    if not censored:

        c[0]  = (y <= r[..., 0])[y_valid].sum()
        c[-1] = (y >  r[..., -1])[y_valid].sum()

        for i in range(1, r.shape[-1]):
            c[i] = torch.logical_and(y > r[..., i - 1], y <= r[..., i])[y_valid].sum()

        c_tot = y_valid.sum()

    else:

        r[r < 0.0] = 0.0
        y_0 = torch.logical_and(y_valid, y <= 0.0)
        y_p = torch.logical_and(y_valid, y >  0.0)

        if y_0.any():
            
            _c = (y[..., None] >= r).sum(axis = -1)[y_0]

            c[0] += (_c == 0).sum() # Outliers when forecast value was > 0

            # _c == 1 means that y was bigger or equal than r[0] and r[1], since y is 0 r[0] == r[1] == 0, do random assignment
            for i in _c.unique():
               
                i = i.item()

                if i == 0:
                    continue

                assign: torch.Tensor = torch.randint(low = 0, high = i + 1, size = ((_c == i).sum().item(),))
                for j in assign.unique():
                    c[j] += (assign == j).sum()

        if y_p.any():

            c[0]  += (y <= r[...,  0])[y_p].sum()
            c[-1] += (y >  r[..., -1])[y_p].sum()

            idx: torch.Tensor = torch.logical_and((y > r[..., -1]).any(axis = -1), y_p.any(axis = -1))

            for i in range(1, r.shape[-1]):
                c[i] += torch.logical_and(y > r[..., i - 1], y <= r[..., i])[y_p].sum()

    return c/y_valid.sum() 

def coverage(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    t0: torch.Tensor = torch.logical_or((x < y[..., None]).all(dim = -1), (x > y[..., None]).all(dim = -1)).sum(dim = (0, 1))
    t1: torch.Tensor = (~(y.isnan())).sum(dim = (0, 1))

    return (1 - t0/t1)/(1 - 2*1/x.shape[-1])

def sharpness_composite(x: torch.Tensor, p: float = 0.5, estimate_quantiles: bool = False):
    
    if estimate_quantiles:
        x = x.quantile(q = torch.linspace(1/20, 1 - 1/20, steps = 19), dim = -1, keepdim = True).swapaxes(0, -1)[0]

    N: int = x.shape[-1]
    i: float = 1.0/(1.0 + N)

    y: int = int(math.ceil(p/i))

    d: torch.Tensor = (x[..., 1:] - x[..., :-1]).sort(dim = -1)[0][..., :y].sum(dim = -1).flatten()
    #d = d[~d.isnan()]

    return d

