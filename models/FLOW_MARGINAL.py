import torch
import torch.nn as nn
import torch.jit as jit
import math
from math import sqrt, log

from typing import Optional, Tuple, List

class SplineBlock(nn.Module):

    """
    Rational cubic spline transformation with support for forward
    evaluation, inverse evaluation, and derivative computation.

    The spline is defined by knot locations, knot values, and knot
    derivatives and is extended linearly outside the boundary knots.
    """

    def  __init__(self):
        """
        Initialize the spline block.
        """
        super().__init__()

    def forward(self, x: torch.Tensor, _t: torch.Tensor, _y: torch.Tensor, _d: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the spline at the specified input locations.

        Args:
            x: Input values.
            _t: Knot locations.
            _y: Knot values.
            _d: Knot derivatives.

        Returns:
            Spline-interpolated values.
        """

        # x:    [batch*lead, 1, 1]
        # prm_: [batch*lead, 1, number_of_knots]

        i = (x > _t).sum(dim = -1)
        k0 = i - 1
        k1 = i

        eq0 = i == 0
        eq1 = i == _t.shape[-1]

        k0[eq0] = 0
        k1[eq0] = 1

        k0[eq1] = _t.shape[-1] - 2
        k1[eq1] = _t.shape[-1] - 1

        idx = torch.logical_not(torch.logical_or(eq0, eq1))

        k0 = torch.unsqueeze(k0, dim = 1)
        k1 = torch.unsqueeze(k1, dim = 1)

        t0 = torch.squeeze(torch.gather(_t, -1, k0), dim = 1)
        t1 = torch.squeeze(torch.gather(_t, -1, k1), dim = 1)

        y0 = torch.squeeze(torch.gather(_y, -1, k0), dim = 1)
        y1 = torch.squeeze(torch.gather(_y, -1, k1), dim = 1)

        d0 = torch.squeeze(torch.gather(_d, -1, k0), dim = 1)
        d1 = torch.squeeze(torch.gather(_d, -1, k1), dim = 1)

        _t0 = t0[idx]
        _t1 = t1[idx]

        _y0 = y0[idx]
        _y1 = y1[idx]

        _d0 = d0[idx]
        _d1 = d1[idx]

        delta_t = _t1 - _t0
        delta_y = _y1 - _y0

        s = delta_y/delta_t

        x = torch.squeeze(x, dim = -1)
        e = (x[idx] - _t0)/delta_t

        n0 = delta_y*(s*e*e + _d0*e*(1.0 - e))
        n1 = s + (_d1 + _d0 - 2.0*s)*e*(1.0 - e)

        p = torch.clone(x)

        if idx.sum() > 0:
            p[idx] = _y0 + n0/n1

        # Compute asimptotics

        a0 = d0
        b0 = y0 - a0*t0

        a1 = d1
        b1 = y1 - a1*t1

        if eq0.sum() > 0:
            p[eq0] = a0[eq0]*x[eq0] + b0[eq0]
        if eq1.sum() > 0:
            p[eq1] = a1[eq1]*x[eq1] + b1[eq1]

        return torch.unsqueeze(p, dim = -1)

    @jit.export
    def backward(self, y: torch.Tensor, _t: torch.Tensor, _y: torch.Tensor, _d: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse spline transformation.

        Args:
            y: Output values.
            _t: Knot locations.
            _y: Knot values.
            _d: Knot derivatives.

        Returns:
            Input values corresponding to the supplied outputs.
        """
        
        # y:    [batch*lead, quantiles, 1]
        # prm_: [batch*lead, 1, number_of_knots]

        i = (y > _y).sum(dim = -1)
        k0 = i - 1
        k1 = i

        eq0 = i == 0
        eq1 = i == _y.shape[-1]

        k0[eq0] = 0
        k1[eq0] = 1

        k0[eq1] = _y.shape[-1] - 2
        k1[eq1] = _y.shape[-1] - 1

        idx = torch.logical_not(torch.logical_or(eq0, eq1))

        k0 = torch.unsqueeze(k0, dim = 1)
        k1 = torch.unsqueeze(k1, dim = 1)

        t0 = torch.squeeze(torch.gather(_t, -1, k0), dim = 1)
        t1 = torch.squeeze(torch.gather(_t, -1, k1), dim = 1)

        y0 = torch.squeeze(torch.gather(_y, -1, k0), dim = 1)
        y1 = torch.squeeze(torch.gather(_y, -1, k1), dim = 1)

        d0 = torch.squeeze(torch.gather(_d, -1, k0), dim = 1)
        d1 = torch.squeeze(torch.gather(_d, -1, k1), dim = 1)

        _t0 = t0[idx]
        _t1 = t1[idx]

        _y0 = y0[idx]
        _y1 = y1[idx]

        _d0 = d0[idx]
        _d1 = d1[idx]

        delta_t = _t1 - _t0
        delta_y = _y1 - _y0

        s = delta_y/delta_t
        o = _d1 + _d0 - 2.0*s

        y = torch.squeeze(y, dim = -1)

        df = y[idx] - _y0
        a = delta_y*(s - _d0) + df*o
        b = delta_y*_d0 - df*o
        c = -s*df

        if idx.sum() > 0:
            y[idx] = 2.0*c*delta_t/(-b - torch.sqrt(b*b - 4.0*a*c)) + _t0

        # Compute asimptotics

        a0 = d0
        b0 = y0 - a0*t0

        a1 = d1
        b1 = y1 - a1*t1

        if eq0.sum() > 0:
            y[eq0] = (y[eq0] - b0[eq0])/a0[eq0]
        if eq1.sum() > 0:
            y[eq1] = (y[eq1] - b1[eq1])/a1[eq1]

        return torch.unsqueeze(y, dim = -1)

    @jit.export
    def dt(self, x: torch.Tensor, _t: torch.Tensor, _y: torch.Tensor, _d: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the derivative of the spline with respect to x.

        Args:
            x: Input values.
            _t: Knot locations.
            _y: Knot values.
            _d: Knot derivatives.

        Returns:
            First derivative of the spline.
        """
        # x:    [batch*lead, 1, 1]
        # prm_: [batch*lead, 1, number_of_knots]
       
        i = (x > _t).sum(dim = -1)
        k0 = i - 1
        k1 = i

        eq0 = i == 0
        eq1 = i == _t.shape[-1]

        k0[eq0] = 0
        k1[eq0] = 1

        k0[eq1] = _t.shape[-1] - 2
        k1[eq1] = _t.shape[-1] - 1

        idx = torch.logical_not(torch.logical_or(eq0, eq1))

        k0 = torch.unsqueeze(k0, dim = 1)
        k1 = torch.unsqueeze(k1, dim = 1)

        t0 = torch.squeeze(torch.gather(_t, -1, k0), dim = 1)
        t1 = torch.squeeze(torch.gather(_t, -1, k1), dim = 1)

        y0 = torch.squeeze(torch.gather(_y, -1, k0), dim = 1)
        y1 = torch.squeeze(torch.gather(_y, -1, k1), dim = 1)

        d0 = torch.squeeze(torch.gather(_d, -1, k0), dim = 1)
        d1 = torch.squeeze(torch.gather(_d, -1, k1), dim = 1)

        _t0 = t0[idx]
        _t1 = t1[idx]

        _y0 = y0[idx]
        _y1 = y1[idx]

        _d0 = d0[idx]
        _d1 = d1[idx]

        delta_t = _t1 - _t0
        delta_y = _y1 - _y0
        s = delta_y/delta_t

        x = torch.squeeze(x, dim = -1)
        e = (x[idx] - _t0)/delta_t

        a0 = delta_y*s
        a1 = delta_y*_d0
        b = _d1 + _d0 - 2.0*s
        
        p = torch.clone(x)

        if idx.sum() > 0:
            
            div0 = torch.pow(s, 2)*(_d1*torch.pow(e, 2) + 2.0*s*e*(1.0 - e) + _d0*torch.pow(1.0 - e, 2))
            div1 = torch.pow(s + b*e*(1.0 - e), 2)

            p[idx] = div0/div1

        if eq0.sum() > 0:
            p[eq0] = d0[eq0]

        if eq1.sum() > 0:
            p[eq1] = d1[eq1]

        return torch.unsqueeze(p, dim = -1)

class Normal(nn.Module):
    """
    Standard normal distribution helper class providing sampling,
    quantile, log-density, and cumulative distribution functions.
    """

    def __init__(self):
        """
        Initialize the standard normal distribution helper.
        """
        super().__init__()

        self._a = nn.Parameter(torch.tensor(0.0, dtype = torch.float32), requires_grad = False)
        self.type: int = 0

    def sample(self, number_of_samples: int) -> torch.Tensor:
        """
        Draw samples from a standard normal distribution.

        Args:
            number_of_samples: Number of samples to generate.

        Returns:
            Tensor containing random samples.
        """
        return torch.randn(number_of_samples, dtype = torch.float32, device = self._a.device)

    def quantile(self, number_of_quantiles: int, percentile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute quantiles of the standard normal distribution.

        Args:
            number_of_quantiles: Number of evenly spaced quantiles to
                generate when percentile is not provided.
            percentile: Optional tensor of percentile values in the
                interval [0, 1].

        Returns:
            Tensor containing the corresponding quantile values.
        """

        if percentile is None:
            percentile = torch.arange(1.0/(number_of_quantiles + 1), 1.0, step = 1.0/(number_of_quantiles + 1), dtype = torch.float32, device = self._a.device)

        return sqrt(2.0)*torch.erfinv(2.0*percentile - 1.0)

    def lpdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability density function.

        Args:
            x: Input values.

        Returns:
            Log-density evaluated at x.
        """
        return -0.5*log(2.0*math.pi) - 0.5*x.pow(2)

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the cumulative distribution function.

        Args:
            x: Input values.

        Returns:
            Cumulative probability evaluated at x.
        """
        return 0.5 + 0.5*torch.erf(x/sqrt(2.0))
        
class Logistic(nn.Module):
    """
    Standard logistic distribution helper class providing sampling,
    quantile, log-density, and cumulative distribution functions.
    """

    def __init__(self):
        """
        Initialize the standard logistic distribution helper.
        """

        super().__init__()

        self._a = nn.Parameter(torch.tensor(0.0, dtype = torch.float32), requires_grad = False)
        self.type: int = 1

    def sample(self, number_of_samples: int) -> torch.Tensor:
        """
        Draw samples from a standard logistic distribution.

        Samples are generated by applying the inverse logistic CDF
        (logit transform) to uniformly distributed random values.

        Args:
            number_of_samples: Number of samples to generate.

        Returns:
            Tensor containing random samples.
        """
        
        samples: torch.Tensor = torch.rand(number_of_samples, dtype = torch.float32, device = self._a.device).clamp(min = 1e-10, max = 1.0 - 1e-10)

        return (samples/(1.0 - samples)).log()

    def quantile(self, number_of_quantiles: int, percentile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute quantiles of the standard logistic distribution.

        Args:
            number_of_quantiles: Number of evenly spaced quantiles to
                generate when percentile is not provided.
            percentile: Optional tensor of percentile values in the
                interval [0, 1].

        Returns:
            Tensor containing the corresponding quantile values.
        """

        if percentile is None:
            percentile = torch.arange(1.0/(number_of_quantiles + 1), 1.0, step = 1.0/(number_of_quantiles + 1), dtype = torch.float32, device = self._a.device)

        return (percentile/(1.0 - percentile)).log() 

    def lpdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability density function.

        Args:
            x: Input values.

        Returns:
            Log-density evaluated at x.
        """
        return -x - 2.0*(1.0 + (-x).exp()).log()

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the cumulative distribution function.

        Args:
            x: Input values.

        Returns:
            Cumulative probability evaluated at x.
        """
        return 1.0/(1.0 + (-x).exp())

class Student(nn.Module):
    """
    Student's t distribution helper class providing sampling,
    quantile evaluation, log-density evaluation, cumulative
    distribution evaluation, and incomplete beta function utilities.
    """

    def __init__(self):
        """
        Initialize the Student's t distribution helper.
        """

        super().__init__()

        self._a = nn.Parameter(torch.tensor(0.0, dtype = torch.float32), requires_grad = False)
        self.type: int = 2

    def sample(self, number_of_samples: int, v: torch.Tensor) -> torch.Tensor:
        """
        Draw samples from a Student's t distribution.

        Samples are generated using the representation of a
        Student's t random variable as a standard normal variable
        divided by the square root of a scaled chi-squared variable.

        Args:
            number_of_samples: Number of samples to generate.
            v: Degrees-of-freedom parameters.

        Returns:
            Tensor containing random samples.
        """

        Z: torch.Tensor = torch._standard_gamma(0.5*v)/0.5
        X: torch.Tensor = torch.randn((len(v), number_of_samples), dtype = torch.float32, device = v.device)
       
        return X*(torch.rsqrt(Z/v)[:, None])

    def quantile(self, number_of_quantiles: int, v: torch.Tensor, percentile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute quantiles of the Student's t distribution.

        Quantiles are obtained through inversion of the regularized
        incomplete beta function.

        Args:
            number_of_quantiles: Number of quantiles to generate.
            v: Degrees-of-freedom parameters.
            percentile: Optional percentile values in the interval
                [0, 1].

        Returns:
            Tensor containing the corresponding quantile values.
        """
        
        s0: int = v.shape[0]

        percentile = torch.arange(1.0/(number_of_quantiles + 1), 1.0, step = 1.0/(number_of_quantiles + 1), dtype = torch.float32, device = self._a.device)[None] if percentile is None else percentile

        percentile = percentile.expand(v.shape[0], -1).flatten()
        v = v[..., None].expand(-1, number_of_quantiles).flatten()

        half: torch.Tensor = torch.tensor(0.5, dtype = torch.float32, device = v.device)

        p_lq: torch.Tensor = percentile <= 0.5
        p_gq: torch.Tensor = percentile >  0.5

        percentile[p_lq] = self.iibf(2.0*percentile[p_lq], v[p_lq]*0.5, half)
        percentile[p_gq] = self.iibf(2.0*(1.0 - percentile[p_gq]), v[p_gq]*0.5, half)

        percentile[p_lq] = -(v[p_lq]/percentile[p_lq] - v[p_lq]).sqrt()
        percentile[p_gq] =  (v[p_gq]/percentile[p_gq] - v[p_gq]).sqrt()

        return percentile.view(s0, number_of_quantiles)

    def lpdf(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability density function.

        Args:
            x: Input values.
            v: Degrees-of-freedom parameters.

        Returns:
            Log-density evaluated at x.
        """
        return torch.lgamma((v + 1.0)/2.0) - torch.lgamma(v/2.0) - 0.5*(math.pi*v).log() - (1 + x.pow(2)/v).log()*(v + 1.0)/2.0

    def cdf(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the cumulative distribution function.

        Args:
            x: Input values.
            v: Degrees-of-freedom parameters.

        Returns:
            Cumulative probability evaluated at x.
        """
       
        batch, samples = x.shape[0], x.shape[1]

        x = x.view(batch*samples)
        v = v[:, None].expand(-1, samples).view(batch*samples)
        half: torch.Tensor = torch.tensor(0.5, dtype = torch.float32, device = x.device)

        F: torch.Tensor = torch.zeros_like(x)

        x_lq: torch.Tensor = x <= 0.0
        x_gq: torch.Tensor = x > 0.0

        _x: torch.Tensor = v/(v + x.pow(2))

        if x_gq.any():
            F[x_gq] = 1.0 - 0.5*self.ibf(_x[x_gq], v[x_gq]*0.5, half)
        
        if x_lq.any():
            F[x_lq] = 0.5*self.ibf(_x[x_lq], v[x_lq]*0.5, half)

        return F

    def iibf(self, p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, N: int = 100, E: float = 1e-6) -> torch.Tensor:
        """
        Compute the inverse regularized incomplete beta function.

        Uses iterative refinement to solve for x such that
        I_x(a, b) = p.

        Args:
            p: Target cumulative probabilities.
            a: First shape parameter.
            b: Second shape parameter.
            N: Maximum number of iterations.
            E: Convergence tolerance.

        Returns:
            Inverse incomplete beta values.
        """
        
        x: torch.Tensor = torch.zeros_like(p)
        x[p >= 1.0] = 1.0

        i0: torch.Tensor = torch.logical_and(a >= 1.0, b >= 1.0)
        i1: torch.Tensor = ~i0
        
        if i0.any():

            pp: torch.Tensor = p[i0]
            pp[pp >= 0.5] = 1.0 - pp

            t: torch.Tensor = (-2.0*pp.log()).sqrt()
            x[i0] = (2.30753 + t*0.27061)/(1.0 + t*(0.99229 + t*0.04481)) - t
            x[p[i0] < 0.5] *= -1.0

            al: torch.Tensor = (x[i0].sqrt() - 3.0)/6.0
            h: torch.Tensor = 2.0/(1.0/(2.0*a[i0] - 1.0) + 1.0/(2.0*b - 1.0))
            w: torch.Tensor = (x[i0]*(al + h).sqrt()/h) - (1.0/(2.0*b - 1.0) - 1.0/(2.0*a[i0] - 1.0))*(al + 5.0/6.0 - 2.0/(3.0*h))
            x[i0] = a[i0]/(a[i0] + b*(2.0*w).exp())

        if i1.any():

            lna: torch.Tensor = (a[i1]/(a[i1] + b)).log()
            lnb: torch.Tensor = (b/(a[i1] + b)).log()

            t: torch.Tensor = (a[i1]*lna).exp()/a[i1]
            u: torch.Tensor = (b*lnb).exp()/b

            w: torch.Tensor = t + u

            j0: torch.Tensor = p[i1] < t/w
            j1: torch.Tensor = ~j0

            _x: torch.Tensor = x[i1]

            if j0.any():
                _x[j0] = (a[i1][j0]*w[j0]*p[i1][j0]).pow(1.0/a[i1][j0])
            if j1.any():
                _x[j1] = 1.0 - (b*w[j1]*(1.0 - p[i1][j1])).pow(1.0/b)

            x[i1] = _x

        i: torch.Tensor = torch.logical_and(x > 0.0, x < 1.0)
        a1: torch.Tensor = a[i] - 1.0
        b1: torch.Tensor = b - 1.0
        afac: torch.Tensor = -torch.lgamma(a[i]) - torch.lgamma(b) + torch.lgamma(a[i] + b)

        for j in range(N):

            err = self.ibf(x[i], a[i], b) - p[i]

            t: torch.Tensor = (a1*x[i].log() + b1*(1.0 - x[i]).log() + afac).exp()
            u: torch.Tensor = err/t
            t = u/(1.0 - 0.5*(u*(a1/x[i] - b1/(1.0 - x[i]))).clamp(min = None, max = 1.0))
            x[i] -= t

            _x: torch.Tensor = x[i]

            j0: torch.Tensor = _x <= 0.0
            j1: torch.Tensor = _x >= 1.0

            if j0.any():
                _x[j0] = 0.5*(_x[j0] + t[j0])
            if j1.any():
                _x[j1] = 0.5*(_x[j1] + t[j1] + 1.0)

            x[i] = _x

            if j > 2:
                break

            if (t.abs() < E*x[i]).all() and j > 0:
                break

        return x

    def ibf(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the regularized incomplete beta function.

        Args:
            x: Evaluation points.
            a: First shape parameter.
            b: Second shape parameter.

        Returns:
            Values of the regularized incomplete beta function.
        """
        # TODO a > 3000 and b > 3000 case not handled differently with an optimization

        p: torch.Tensor = torch.zeros_like(x)
        p[x == 1.0] += 1.0

        i: torch.Tensor = torch.logical_and(x > 0.0, x < 1.0)
        i0: torch.Tensor = x < (a + 1.0)/(a + b + 2.0)
        i1: torch.Tensor = ~i0

        i0 = torch.logical_and(i0, i)
        i1 = torch.logical_and(i1, i)


        if i1.any():
            beta: torch.Tensor = (torch.lgamma(a[i1] + b) - torch.lgamma(a[i1]) - torch.lgamma(b) + a[i1]*x[i1].log() + b*(1.0 - x[i1]).log()).exp()
            p[i1] = 1.0 - beta*self.betacf(1.0 - x[i1], b, a[i1])/b

        if i0.any():
            beta: torch.Tensor = (torch.lgamma(a[i0] + b) - torch.lgamma(a[i0]) - torch.lgamma(b) + a[i0]*x[i0].log() + b*(1.0 - x[i0]).log()).exp()
            p[i0] = beta*self.betacf(x[i0], a[i0], b)/a[i0] 

        return p

    def betacf(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, M: int = 1000, E: float = 1e-6) -> torch.Tensor:
        """
        Evaluate the continued-fraction representation used in the
        computation of the incomplete beta function.

        The implementation follows a modified Lentz algorithm for
        stable evaluation of the continued fraction.

        Args:
            x: Evaluation points.
            a: First shape parameter.
            b: Second shape parameter.
            M: Maximum number of continued-fraction iterations.
            E: Convergence tolerance.

        Returns:
            Continued-fraction approximation values.
        """
        #fpmin: torch.Tensor = torch.finfo(torch.float32).tiny/E
        fpmin: torch.Tensor = torch.tensor(1.1754943508222875e-38/E, dtype = torch.float32, device = x.device)

        qab: torch.Tensor = a + b
        qap: torch.Tensor = a + 1.0
        qam: torch.Tensor = a - 1.0
        c: torch.Tensor = torch.tensor(1.0, dtype = torch.float32, device = x.device)
        d: torch.Tensor = 1.0 - qab*x/qap

        d[d.abs() < fpmin] -= d[d.abs() < fpmin] + fpmin
        d = 1.0/d
        h: torch.Tensor = d

        for m in range(1, M):

            m2: float = 2.0*m
            aa: torch.Tensor = m*(b - m)*x/((qam + m2)*(a + m2))
            d = 1.0 + aa*d.clone()
            d[d.abs() < fpmin] -= d[d.abs() < fpmin] + fpmin
            
            c = 1.0 + aa/c
            c[c.abs() < fpmin] -= c[c.abs() < fpmin] + fpmin

            d = 1.0/d
            h *= d*c
            aa = -(a + m)*(qab + m)*x/((a + m2)*(qap + m2))
            d = 1.0 + aa*d
            d[d.abs() < fpmin] -= d[d.abs() < fpmin] + fpmin

            c = 1.0 + aa/c
            c[c.abs() < fpmin] -= c[c.abs() < fpmin] + fpmin

            d = 1.0/d
            dl: torch.Tensor = d*c
            h *= dl

            if ((dl - 1.0).abs() <= E).all():
                break

        return h

class ModelMarginal(nn.Module):
    """
    Marginal distribution model based on a sequence of rational cubic
    spline flow transformations applied to a chosen base distribution.

    The model maps samples from a base distribution (Normal, Logistic,
    or Student's t) through one or more spline blocks to obtain a
    flexible marginal distribution whose parameters are predicted by
    an external neural network.
    """

    def __init__(self, 
                 number_of_blocks: int = 1, 
                 number_of_knots: int = 6,
                 base_type: int = 0):
        """
        Initialize the marginal flow model.

        Args:
            number_of_blocks: Number of spline flow transformations.
            number_of_knots: Number of knots used by each spline block.
            base_type: Base distribution type
                (0 = Normal, 1 = Logistic, 2 = Student's t).
        """

        super().__init__()

        self.sqrt2   = torch.sqrt(torch.tensor(2.0, dtype = torch.float32))
        self.sqrt2pi = torch.sqrt(torch.tensor(2.0, dtype = torch.float32)*math.pi)
    
        self.number_of_blocks = number_of_blocks
        self.number_of_knots  = number_of_knots
       
        self.spline_block = SplineBlock()

        # Define parameter regression neural network for estimating the normalizing spline flow parameters (knot-value pairs)
        
        self.ones = nn.Parameter(torch.ones((1, 1), dtype = torch.float32), requires_grad = False)

        self.censored: bool = False

        self.base_n = Normal()
        self.base_l = Logistic()
        self.base_s = Student()

        self.type = base_type

        if self.type <= 1: 
            self.number_of_required_parameters = self.number_of_blocks*self.number_of_knots*2
        elif self.type == 2:
            self.number_of_required_parameters = self.number_of_blocks*self.number_of_knots*2 + 1
        else:
           raise RuntimeError('Unsupported base distribution class.')

    def forward(self, f: torch.Tensor, model_parameters: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply the sequence of spline flow transformations.

        Args:
            f: Samples from the base distribution or intermediate
                transformed values.
            model_parameters: Spline parameters generated by
                set_parameters().

        Returns:
            Transformed samples after all spline blocks have been
            applied.
        """
        
        for i in torch.arange(self.number_of_blocks):
            _t, _y, _d = self.initialize_spline(model_parameters, i)
            f = self.spline_block(f, _t, _y, _d)

        return f

    @jit.export
    def set_parameters(self, parameters: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert a flattened parameter tensor into structured spline
        parameters.

        Depending on the selected base distribution, this method
        extracts spline knot parameters and optional distribution
        parameters.

        Args:
            parameters: Raw model output tensor.

        Returns:
            List containing spline knot parameters and any additional
            base-distribution parameters.
        """

        if self.type <= 1:
            parameters = parameters.view((parameters.shape[0], parameters.shape[1], self.number_of_blocks, 2, self.number_of_knots))
            return [parameters[:, :, :, 0].flatten(0, 1), parameters[:, :, :, 1].flatten(0, 1)]

        elif self.type == 2:
            v: torch.Tensor = parameters[..., 0].flatten(0, 1)
            parameters = parameters[..., 1:].view((parameters.shape[0], parameters.shape[1], self.number_of_blocks, 2, self.number_of_knots))
            #return [parameters[:, :, :, 0].flatten(0, 1), parameters[:, :, :, 1].flatten(0, 1), v.exp() + 1.0]
            return [parameters[:, :, :, 0].flatten(0, 1), parameters[:, :, :, 1].flatten(0, 1), v*0.0 + 1.0]

        else:
            raise RuntimeError('Unsupported base distribution class.')

    def initialize_spline(self, model_parameters: List[torch.Tensor], i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct spline knot locations, knot values, and knot
        derivatives for a specific spline block.

        Args:
            model_parameters: Structured spline parameters.
            i: Index of the spline block.

        Returns:
            Tuple containing:
                - knot locations (t),
                - knot values (y),
                - knot derivatives (d).
        """

        prm_t, prm_y = model_parameters
            
        t = torch.cumsum(torch.cat([prm_t[:, i, 0][..., None], 1e-3 + nn.functional.softplus(prm_t[:, i, 1:])], dim = -1), dim = -1)
        y = torch.cumsum(torch.cat([prm_y[:, i, 0][..., None], 1e-3 + nn.functional.softplus(prm_y[:, i, 1:])], dim = -1), dim = -1)

        h  = t[:, 1:] - t[:, :-1]
        df = (y[:, 1:] - y[:, :-1])/h

        di = df[:, :-1]*df[:, 1:]/((y[:, 2:] - y[:, :-2])/(t[:, 2:] - t[:, :-2]))
        d = torch.cat([self.ones.expand(di.shape[0], -1), di, self.ones.expand(di.shape[0], -1)], dim = -1)

        return t[:, None], y[:, None], d[:, None]

    @jit.export
    def icdf(self, 
             number_of_quantiles: int,
             model_parameters: List[torch.Tensor],
             percentile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate the inverse cumulative distribution function (quantile
        function) of the learned marginal distribution.

        Quantiles are first generated from the selected base
        distribution and then transformed through the inverse spline
        flow.

        Args:
            number_of_quantiles: Number of quantiles to generate.
            model_parameters: Structured spline and distribution
                parameters.
            percentile: Optional percentile values in the interval
                [0, 1].

        Returns:
            Quantiles of the learned marginal distribution.
        """

        if self.type <= 1:

            if self.type == 0:
                quantiles: torch.Tensor = self.base_n.quantile(number_of_quantiles, percentile)
            else:
                quantiles: torch.Tensor = self.base_l.quantile(number_of_quantiles, percentile)
        
        elif self.type == 2:

            quantiles: torch.Tensor = self.base_s.quantile(number_of_quantiles, model_parameters[2], percentile)[..., None]
            model_parameters = [model_parameters[0], model_parameters[1]]
        else:
            raise RuntimeError('Unsupported base distribution class.')

        if len(quantiles.shape) == 1:
            quantiles = quantiles[None, :, None].expand(model_parameters[0].shape[0], -1, -1).clone()

        # [batch*lead, quantiles, 1]

        return self.backward(quantiles, model_parameters)

    @jit.export
    def cdf(self, y: torch.Tensor, model_parameters: List[torch.Tensor]) -> torch.Tensor:
       
        if self.type <= 1:

            if self.type == 0:
                y = self.base_n.cdf(self(y, model_parameters))
            else:
                y = self.base_l.cdf(self(y, model_parameters))

        elif self.type == 2:
            y = self.base_s.cdf(self(y, [model_parameters[0], model_parameters[1]]), model_parameters[2])
        else:
            raise RuntimeError('Unsupported base distribution class.')
        
        return y.clamp(min = 1e-8, max = None)

    @jit.export
    def cdf(self, y: torch.Tensor, model_parameters: List[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the cumulative distribution function of the learned
        marginal distribution.

        The input values are first transformed into the latent base
        distribution space using the spline flow and then evaluated
        using the selected base distribution CDF.

        Args:
            y: Values at which to evaluate the CDF.
            model_parameters: Structured spline and distribution
                parameters.

        Returns:
            Cumulative probabilities corresponding to y.
        """
       
        if self.type <= 1:

            if self.type == 0:
                y = self.base_n.cdf(self(y, model_parameters))
            else:
                y = self.base_l.cdf(self(y, model_parameters))

        elif self.type == 2:
            y = self.base_s.cdf(self(y, [model_parameters[0], model_parameters[1]]), model_parameters[2])
        else:
            raise RuntimeError('Unsupported base distribution class.')
        
        return y.clamp(min = 1e-8, max = None)

    @jit.export
    def ecdf(self, y: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the marginal CDF using the raw parameter tensor.

        This convenience wrapper converts the raw model output into
        structured parameters before calling cdf().

        Args:
            y: Values at which to evaluate the CDF.
            parameters: Raw parameter tensor predicted by the model.

        Returns:
            Cumulative probabilities with the same leading dimensions
            as the input.
        """

        s0, s1 = y.shape[0], y.shape[1]

        model_parameters: List[torch.Tensor] = self.set_parameters(parameters)

        return self.cdf(y.flatten()[:, None, None], model_parameters).view(s0, s1)

    @jit.export
    def lpdf(self, x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probability density function of the learned
        marginal distribution.

        The log-density is computed using the change-of-variables
        formula by combining the base-distribution log-density with
        the accumulated spline Jacobian terms.

        Args:
            x: Values at which to evaluate the log-density.
            parameters: Raw parameter tensor predicted by the model.

        Returns:
            Log-density values with the same shape as x.
        """

        model_parameters: List[torch.Tensor] = self.set_parameters(parameters)        

        x_shape = x.shape

        x = x.flatten(end_dim = -1).view(-1, 1, 1)
        
        if self.type <= 1:

            dt: float = 0.0 
            for i in range(self.number_of_blocks):

                _t, _y, _d = self.initialize_spline(model_parameters, i)

                block_dt = self.spline_block.dt(x, _t, _y, _d)
                log_dt = torch.log(block_dt)

                dt += log_dt
                x  = self.spline_block(x, _t, _y, _d)

            #x = self(x, model_parameters)

            if self.type == 0:
                return (self.base_n.lpdf(x) + dt).view(x_shape)
            else:
                return (self.base_l.lpdf(x) + dt).view(x_shape)

        elif self.type == 2:
            
            x = self(x, [model_parameters[0], model_parameters[1]])

            return self.base_s.lpdf(x, model_parameters[2]).view(x_shape)
        else:
            raise RuntimeError('Unsupported base distribution class.')

    @jit.export
    def backward(self, 
                 p: torch.Tensor,
                 model_parameters: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply the inverse spline flow transformation.

        The spline blocks are traversed in reverse order to map values
        from the latent base-distribution space back into the observed
        data space.

        Args:
            p: Values in latent distribution space.
            model_parameters: Structured spline parameters.

        Returns:
            Values transformed into observation space.
        """

        # p: [batch*lead, quantiles, 1]
        # prm_: [batch*lead, nsplines, number_of_knots]

        for i in torch.flip(torch.arange(self.number_of_blocks), dims = (0,)):
            _t, _y, _d = self.initialize_spline(model_parameters, i)
            p = self.spline_block.backward(p, _t, _y, _d)

        return p

    @jit.export
    def sample_marginal(self, number_of_samples: int, parameters: torch.Tensor) -> torch.Tensor:
        """
        Generate random samples from the learned marginal distribution.

        Samples are first drawn from the selected base distribution and
        then transformed through the inverse spline flow.

        Args:
            number_of_samples: Number of samples to generate.
            parameters: Raw parameter tensor predicted by the model.

        Returns:
            Samples with shape [batch, lead, number_of_samples].
        """
        
        batch, lead = parameters.shape[0], parameters.shape[1]
        model_parameters: List[torch.Tensor] = self.set_parameters(parameters)
    
        if self.type <= 1:

            if self.type == 0:
                base = self.base_n
                samples: torch.Tensor = base.sample(batch*lead*number_of_samples).view(batch*lead, number_of_samples, 1)
            else:
                base = self.base_l
                samples: torch.Tensor = base.sample(batch*lead*number_of_samples).view(batch*lead, number_of_samples, 1)
            
        elif self.type == 2:

            samples: torch.Tensor = self.base_s.sample(number_of_samples, model_parameters[2]).view(batch*lead, number_of_samples, 1)
            model_parameters = [model_parameters[0], model_parameters[1]]
        else:
            raise RuntimeError('Unsupported base distribution class.')

        return self.backward(samples, model_parameters).view(batch, lead, number_of_samples)

    @jit.export
    def sample_marginal_quantile(self, number_of_quantiles: int, parameters: torch.Tensor, percentile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate quantiles from the learned marginal distribution.

        Args:
            number_of_quantiles: Number of quantiles to generate.
            parameters: Raw parameter tensor predicted by the model.
            percentile: Optional percentile values in the interval
                [0, 1].

        Returns:
            Quantiles with shape
            [batch, lead, number_of_quantiles].
        """
    
        batch, lead = parameters.shape[0], parameters.shape[1]
        model_parameters: List[torch.Tensor] = self.set_parameters(parameters)

        quantiles: torch.Tensor = self.icdf(number_of_quantiles, model_parameters, percentile).view(batch, lead, number_of_quantiles)

        return quantiles

    @jit.ignore
    def loss(self, 
             f: torch.Tensor,
             parameters: torch.Tensor):
        """
        Compute the negative log-likelihood training loss.

        Observations are transformed into the latent base-distribution
        space and evaluated using the change-of-variables formula.
        When censoring is enabled, additional likelihood terms are
        included for censored observations.

        Args:
            f: Observed values with shape [batch, lead].
            parameters: Raw parameter tensor predicted by the model.

        Returns:
            Per-sample negative log-likelihood values with shape
            [batch, lead], or None if no valid observations are
            available.
        """

        model_parameters: List[torch.Tensor] = self.set_parameters(parameters)

        # f: [batch, lead]

        dim_batch: int = f.shape[0]
        dim_lead: int  = f.shape[1] 

        f = f.reshape(dim_batch*dim_lead, 1, 1)

        idx: torch.Tensor = (~(f.isnan())).squeeze()
        if ~(idx.any()):
            return None

        loss_out: torch.Tensor = torch.zeros_like(f[..., 0, 0])*torch.nan

        # Transform y into latent distribution

        if self.censored:
            idx_less: torch.Tensor = f.squeeze() <= 0.0
            idx_cns: torch.Tensor = torch.logical_and(~idx_less, idx)
        else:
            idx_cns: torch.Tensor = idx

        if idx_cns.any():

            f = f[idx_cns]

            _model_parameters: List[torch.Tensor] = (model_parameters[0][idx_cns], model_parameters[1][idx_cns])

            dt: torch.Tensor = 0.0
            for i in range(self.number_of_blocks):

                _t, _y, _d = self.initialize_spline(_model_parameters, i)

                block_dt = self.spline_block.dt(f, _t, _y, _d)
                log_dt = torch.log(block_dt)

                dt += log_dt
                f  = self.spline_block(f, _t, _y, _d)

            f  = torch.squeeze(f)
            dt = torch.squeeze(dt)

            match self.type:
                case 0:
                    base = self.base_n
                    loss_out[idx_cns] = -base.lpdf(f) - dt
                case 1:
                    base = self.base_l
                    loss_out[idx_cns] = -base.lpdf(f) - dt
                case 2:
                    base = self.base_s
                    loss_out[idx_cns] = -base.lpdf(f, model_parameters[2][idx_cns]) - dt

        if self.censored:

            i0: torch.Tensor = torch.logical_and(idx_less, idx)

            if i0.any():
                _model_parameters: List[torch.Tensor] = [m[i0] for m in model_parameters]
                _cdf: torch.Tensor = self.cdf(torch.zeros((i0.sum(), 1, 1), dtype = torch.float32, device = f.device), _model_parameters)
                loss_out[i0] = -(_cdf + 1e-10).log().squeeze()

            return loss_out.view(dim_batch, dim_lead)

        return loss_out.view(dim_batch, dim_lead)

