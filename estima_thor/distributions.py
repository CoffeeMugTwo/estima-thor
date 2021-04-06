from scipy.stats import betaprime
from scipy.stats import norm
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np


class BetaPrimeFit(object):
    """
    On instance represents one fitted beta distribution
    """

    def __init__(self, a: float, b: float, loc: float):
        """
        Constructor

        Parameters
        ----------
        a : float
            First parameter
        b : float
            Second parameter
        loc : float
            Location of the distribution
        scale : float
            Scaling factor

        """
        self.a = a
        self.b = b
        self.loc = loc
        return

    def __repr__(self):
        string = f"a = {self.a} b = {self.b}"
        return string

    def pdf(self, x:float):
        """
        Returns  PDF(x)

        Parameters
        ----------
        x : float
            x in [0, +inf]

        Returns
        -------

        """
        return betaprime.pdf(x, a=self.a, b=self.b)

    def cdf(self, x:float) -> float:
        """
        Returns CDF(x)

        Parameters
        ----------
        x : float
            x in [0, +inf]

        Returns
        -------

        """
        return betaprime.cdf(x, a=self.a, b=self.b)

    @classmethod
    def beta_prime_cdf_wrapper(cls,
                               x: float,
                               a: float,
                               b: float,
                               loc: float):
        """
        A wrapper for the beta prime cdf. In order to be usable by curve fit
        Parameters
        ----------
        x : float
        a : float
        b : float
        loc: float
        scale: float

        Returns
        -------
        evaluated beta prime function cdf
        """
        print("DEBUG####################")
        print(x, a, b, loc)
        return betaprime.cdf(x, a, b, loc=loc)


class NormFit(object):
    """
    On instance represents one fitted normal distribution
    """

    def __init__(self, loc: float, scale: float):
        """
        Constructor

        Parameters
        ----------
        loc : float
            Location of the distribution
        scale : float
            Scaling factor

        """
        self.loc = loc
        self.scale = scale
        return

    def __repr__(self):
        string = f"loc = {self.loc} scale = {self.scale}"
        return string

    def pdf(self, x:float):
        """
        Returns  PDF(x)

        Parameters
        ----------
        x : float
            x in [0, +inf]

        Returns
        -------

        """
        return norm.pdf(self.scale * (x- self.loc))

    def cdf(self, x:float) -> float:
        """
        Returns CDF(x)

        Parameters
        ----------
        x : float
            x in [0, +inf]

        Returns
        -------

        """
        return norm.cdf( self.scale * (x- self.loc) )

    @classmethod
    def norm_cdf_wrapper(cls,
                         x: float,
                         loc: float,
                         scale: float):
        """
        A wrapper for the normal distribution cdf. In order to be usable by curve fit
        Parameters
        ----------
        x : float
        a : float
        b : float
        loc: float
        scale: float

        Returns
        -------
        evaluated normal distribution cdf
        """
        print("DEGUG############################")
        print(x, loc, scale)
        return norm.cdf((x - loc) * scale)


class SplineFit(object):
    """Class for the spline fit model"""

    def __init__(self,
                 p0: float,
                 p1_0: float,
                 p1_1: float,
                 p2_0: float,
                 p2_1: float,
                 p3: float,
                 x0: float,
                 x1: float,
                 x2: float):
        """
        Constructor

        Parameters
        ----------
        p0 : float
            Scaling parameter for f0
        p1_0 : float
            Gradient for first linear spline
        p1_1 : float
            Offset for first linear spline
        p2_0 : float
            Gradient for second linear spline
        p2_1 : float
            Offset for second linear spline
        p3 : float
            Offset for f3
        x0 : float
            Defines the upper boarder of f0
        x1 : float
            Defines the upper boarder of f1
        x2 : float
            Defines the upper boarder of f2
        """

        # Set parameters
        self.p0 = p0
        self.p1_0 = p1_0
        self.p1_1 = p1_1
        self.p2_0 = p2_0
        self.p2_1 = p2_1
        self.p3 = p3

        # Set domains of definition
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2

        return

    def __repr__(self):
        """
        A nice representation method

        Returns
        -------
        repr_string : str
        """
        repr_string = (
            f"f0(x) = {self.p0} * (e^x - 1)\n"
            f"f1(x) = {self.p1_0} * x + {self.p1_1}\n"
            f"f2(x) = {self.p2_0} * x + {self.p2_1}\n"
            f"f3(x) = 1 - 1 / (x - {self.p3})"
        )
        return repr_string

    def f0(self, x):
        """
        Exponential Function for the first section: f0(x) = p0 * (e^x - 1)

        Parameters
        ----------
        x : float
            0 <= x < x0

        Returns
        -------
        f0(x) : float
            f0(x) = p0 * (e^x - 1)
        """
        if not (x <= x < self.x0):
            raise ValueError(f"f0 is only defined between 0 and x0={self.x0}")
        return (self.p0 * (np.exp(x) - 1))

    def inv_f0(self, x):
        """
        Inverted f0: f0^-1(x) = ln(x / p0 + 1)

        Parameters
        ----------
        x : float
            0 <= x < f1(x0)

        Returns
        -------
        f0^-1(x) : float
            f0^-1(x) = ln(x / p0 + 1)
        """
        if not (0 <= x < self.f1(self.x0)):
            raise ValueError(f"f0^-1 is only defined between 0 and f1(x0)={self.f1(self.x0)}")
        return np.log((x / self.p0) + 1)

    def f1(self, x):
        """
        Linear function for the second section : f1(x) = p1_0 * x + p1_1

        Parameters
        ----------
        x : float
            x0 <= x < x1

        Returns
        -------
        f1(x) : float
            f1(x) = p1_0 * x + p1_1
        """
        if not (self.x0 <= x < self.x1):
            raise ValueError(f"f1 is only defined for {self.x0} <= x < {self.x1}")
        return (self.p1_0 * x + self.p1_1)

    def inv_f1(self, x):
        """
        Inverted f1(x): f1^-1(x) = (x - p1_1) / p1_0

        Parameters
        ----------
        x : float
            f1(x0) <= x < f2(x1)

        Returns
        -------
        f1^-1(x) : float
            f1^-1(x) = (x - p1_1) / p1_0
        """
        if not (self.f1(self.x0) <= x < self.f2(self.x1)):
            raise ValueError(
                f"f1^-1 is only defined for f1(x0)={self.f1(self.x0)} <= x < f2(x1)={self.f2(self.x1)}"
            )
        return (x - self.p1_1) / self.p1_0

    def f2(self, x):
        """
        Linear function for the third section : f2(x) = p2_0 * x + p2_1

        Parameters
        ----------
        x : float
            x1 <= x < x2

        Returns
        -------
        f2(x) : float
            f2(x) = p2_0 * x + p2_1
        """
        if not (self.x1 <= x < self.x2):
            raise ValueError(f"f2 is only defined for {self.x1} <= x < {self.x2}")
        return (self.p2_0 * x + self.p2_1)

    def inv_f2(self, x):
        """
        Inverted f2(x): f2^-1(x) = (x - p2_1) / p2_0

        Parameters
        ----------
        x : float
            f2(x1) <= x < f3(x2)

        Returns
        -------
        f2^-1(x) : float
            f2^-1(x) = (x - p2_1) / p2_0
        """
        if not (self.f2(self.x1) <= x < self.f3(self.x2)):
            raise ValueError(
                f"f2^-1 is only defined for f2(x0)={self.f2(self.x1)} <= x < f3(x2)={self.f3(self.x2)}"
            )
        return (x - self.p2_1) / self.p2_0

    def f3(self, x):
        """
        Power function with negative power for fourth section:
        f3(x) = 1 - 1 / (x - p3)

        Parameters
        ----------
        x : float
            x2 <= x < +inf

        Returns
        -------
        f3(x) : float
           f3(x) = 1 - 1 / (x - p3)
        """
        if x < self.x2:
            raise ValueError(f"f3 is only defined for {self.x2} <= x")
        return (1 - 1 / (x - self.p3))

    def inv_f3(self, x):
        """
        Inverted f3(x): f3^-1(x) = p3 + 1 / (1 - x)
        Parameters
        ----------
        x : float
            f3(x2) <= x < 1

        Returns
        -------
        f3^-1(x) : float
            f3^-1(x) = p3 + 1 / (1 - x)
        """
        if not (self.f3(self.x2) <= x <1):
            raise ValueError(
                f"f3^-1(x) is only defined for: f3(x2)={self.f3(self.x2)} <= x < 1"
            )
        return self.p3 + 1 / (1 - x)

    def pdf(self, x: float):
        """Return the probability density function evaluated at x"""
        raise NotImplementedError()

    def cdf(self, x: float) -> float:
        """
        Return the cumulative probability distribution evaluated at x.

        Parameters
        ----------
        x : float
            0 <= x < +inf
        Returns
        -------
        cdf(x) : float
            0 <= cdf(x) < 1
        """
        if x < self.x0:
            return self.f0(x)
        elif self.x0 <= x < self.x1:
            return self.f1(x)
        elif self.x1 <= x < self.x2:
            return self.f2(x)
        else:
            return self.f3(x)

    def inv_cdf(self, x):
        """
        Returns the inverted CDF evaluated at x.

        Parameters
        ----------
        x : float
            0 <= x < 1

        Returns
        -------
        cdf^-1(x) : float
        """
        value_error = (
            f"CDF^-1(x) is only defined for 0 <= x < 1"
        )

        if (x < 0) or (1 <= x):
            raise ValueError(value_error)
        elif 0 <= x < self.f1(self.x0):
            return self.inv_f0(x)
        elif self.f1(self.x0) <= x < self.f2(self.x1):
            return self.inv_f1(x)
        elif self.f2(self.x1) <= x < self.f3(self.x2):
            return self.inv_f2(x)
        else:
            return self.inv_f3(x)

    @classmethod
    def fit_spline_function(cls,
                            task_estimations,
                            estimated_quantiles):
        """
        Returns a SplineFit instance with the parameters set according to the
        provided task estimations and estimated quantiles

        Parameters
        ----------
        task_estimations : array like
            estimated values (ascending)
        estimated_quantiles : array like
            quantiles for the estimations (ascending)

        Returns
        -------
        spline_fit : SplineFit
            SplineFit instance with respective parameters set
        """
        # Scaling parameter for f0
        p0 = estimated_quantiles[0] / (np.exp(task_estimations[0]) - 1)

        # Gradient and offset for f1
        p1_0 = (estimated_quantiles[1] - estimated_quantiles[0]) / \
               (task_estimations[1] - task_estimations[0])

        p1_1 = estimated_quantiles[0] - p1_0 * task_estimations[0]

        # Gradient and offset for f2
        p2_0 = (estimated_quantiles[2] - estimated_quantiles[1]) / \
                (task_estimations[2] - task_estimations[1])

        p2_1 = estimated_quantiles[1] - p2_0 * task_estimations[1]

        # Offset parameter for f3
        p3 = task_estimations[2] - 1 / (1 - estimated_quantiles[2])

        spline_fit = cls(p0,
                         p1_0,
                         p1_1,
                         p2_0,
                         p2_1,
                         p3,
                         task_estimations[0],
                         task_estimations[1],
                         task_estimations[2])

        return spline_fit






def fit_distribution(task_estimations: pd.Series,
                     estimated_quantiles=[0.05, 0.5, 0.95],
                     distribution_name=""):
    """
    Use least square fit to estimate a and b parameters of a prime beta
    distribution

    Parameters
    ----------
    task_estimations: pd.Series
        Containing 5%-quantile, 50%-quantile and 95%-quantile, array-type
    estimated_quantiles: array like
        Definition of the estimated quantiles (e.g. 0.05, 0.5, 0.95)
    distribution_name: str
        Name of the distribution, that should be used for the fits

    Returns
    -------
    fitted_prime_beta : BetaPrimeFit
        A BetaPrimeFit instance with the fitted parameters
    """


    if distribution_name == "betaprime":
        popt, pcov = curve_fit(f=BetaPrimeFit.beta_prime_cdf_wrapper,
                               xdata=np.array(task_estimations),
                               ydata=estimated_quantiles)

        fitted_prime_beta = BetaPrimeFit(a=popt[0],
                                         b=popt[1],
                                         loc=popt[2])

        return fitted_prime_beta

    elif distribution_name == "norm":
        p0 = [task_estimations.values[1],
              1 / np.sqrt(task_estimations.values[2] - task_estimations.values[0])]
        popt, pcov = curve_fit(f=NormFit.norm_cdf_wrapper,
                               xdata=task_estimations.values,
                               ydata=estimated_quantiles,
                               p0=p0)

        fitted_norm = NormFit(loc=popt[0],
                              scale=popt[1])

        return fitted_norm

    elif distribution_name == "spline":
        spline_fit = SplineFit.fit_spline_function(task_estimations,
                                                   estimated_quantiles)
        return spline_fit
