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


def fit_distribution(task_estimation: pd.Series,
                     estimated_quantiles=[0.05, 0.5, 0.95],
                     distribution_name=""):
    """
    Use least square fit to estimate a and b parameters of a prime beta
    distribution

    Parameters
    ----------
    task_estimation: pd.Series
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
                               xdata=np.array(task_estimation),
                               ydata=estimated_quantiles)

        fitted_prime_beta = BetaPrimeFit(a=popt[0],
                                         b=popt[1],
                                         loc=popt[2])

        return fitted_prime_beta

    elif distribution_name == "norm":
        p0 = [task_estimation.values[1],
              1 / np.sqrt(task_estimation.values[0] + task_estimation.values[2])]
        popt, pcov = curve_fit(f=NormFit.norm_cdf_wrapper,
                               xdata=task_estimation.values,
                               ydata=estimated_quantiles,
                               p0=p0)

        fitted_norm = NormFit(loc=popt[0],
                              scale=popt[1])

        return fitted_norm




