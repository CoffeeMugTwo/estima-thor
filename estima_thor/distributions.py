from scipy.stats import betaprime
from scipy.optimize import curve_fit
import pandas as pd


class BetaPrimeFit(object):
    """
    On instance represents one fitted beta distribution
    """

    def __init__(self, a: float, b: float):
        """
        Constructor

        Parameters
        ----------
        a : float
            First parameter
        b : float
            Second parameter
        scaling_factor : float

        """
        self.a = a
        self.b = b
        return

    def pdf(self, x:float):
        """
        Returns  PDF(x)

        Parameters
        ----------
        x : float
            x in [0, 1]

        Returns
        -------

        """
        return betaprime.pdf(x, a=self.a, b=self.b)


def fit_beta_prim(task_estimation: pd.Series,
                  estimated_quantiles=[0.05, 0.5, 0.95]):
    """
    Use least square fit to estimate a and b parameters of a prime beta
    distribution

    Parameters
    ----------
    task_estimation: pd.Series
        Containing 5%-quantile, 50%-quantile and 95%-quantile, array-type

    Returns
    -------
    fitted_prime_beta : BetaPrimeFit
        A BetaPrimeFit instance with the fitted parameters
    """
    popt, pcov = curve_fit()
