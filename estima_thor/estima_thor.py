"""Main module."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from estima_thor.distributions import fit_distribution


class EstimaThor(object):
    """
    Main class of this tool
    """

    def __init__(self,
                 task_estimations_df: pd.DataFrame,
                 estimation_quantiles=(0.05, 0.5, 0.95),
                 distribution_name="spline"):
        """
        Constructor for this class

        Parameters
        ----------
        task_estimations_df : pd.DataFrame
            Data frame containing the task and respective estimations
        estimation_quantiles : array
            List of the estimation quantiles
        distribution_name : str
            Name of the distribution used for the internal fit
        """

        self.run_time_list = np.array(list())
        self.n = 0
        self.task_estimations_df = task_estimations_df
        self.estimation_quantiles = estimation_quantiles
        self.distribution_name = distribution_name
        self.model_list = list()

        # Workaround for matplotlib dpi scaling problem
        target_dpi = 100.0
        test_fig = plt.figure(dpi=target_dpi)
        self.dpi_scale = target_dpi / test_fig.dpi

        return

    @classmethod
    def read_task_estimations_csv(cls, path_to_file: Path) -> pd.DataFrame:
        """
        Read the task estimations form a csv file into a pandas data frame.

        Parameters
        ----------
        path_to_file : Path
            Path to file

        Returns
        -------
        task_estimations_df : pd.DataFrame
        """
        task_estimations_df = pd.read_csv(path_to_file,
                                          sep=',',
                                          index_col=0)
        return task_estimations_df

    def fit_distributions(self):
        """
        Perform a least square fit for each task estimation
        """
        model_list = list()
        for index, task_estimation in self.task_estimations_df.iterrows():
            model = fit_distribution(task_estimation.iloc[1:],
                                     self.estimation_quantiles,
                                     distribution_name=self.distribution_name)
            model_list.append(model)
        self.model_list = model_list
        return

    def simulate_project(self, n: int = 1000):
        """
        Simulate the duration of the project n-times.

        Parameters
        ----------
        n : int
            Number of times the simulation should be run

        Returns
        -------
        run_time_list : array like
            Simulated duration for each simulation, n-dimensional.
        """
        self.n = n
        self.run_time_list = np.zeros(shape=n)
        for i in tqdm(range(n)):
            # get random (distributed according to task estimations) duration for each task
            for model in self.model_list:
                task_duration = model.inv_cdf(np.random.rand())
                self.run_time_list[i] += task_duration

        return

    def plot_simulation_results(self, file_name='project_duration_distribution.pdf'):
        """
        Create a histogram of simulated project durations with some statistics

        Parameters
        ----------
        file_name : string
            Path of the output file.
        """

        # create figure instance
        fig = plt.figure(dpi=200 * self.dpi_scale,
                         figsize=(8, 4.5))
        ax_hist = fig.add_subplot(1, 1, 1)
        ax_hist.grid(True)

        # Plot histogram of task project estimations
        median = np.quantile(self.run_time_list, q=0.5)
        ax_hist.hist(self.run_time_list,
                     bins=int(self.n/10),
                     range=[0, median * 3],
                     color="C0")

        # Plot some statistics
        qtl_05 = np.quantile(self.run_time_list, q=0.05)
        qtl_95 = np.quantile(self.run_time_list, q=0.95)
        mean = np.mean(self.run_time_list)
        ax_hist.axvline(qtl_05, label=f"5% Quantile = {np.round(qtl_05):.0f}", color="C4")
        ax_hist.axvline(median, label=f"Median = {np.round(median):.0f}", color="C1")
        ax_hist.axvline(mean, label=f"Mean = {np.round(mean):.0f}", color="C2")
        ax_hist.axvline(qtl_95, label=f"95% Quantile = {np.round(qtl_95):.0f}", color="C3")
        ax_hist.legend()

        # Save figure
        fig.savefig(file_name)
        return
