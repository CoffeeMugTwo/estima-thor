"""Main module."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from estima_thor.distributions import fit_distribution


def test_function():
    print("Hello World! asdfasdfsadf")
    return

class EstimaThor(object):
    """
    Main class of this tool
    """
    dpi_scale = 0.14760147601476015

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
        estimated_quantiles : array
            List of the estimation quantiles
        distribution_name : str
            Name of the distribution used for the internal fit
        """

        self.task_estimations_df = task_estimations_df
        self.estimation_quantiles = estimation_quantiles
        self.distribution_name = distribution_name
        self.model_list = list()
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

    def get_project_estimations(self):
        # ToDo: this is just a data flow test
        total_min = self.task_estimations_df['5_qtl'].sum()
        total_max = self.task_estimations_df['95_qtl'].sum()
        total_peak = self.task_estimations_df['peak'].sum()
        return total_min, total_peak, total_max


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

    def simulate_project(self, n: int=1000):
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
        run_time_list = np.zeros(shape=n)
        for i in range(n):
            # get random (distributed according to task estimations) duration for each task
            for model in self.model_list:
                task_duration = model.inv_cdf(np.random.rand())
                run_time_list[i] += task_duration

        # return results
        fig = plt.figure(dpi=self.dpi_scale,
                         figsize=(8, 4.5))
        ax_hist = fig.add_subplot(1, 1, 1)
        ax_hist.grid(True)

        # Plot histogram of task project estimations
        median = np.quantile(run_time_list, q=0.5)
        ax_hist.hist(run_time_list,
                     bins=int(n/10),
                     range=[0, median * 3],
                     color="C0")

        # Plot some statistics
        qtl_05 = np.quantile(run_time_list, q=0.05)
        qtl_95 = np.quantile(run_time_list, q=0.95)
        mean = np.mean(run_time_list)
        ax_hist.axvline(qtl_05, label=f"5% Quantile = {np.round(qtl_05):.0f}", color="C4")
        ax_hist.axvline(median, label=f"Median = {np.round(median):.0f}", color="C1")
        ax_hist.axvline(mean, label=f"Mean = {np.round(mean):.0f}", color="C2")
        ax_hist.axvline(qtl_95, label=f"95% Quantile = {np.round(qtl_95):.0f}", color="C3")
        ax_hist.legend()

        # Save figure
        fig.savefig("project_durations.pdf")
        return

    def test_function(self):
        """
        A test function
        Returns
        -------

        """

        fig = plt.figure(figsize=(8, 4.5),
                         dpi=0.14760147601476015)
        time_list = np.linspace(0, 100, 1000)

        model_list = list()
        for index, task_estimation in self.task_estimations_df.iterrows():
            model = fit_distribution(task_estimation.iloc[1:],
                                     self.estimation_quantiles,
                                     distribution_name="spline")
            model_list.append(model)



            # add pdf to plot
            fig.clf()
            ax_pdf = fig.add_subplot(2, 1, 1)
            sampled = [model.inv_cdf(x) for x in np.random.rand(10000)]
            ax_pdf.hist(sampled,
                        bins=100,
                        range=[0, 100])

            # ax_pdf.plot(time_list,
            #             np.vectorize(model.pdf)(time_list),
            #             label="PDF Fit")
            # ax_pdf.grid(True)
            # ax_pdf.legend()
            # ax_pdf.text(x=0.8,
            #             y=0.5,
            #             s=f"a = {np.round(model.loc, 5)} \n b = {np.round(model.scale, 5)}",
            #             transform=ax_pdf.transAxes)


            print("DEBUG ######################")
            print(model)
            x = 4
            y = model.cdf(x)
            vec_cdf = np.vectorize(model.cdf)
            # print(vec_cdf(time_list))
            print(x, y)

            # add cdf to plot
            ax_cdf = fig.add_subplot(2, 1, 2)
            x = time_list
            y = np.vectorize(model.cdf)(time_list)
            ax_cdf.plot(time_list,
                        np.vectorize(model.cdf)(time_list),
                        label="CDF Fit")

            # add estimations to cdf plot
            ax_cdf.plot(task_estimation.iloc[1:],
                        self.estimation_quantiles,
                        marker='o',
                        linestyle="",
                        label="Estimated Quantiles")

            ax_cdf.grid(True)
            ax_cdf.legend()

            fig.savefig(f"spline_plot_{task_estimation.iloc[0]}.pdf")


        return

