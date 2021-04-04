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

    def __init__(self,
                 task_estimations_df: pd.DataFrame,
                 estimation_quantiles=(0.05, 0.5, 0.95),
                 distribution_name="norm"):
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
                                     distribution_name="norm")
            model_list.append(model)



            # add pdf to plot
            fig.clf()
            ax_pdf = fig.add_subplot(2, 1, 1)
            ax_pdf.plot(time_list,
                        np.vectorize(model.pdf)(time_list),
                        label="PDF Fit")
            ax_pdf.grid(True)
            ax_pdf.legend()
            ax_pdf.text(x=0.8,
                        y=0.5,
                        s=f"a = {np.round(model.loc, 5)} \n b = {np.round(model.scale, 5)}",
                        transform=ax_pdf.transAxes)


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

            fig.savefig(f"plot_{task_estimation.iloc[0]}.pdf")


        return

