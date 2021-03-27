"""Main module."""

from pathlib import Path

import pandas as pd


def test_function():
    print("Hello World! asdfasdfsadf")
    return

class EstimaThor(object):
    """
    Main class of this tool
    """

    def __init__(self, task_estimations_df: pd.DataFrame):
        """
        Constructor for this class

        Parameters
        ----------
        task_estimations_df : pd.DataFrame
            Data frame containing the task and respective estimations
        """

        self.task_estimations_df = task_estimations_df

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


