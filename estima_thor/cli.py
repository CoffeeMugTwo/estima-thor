"""Console script for estima_thor."""
import argparse
import sys

from estima_thor.estima_thor import EstimaThor

import matplotlib.pyplot as plt
import numpy as np

def main():
    """Console script for estima_thor."""
    parser = argparse.ArgumentParser()
    parser.add_argument('task_estimations_file',
                        help="CSV file containing the task estimations",
                        default="/home/logicwizard/projects/deployment_tests/estima_thor/test_file.csv")
    parser.add_argument("--n",
                        help="Number of times a project should be simulated.",
                        type=int,
                        default=10000)
    args = parser.parse_args()

    # Read task estimations
    task_estimations_df = EstimaThor.read_task_estimations_csv(args.task_estimations_file)

    # Initialize EstimaThor
    estima_thor = EstimaThor(task_estimations_df)

    # test beta prime fit
    estima_thor.fit_distributions()
    estima_thor.simulate_project(args.n)

    return


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
