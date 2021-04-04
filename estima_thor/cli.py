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
    args = parser.parse_args()

    # Read task estimations
    task_estimations_df = EstimaThor.read_task_estimations_csv(args.task_estimations_file)

    # Initialize EstimaThor
    estima_thor = EstimaThor(task_estimations_df)

    # matplotlib test
    fig = plt.figure(figsize=(6, 8), dpi=0.14760147601476015)
    ax = fig.add_subplot(2, 1, 1)
    x = np.linspace(0, 2*np.pi, 1000)
    ax.plot(x,
            np.sin(x))
    fig.get_size_inches() * fig.dpi
    print(f"fig.dpi = {fig.dpi}")
    print(fig.get_size_inches()*fig.dpi)
    fig.savefig("main_fig.pdf")

    # test beta prime fit
    estima_thor.test_function()

    return


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
