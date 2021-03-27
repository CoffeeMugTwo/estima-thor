"""Console script for estima_thor."""
import argparse
import sys

from estima_thor.estima_thor import EstimaThor


def main():
    """Console script for estima_thor."""
    parser = argparse.ArgumentParser()
    parser.add_argument('task_estimations_file',
                        help="CSV file containing the task estimations")
    args = parser.parse_args()

    # Read task estimations
    task_estimations_df = EstimaThor.read_task_estimations_csv(args.task_estimations_file)

    # Initialize EstimaThor
    estima_thor = EstimaThor(task_estimations_df)

    # Get estimations
    est_min, est_peak, est_max = estima_thor.get_project_estimations()

    # Print estimations on screen
    print(f"Min = {est_min} \nPeak = {est_peak} \nMax = {est_max}")

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
