"""Console script for estima_thor."""
import argparse
import sys

from estima_thor.estima_thor import EstimaThor


def main():
    """Console script for estima_thor."""
    parser = argparse.ArgumentParser(
        description=('A simple tool to estimate the duration of projects with a '
                     'Monte Carlo simulation.')
    )
    parser.add_argument('task_estimations_file',
                        help="CSV file containing the task estimations")
    parser.add_argument("-n",
                        help="Number of times a project should be simulated.",
                        type=int,
                        default=10000)
    parser.add_argument('--output-name',
                        help="Name of the output file.",
                        default="project_duration_estimation.pdf",
                        type=str)
    args = parser.parse_args()

    # Read task estimations
    print(f'Reading task estimations from file: {args.task_estimations_file}')
    task_estimations_df = EstimaThor.read_task_estimations_csv(args.task_estimations_file)

    # Initialize EstimaThor
    estima_thor = EstimaThor(task_estimations_df)

    # Fit distributions to task estimations
    print(f'Fit distributions to task estimations')
    estima_thor.fit_distributions()

    # Simulate project duration
    print(f'Simulation project duration {args.n} times')
    estima_thor.simulate_project(args.n)

    # Store results
    print(f'Store results in file: {args.output_name}')
    estima_thor.plot_simulation_results(file_name=args.output_name)

    print('Done')
    return


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
