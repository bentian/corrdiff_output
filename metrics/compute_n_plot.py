import argparse
from aggregate_metrics import aggregate_metrics
from compute_density_n_error import compute_density_n_error
from score_samples import main as score_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Path for the output file.")
    parser.add_argument("score", type=str, help="Path to the score file.")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble members.")
    args = parser.parse_args()

    # # CorrDiff scores
    score_samples(args.output, args.score, args.n_ensemble)

    # # Aggregate metrics
    aggregate_metrics(args.score, args.score[:-3])

    # # Create PRCP PDF
    compute_density_n_error(args.output, args.output[:-3], args.n_ensemble)

