import argparse
import os
import sys
from typing import Optional

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scripts.data_gen import generate_mgf_data
from scripts.test import load_test_config, run_predictive_tests
from scripts.train import train_mgf_prediction
from src.utils import apply_experiment_id_to_paths, generate_experiment_id, load_full_config


def run_pipeline(experiment_id: Optional[str] = None) -> str:
    run_id = experiment_id or generate_experiment_id()
    os.environ["EXPERIMENT_ID"] = run_id

    train_cfg = load_full_config()
    train_cfg.setdefault("paths", {})["experiment_id"] = run_id
    apply_experiment_id_to_paths(train_cfg)

    print(f"Running pipeline with experiment_id={run_id}")
    generate_mgf_data(train_cfg)
    train_mgf_prediction(train_cfg)

    test_cfg = load_test_config()
    test_cfg.setdefault("paths", {})["experiment_id"] = run_id
    apply_experiment_id_to_paths(test_cfg)
    run_predictive_tests(test_cfg)

    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data generation, training, and testing pipeline.")
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional experiment ID prefix. If omitted, a unique ID is generated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    final_id = run_pipeline(args.experiment_id)
    print(f"Pipeline completed for experiment_id={final_id}")
