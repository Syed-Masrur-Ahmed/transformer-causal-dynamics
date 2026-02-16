
import csv
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mgf_dataset import simulate_ou_process
from src.model import SimpleTransformer
from src.utils import apply_experiment_id_to_paths, deep_update, load_full_config


def load_test_config() -> Dict:
    """Merge base configs with configs/test.yaml."""
    cfg = load_full_config()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_cfg_path = os.path.join(project_root, "configs", "test.yaml")

    if not os.path.exists(test_cfg_path):
        raise FileNotFoundError(f"Missing test config at {test_cfg_path}")

    with open(test_cfg_path, "r") as f:
        test_cfg = yaml.safe_load(f) or {}

    deep_update(cfg, test_cfg)
    apply_experiment_id_to_paths(cfg)
    return cfg


def load_trained_model(cfg: Dict, device: torch.device) -> SimpleTransformer:
    """Load trained model checkpoint."""
    model = SimpleTransformer(**cfg["architecture"]).to(device)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = cfg.get("paths", {}).get("save_dir", "experiments")
    model_name = cfg.get("paths", {}).get("mgf_model_name", "model_mgf.pth")
    model_path = os.path.join(project_root, save_dir, model_name)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_sequence_lengths(seq_min: int, seq_max: int, seq_step: int) -> List[int]:
    """Create sequence-length sweep and ensure max length is included."""
    lengths = list(range(seq_min, seq_max + 1, seq_step))
    if lengths[-1] != seq_max:
        lengths.append(seq_max)
    return lengths


def compute_truth_final_step(
    trajectories: torch.Tensor,
    theta: float,
    mu: float,
    dt: float,
) -> torch.Tensor:
    """
    Ground-truth conditional mean at final step, matching notebook logic:
    truth = mu + (X_t - mu) * exp(-theta * dt)
    """
    return mu + (trajectories[:, -1, 0] - mu) * np.exp(-theta * dt)


def compute_relative_error_decomposition(preds: torch.Tensor, truth: torch.Tensor) -> Dict[str, float]:
    """Return squared bias, variance, and total relative error."""
    eps = 1e-12
    denom = torch.where(truth.abs() < eps, truth.sign() * eps, truth)
    denom = torch.where(denom == 0, torch.full_like(denom, eps), denom)
    rel_err = (preds - truth) / denom

    squared_bias = torch.mean(rel_err).item() ** 2
    variance = torch.var(rel_err, unbiased=False).item()
    relative_error = squared_bias + variance
    return {
        "squared_bias": float(squared_bias),
        "variance": float(variance),
        "relative_error": float(relative_error),
    }


def get_test_sweeps(cfg: Dict) -> List[Dict]:
    """
    Normalize supported test config shapes into a list of sweep configs.
    Supports:
    - test_dataset: {...}
    - theta_sweeps / sequence_length_sweeps: {...}
    """
    sweeps = []
    if "test_dataset" in cfg:
        sweep = dict(cfg["test_dataset"])
        sweep["name"] = "test_dataset"
        sweeps.append(sweep)
        return sweeps

    for key in ["theta_sweeps", "sequence_length_sweeps"]:
        if key in cfg and isinstance(cfg[key], dict):
            sweep = dict(cfg[key])
            sweep["name"] = key
            if "num_theta_values" not in sweep and "theta_values" in sweep:
                sweep["num_theta_values"] = sweep["theta_values"]
            sweeps.append(sweep)

    if not sweeps:
        raise KeyError(
            "No valid test sweep found. Expected `test_dataset` or "
            "`theta_sweeps`/`sequence_length_sweeps` in configs/test.yaml."
        )

    return sweeps


def run_predictive_tests(cfg: Dict) -> None:
    apply_experiment_id_to_paths(cfg)
    device_cfg = cfg.get("system", {}).get("device", "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() and device_cfg == "cuda" else "cpu")

    trained_model = load_trained_model(cfg, device)
    untrained_model = SimpleTransformer(**cfg["architecture"]).to(device)
    untrained_model.eval()

    sweeps = get_test_sweeps(cfg)
    sweep_jobs = []
    for sweep in sweeps:
        sequence_lengths = build_sequence_lengths(
            int(sweep["sequence_length_min"]),
            int(sweep["sequence_length_max"]),
            int(sweep["sequence_length_step"]),
        )
        theta_grid = np.linspace(
            float(sweep["theta_min"]),
            float(sweep["theta_max"]),
            int(sweep["num_theta_values"]),
            dtype=np.float64,
        )
        sweep_jobs.append((sweep, sequence_lengths, theta_grid))

    rows = []
    total_jobs = sum(len(lengths) * len(thetas) for _, lengths, thetas in sweep_jobs)

    print(f"Running predictive tests on {total_jobs} (sequence_length, theta) settings...")
    progress = tqdm(total=total_jobs, desc="Predictive sweep")

    with torch.no_grad():
        for sweep, sequence_lengths, theta_grid in sweep_jobs:
            num_replicates = int(sweep["num_replicates"])
            mu = float(sweep["mu"])
            dt = float(sweep["dt"])
            fixed_marginal_variance = float(sweep["fixed_marginal_variance"])

            for seq_len in sequence_lengths:
                for theta in theta_grid:
                    theta_tensor = torch.full((num_replicates,), float(theta), dtype=torch.float32)
                    d_tensor = fixed_marginal_variance * theta_tensor

                    trajectories = simulate_ou_process(
                        num_trajectories=num_replicates,
                        sequence_length=seq_len,
                        theta_tensor=theta_tensor,
                        mu=mu,
                        D_tensor=d_tensor,
                        dt=dt,
                    )

                    traj_device = trajectories.to(device)
                    trained_preds = trained_model(traj_device)[0][:, -1, 0].detach().cpu()
                    untrained_preds = untrained_model(traj_device)[0][:, -1, 0].detach().cpu()
                    truth = compute_truth_final_step(trajectories, theta=float(theta), mu=mu, dt=dt)

                    trained_metrics = compute_relative_error_decomposition(trained_preds, truth)
                    untrained_metrics = compute_relative_error_decomposition(untrained_preds, truth)

                    rows.append(
                        {
                            "sweep_name": sweep["name"],
                            "sequence_length": int(seq_len),
                            "theta": float(theta),
                            "trained_squared_bias": trained_metrics["squared_bias"],
                            "trained_variance": trained_metrics["variance"],
                            "trained_relative_error": trained_metrics["relative_error"],
                            "untrained_squared_bias": untrained_metrics["squared_bias"],
                            "untrained_variance": untrained_metrics["variance"],
                            "untrained_relative_error": untrained_metrics["relative_error"],
                            "num_replicates": int(num_replicates),
                        }
                    )
                    progress.update(1)

    progress.close()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, cfg.get("paths", {}).get("save_dir", "experiments"))
    os.makedirs(output_dir, exist_ok=True)
    output_name = cfg.get("paths", {}).get("predictive_test_results_name", "predictive_test_results.csv")
    output_csv = os.path.join(output_dir, output_name)

    fieldnames = [
        "sweep_name",
        "sequence_length",
        "theta",
        "trained_squared_bias",
        "trained_variance",
        "trained_relative_error",
        "untrained_squared_bias",
        "untrained_variance",
        "untrained_relative_error",
        "num_replicates",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    config = load_test_config()
    run_predictive_tests(config)
