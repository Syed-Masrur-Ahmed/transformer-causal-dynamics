
import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mgf_dataset import simulate_ou_process, simulate_binary_ou_process, compute_mixture_cumulants
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
    """Load trained model checkpoint and sync architecture config."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = cfg.get("paths", {}).get("save_dir", "experiments")
    model_name = cfg.get("paths", {}).get("mgf_model_name", "model_mgf.pth")
    model_path = os.path.join(project_root, save_dir, model_name)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Use architecture from checkpoint if available (has inferred d_input/d_output)
    arch = checkpoint.get("config", {}).get("architecture", cfg["architecture"])
    cfg["architecture"] = arch

    model = SimpleTransformer(**arch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_sequence_lengths(seq_min: int, seq_max: int, seq_step: int) -> List[int]:
    """Create sequence-length sweep and ensure max length is included."""
    lengths = list(range(seq_min, seq_max + 1, seq_step))
    if lengths[-1] != seq_max:
        lengths.append(seq_max)
    return lengths


def compute_truth_all_coefficients(
    trajectories: torch.Tensor,
    theta_tensor: torch.Tensor,
    mu: float,
    dt: float,
    D_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ground truth for all 3 MGF coefficients at final step.

    Returns:
        truth: (num_trajectories, 3) tensor
            - truth[:, 0]: Conditional mean = mu + (X_t - mu) * exp(-theta * dt)
            - truth[:, 1]: Conditional variance = (D/theta) * (1 - exp(-2*theta*dt))
            - truth[:, 2]: Zero (third MGF coefficient)
    """
    num_trajectories = trajectories.shape[0]
    truth = torch.zeros(num_trajectories, 3, dtype=torch.float32)

    # Coefficient 0: Conditional mean
    truth[:, 0] = mu + (trajectories[:, -1, 0] - mu) * torch.exp(-theta_tensor * dt)

    # Coefficient 1: Conditional variance
    truth[:, 1] = (D_tensor / theta_tensor) * (1 - torch.exp(-2 * theta_tensor * dt))

    # Coefficient 2: Zero
    truth[:, 2] = 0.0

    return truth


def compute_truth_binary_observed(
    trajectories: torch.Tensor,
    states: torch.Tensor,
    theta_tensor: torch.Tensor,
    mu: float,
    dt: float,
    kappa: float,
    D_tensor: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """
    Compute ground truth cumulants at final step for binary OU with observed mu.

    Returns:
        truth: (num_trajectories, order) tensor
    """
    import numpy as np

    theta_expanded = theta_tensor.unsqueeze(1)
    D_expanded = D_tensor.unsqueeze(1)
    c = torch.exp(-theta_expanded * dt)
    r = 1 - c
    sigma2 = (D_expanded / theta_expanded) * (1 - torch.exp(-2 * theta_expanded * dt))
    delta = mu * r

    b = 2 * np.exp(-kappa * dt) - 1
    # Use only the final time step state
    p_plus = (1 + b * states[:, -1]) / 2  # (num_traj,)

    cumulant_list = compute_mixture_cumulants(
        p_plus, delta.squeeze(1), sigma2.squeeze(1), order
    )

    truth = torch.zeros(trajectories.shape[0], order, dtype=torch.float32)
    x_t = trajectories[:, -1, 0]
    truth[:, 0] = x_t * c.squeeze(1) + cumulant_list[0]
    for k in range(1, order):
        truth[:, k] = cumulant_list[k]

    return truth


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


def get_coefficient_selection(args_coef) -> int:
    """
    Get user selection for which MGF coefficient to test.
    Returns coefficient index (0, 1, or 2).
    """
    coefficient_names = {
        0: "Conditional Mean (mu + (X_t - mu) * exp(-theta*dt))",
        1: "Conditional Variance ((D/theta) * (1 - exp(-2*theta*dt)))",
        2: "Zero (third MGF coefficient)",
    }

    # If provided via command-line argument
    if args_coef is not None:
        if args_coef not in [0, 1, 2]:
            raise ValueError(f"Invalid coefficient index: {args_coef}. Must be 0, 1, or 2.")
        print(f"\nTesting MGF Coefficient {args_coef}: {coefficient_names[args_coef]}")
        return args_coef

    # Interactive selection
    print("\n" + "=" * 70)
    print("SELECT MGF COEFFICIENT TO TEST")
    print("=" * 70)
    for idx, name in coefficient_names.items():
        print(f"  [{idx}] {name}")
    print("=" * 70)

    while True:
        try:
            selection = input("\nEnter coefficient index (0, 1, or 2): ").strip()
            coef_idx = int(selection)
            if coef_idx in [0, 1, 2]:
                print(f"\n✓ Selected: Coefficient {coef_idx} - {coefficient_names[coef_idx]}\n")
                return coef_idx
            else:
                print("Invalid selection. Please enter 0, 1, or 2.")
        except ValueError:
            print("Invalid input. Please enter a number (0, 1, or 2).")
        except KeyboardInterrupt:
            print("\n\nTest cancelled by user.")
            sys.exit(0)


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


def run_predictive_tests(cfg: Dict, coefficient_idx: int = 0) -> None:
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

    coefficient_names = ["mean", "variance", "zero"]
    print(f"\nRunning predictive tests for MGF coefficient {coefficient_idx} ({coefficient_names[coefficient_idx]})...")
    print(f"Total test conditions: {total_jobs} (sequence_length, theta) combinations\n")
    progress = tqdm(total=total_jobs, desc=f"Testing coef_{coefficient_idx}")

    mode = cfg.get("mode", "standard_ou")

    with torch.no_grad():
        for sweep, sequence_lengths, theta_grid in sweep_jobs:
            num_replicates = int(sweep["num_replicates"])
            physics = cfg.get("physics", {})
            mu = float(sweep.get("mu", physics.get("mu", 0.0)))
            dt = float(sweep.get("dt", physics.get("dt", 0.1)))
            fixed_marginal_variance = float(sweep.get("fixed_marginal_variance", physics.get("marginal_variance", 0.1)))
            kappa = float(physics.get("kappa", 0.5))
            order = int(cfg.get("target", {}).get("order", 2))

            for seq_len in sequence_lengths:
                for theta in theta_grid:
                    theta_tensor = torch.full((num_replicates,), float(theta), dtype=torch.float32)
                    d_tensor = fixed_marginal_variance * theta_tensor

                    if mode == "binary_ou_observed":
                        trajectories, states = simulate_binary_ou_process(
                            num_trajectories=num_replicates,
                            sequence_length=seq_len,
                            theta_tensor=theta_tensor,
                            mu=mu,
                            D_tensor=d_tensor,
                            dt=dt,
                            kappa=kappa,
                        )
                        # Build 2-channel input
                        mu_t = (states * mu).unsqueeze(-1)
                        traj_input = torch.cat([trajectories, mu_t], dim=-1)

                        truth_all = compute_truth_binary_observed(
                            trajectories, states, theta_tensor, mu, dt, kappa, d_tensor, order
                        )
                        truth = truth_all[:, coefficient_idx]
                    else:
                        trajectories = simulate_ou_process(
                            num_trajectories=num_replicates,
                            sequence_length=seq_len,
                            theta_tensor=theta_tensor,
                            mu=mu,
                            D_tensor=d_tensor,
                            dt=dt,
                        )
                        traj_input = trajectories

                        truth_all = compute_truth_all_coefficients(
                            trajectories, theta_tensor, mu, dt, d_tensor
                        )
                        truth = truth_all[:, coefficient_idx]

                    traj_device = traj_input.to(device)

                    # Get predictions for selected coefficient
                    trained_preds = trained_model(traj_device)[0][:, -1, coefficient_idx].detach().cpu()
                    untrained_preds = untrained_model(traj_device)[0][:, -1, coefficient_idx].detach().cpu()

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

    # Include coefficient index in output filename
    base_name = cfg.get("paths", {}).get("predictive_test_results_name", "predictive_test_results.csv")
    name_parts = base_name.rsplit(".", 1)
    if len(name_parts) == 2:
        output_name = f"{name_parts[0]}_coef{coefficient_idx}.{name_parts[1]}"
    else:
        output_name = f"{base_name}_coef{coefficient_idx}"

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
    parser = argparse.ArgumentParser(
        description="Test transformer model on MGF coefficient predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for coefficient selection)
  python scripts/test.py

  # Test conditional mean (coefficient 0)
  python scripts/test.py --coef 0

  # Test conditional variance (coefficient 1)
  python scripts/test.py --coef 1

  # Test zero coefficient (coefficient 2)
  python scripts/test.py --coef 2
        """
    )
    parser.add_argument(
        "--coef",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="MGF coefficient to test: 0=mean, 1=variance, 2=zero (if not provided, interactive prompt)"
    )

    args = parser.parse_args()
    config = load_test_config()

    # Get coefficient selection (from args or interactive prompt)
    coefficient_idx = get_coefficient_selection(args.coef)

    # Run tests
    run_predictive_tests(config, coefficient_idx)
