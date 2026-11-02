"""Rescore existing paired trajectories without running CFD or changing training.

The CSVs have no pre-action frame. A shared first-step midpoint is therefore
reported only as an initial-flux PROXY, never as the measured reset flux.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    plan = json.loads((root / "plan.json").read_text())
    reference = float(plan["protocol"]["reference_flux"])
    alpha = -np.expm1(-1 / float(plan["protocol"]["ewma_timescale"]))
    horizons = (100, 200, 400, 600, 800, 1000)
    names = ("feedback", "constant_001")
    records = []
    series = []
    first_step_disagreements = []
    for snapshot in sorted(root.glob("blanket*")):
        arrays = [np.genfromtxt(snapshot / name / "attempt_001/trajectory.csv",
                               delimiter=",", names=True) for name in names]
        if any(len(a) < max(horizons) for a in arrays):
            raise ValueError(f"Incomplete trajectory: {snapshot}")
        for a in arrays:
            np.testing.assert_array_equal(a["step"], np.arange(1, len(a) + 1))
            assert np.all(np.isfinite(a["first_wall_heat_flux"]))
            assert np.all(a["total_heating_power"] > 0)
            np.testing.assert_allclose(a["total_heating_power"],
                                       a["total_heating_power"][0], rtol=1e-12)
            # Verify that the initial EWMA was the fixed reference, so it
            # cannot be inverted to recover the missing pre-control flux.
            previous = np.r_[reference, a["first_wall_heat_flux_ewma"][:-1]]
            np.testing.assert_allclose(a["first_wall_heat_flux_ewma"],
                (1 - alpha) * previous + alpha * a["first_wall_heat_flux"],
                rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(arrays[0]["total_heating_power"],
                                   arrays[1]["total_heating_power"], rtol=1e-12)
        first = [float(a["first_wall_heat_flux"][0]) for a in arrays]
        shared_proxy = float(np.mean(first))
        first_step_disagreements.append(abs(first[0] - first[1]))
        rewards = []
        for controller, a in zip(names, arrays):
            q = a["first_wall_heat_flux"]
            power = a["total_heating_power"]
            r = {
                "stored_ewma": a["reward"],
                "instant_fixed": (reference - q) / power,
                "instant_shared_start_proxy": (shared_proxy - q) / power,
                "instant_own_first_step_proxy": (q[0] - q) / power,
                "negative_flux_fraction": -q / power,
            }
            rewards.append(r)
            for h in horizons:
                row = {"snapshot": snapshot.name, "controller": controller,
                       "horizon": h, "fixed_reference": reference,
                       "shared_start_proxy": shared_proxy,
                       "first_post_action_flux": float(q[0]),
                       "heating_power": float(power[0])}
                row.update({key: float(value[:h].sum()) for key, value in r.items()})
                records.append(row)
        for h in horizons:
            delta = {k: float(rewards[0][k][:h].sum() - rewards[1][k][:h].sum())
                     for k in rewards[0]}
            np.testing.assert_allclose(delta["instant_shared_start_proxy"],
                                       delta["instant_fixed"], atol=1e-11)
            np.testing.assert_allclose(delta["negative_flux_fraction"],
                                       delta["instant_fixed"], atol=1e-11)
        series.append(rewards)

    if not records:
        raise ValueError("No paired trajectories found")
    summary = {
        "pairs": len(series), "reference": reference,
        "initial_flux_available": False,
        "initial_flux_proxy": "Mean of the two first POST-ACTION fluxes; shared within pair",
        "max_first_step_pair_flux_difference": max(first_step_disagreements),
        "horizons": {},
    }
    for h in horizons:
        metrics = {}
        for metric in series[0][0]:
            returns = np.array([[r[metric][:h].sum() for r in pair] for pair in series])
            delta = returns[:, 0] - returns[:, 1]
            metrics[metric] = {
                "feedback_mean_return": float(returns[:, 0].mean()),
                "constant_mean_return": float(returns[:, 1].mean()),
                "mean_paired_difference": float(delta.mean()),
                "feedback_wins": int((delta > 0).sum()),
            }
        summary["horizons"][str(h)] = metrics

    out = root / "reward_rescoring"
    out.mkdir(exist_ok=True)
    with (out / "returns.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for metric, label, style in [
        ("stored_ewma", "EWMA, fixed reference", "-"),
        ("instant_fixed", "Instantaneous, fixed reference", "--"),
        ("instant_shared_start_proxy", "Instantaneous, shared start proxy", ":"),
    ]:
        means = np.mean([p[0][metric] - p[1][metric] for p in series], axis=0)
        axes[0].plot(np.arange(1, len(means) + 1), np.cumsum(means), style, label=label)
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].axvline(200, color="grey", linewidth=.7)
    axes[0].set(xlabel="Control transitions", ylabel="Mean paired cumulative return difference",
                title="Feedback minus constant 0.01 (10 pairs)")
    axes[0].legend(fontsize=8)
    x = np.arange(len(series))
    for metric, label, marker in [
        ("instant_fixed", "Fixed reference", "o"),
        ("instant_shared_start_proxy", "Shared start proxy", "x"),
    ]:
        for i, controller in enumerate(names):
            values = [p[i][metric][:200].sum() for p in series]
            axes[1].plot(x, values, marker=marker, linestyle="-" if i == 0 else "--",
                         label=f"{controller}: {label}")
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set(xticks=x, xticklabels=[p.name[-2:] for p in sorted(root.glob('blanket*'))],
                xlabel="Snapshot suffix", ylabel="200-step return",
                title="Reference changes offsets, not paired differences")
    axes[1].legend(fontsize=7)
    fig.savefig(out / "reward_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"output": str(out), "200_steps": summary["horizons"]["200"],
                      "1000_steps": summary["horizons"]["1000"]}, indent=2))


if __name__ == "__main__":
    main()
