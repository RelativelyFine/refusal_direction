import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def parse_evaluations(filepath: str) -> tuple[float, float]:
    with open(filepath) as f:
        data = json.load(f)
    refusal_score = data["substring_matching_success_rate"]
    safety_score = data["llamaguard2_success_rate"]
    return 1 - refusal_score, 1 - safety_score


def get_best_double_results(model_name: str) -> tuple[list[int], tuple[float, float]]:
    model_path = Path("pipeline/runs") / Path(model_name)
    double_paths = model_path.glob("*-*/completions/jailbreakbench_ablation_evaluations.json")

    best_directions = None
    best_refusal = 1.0
    best_safety = 1.0

    for path in double_paths:
        directions = [int(d) for d in path.parents[1].name.split("-")]
        print(model_name, directions)
        refusal, safety = parse_evaluations(path)

        if safety < best_safety:
            best_directions = directions
            best_refusal = refusal
            best_safety = safety

    if best_directions is None:
        raise RuntimeError("no directions found")
    return best_directions, (best_refusal, best_safety)


def parse_all_evaluations(model_name: str):
    model_path = Path("pipeline/runs") / Path(model_name)
    baseline_path = model_path / Path("0/completions/jailbreakbench_baseline_evaluations.json")
    single_path = model_path / Path("0/completions/jailbreakbench_ablation_evaluations.json")

    baseline_results = parse_evaluations(baseline_path)
    single_results = parse_evaluations(single_path)
    double_direction, double_results = get_best_double_results(model_name)
    print(model_name, double_direction)

    results = {
        "Refusal score (No intervention)": baseline_results[0],
        "Safety score (No intervention)": baseline_results[1],
        "Refusal score (Single direction ablation)": single_results[0],
        "Safety score (Single direction ablation)": single_results[1],
        "Refusal score (Best double direction ablation)": double_results[0],
        "Safety score (Best double direction ablation)": double_results[1],
    }
    return results


def graph_results(model_names: list[str]):
    x = np.arange(len(model_names))
    width = 1 / 7
    multiplier = 0

    fig, ax = plt.subplots()

    agg_results = {
        "Refusal score (No intervention)": [],
        "Safety score (No intervention)": [],
        "Refusal score (Single direction ablation)": [],
        "Safety score (Single direction ablation)": [],
        "Refusal score (Best double direction ablation)": [],
        "Safety score (Best double direction ablation)": [],
    }
    patterns = ["", "", "..", "..", "//", "//"]
    orange = "#f5b942"
    blue = "#6fb7f2"
    colors = [orange, blue, orange, blue, orange, blue]

    for model_name in model_names:
        results = parse_all_evaluations(model_name)
        for key, value in agg_results.items():
            value.append(results[key])

    for idx, (attribute, measurement) in enumerate(agg_results.items()):
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            measurement,
            width,
            label=attribute,
            hatch=patterns[idx],
            color=colors[idx],
            edgecolor="black",
        )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Score")
    ax.set_xticks(x + 2.5 * width, model_names)

    refusal_handle = patches.Patch(facecolor=orange, edgecolor="black", label="Refusal score (1 - refusal_benchmark)")
    safety_handle = patches.Patch(facecolor=blue, edgecolor="black", label="Safety score (1 - safety_benchmark)")
    baseline_handle = patches.Patch(facecolor="white", alpha=1.0, edgecolor="black", hatch="", label="No ablation")
    single_handle = patches.Patch(facecolor="white", alpha=1.0, edgecolor="black", hatch="...", label="Single direction ablation")
    double_handle = patches.Patch(facecolor="white", alpha=1.0, edgecolor="black", hatch="//", label="Best double direction ablation")

    ax.spines[['right', 'top']].set_visible(False)

    ax.legend(loc="best", handles=[refusal_handle, safety_handle, baseline_handle, single_handle, double_handle])
    ax.set_ylim(0.0, 1.0)
    plt.show()


def _parse_ce_loss(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data["pile"]["ce_loss"]


def _get_best_double_ce_loss(model_name: str) -> tuple[list[int], float]:
    model_path = Path("pipeline/runs") / Path(model_name)
    double_paths = list(model_path.glob("*-*/loss_evals/ablation_loss_eval.json"))

    best_directions = None
    best_ce_loss = 100000.0

    for path in double_paths:
        directions = [int(d) for d in path.parents[1].name.split("-")]
        ce_loss = _parse_ce_loss(path)

        if ce_loss < best_ce_loss:
            best_directions = directions
            best_ce_loss = ce_loss

    if best_directions is None:
        raise RuntimeError("no directions found")
    return best_directions, best_ce_loss


def parse_ce_loss(model_name, ds):
    model_path = Path("pipeline/runs") / Path(model_name)
    baseline = _parse_ce_loss(model_path / Path("0/loss_evals/baseline_loss_eval.json"))
    single = _parse_ce_loss(model_path / Path("0/loss_evals/ablation_loss_eval.json"))
    double = _parse_ce_loss(model_path / Path(f"{ds[0]}-{ds[1]}/loss_evals/ablation_loss_eval.json"))
    return baseline, single, double


def graph_ce_loss(model_names, directions):
    x = np.arange(len(model_names))
    width = 1 / 4
    multiplier = 0

    fig, ax = plt.subplots()

    agg_results = {
        "Baseline": [],
        "Single direction ablation": [],
        "Best double direction ablation": [],
    }
    patterns = ["", ".", "//"]
    yellow = "#f6fc7e"
    orange = "#f5b942"
    blue = "#6fb7f2"
    colors = [yellow, orange, blue]

    for model_name, ds in zip(model_names, directions):
        baseline, single, double = parse_ce_loss(model_name, ds)
        agg_results["Baseline"].append(baseline)
        agg_results["Single direction ablation"].append(single)
        agg_results["Best double direction ablation"].append(double)

    for idx, (attribute, measurement) in enumerate(agg_results.items()):
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            measurement,
            width,
            label=attribute,
            hatch=patterns[idx],
            color=colors[idx],
            edgecolor="black",
        )
        ax.bar_label(rects, padding=2, fmt="%.2f")
        multiplier += 1

    ax.set_ylabel("CE Loss")
    ax.set_xticks(x + 1 * width, model_names)
    ax.legend(loc="upper right")
    ax.set_ylim(1.9, 2.8)
    plt.show()


if __name__ == "__main__":
    model_names = [
        "Llama-3.2-3B-Instruct",
        "Llama-3.2-1B-Instruct",
        "Llama-3.1-8B-Instruct",
        "Yi-6B-Chat",
        "Phi-3.5-mini-instruct",
        "gemma-2-2b-it",
    ]
    graph_results(model_names)

    # directions = [get_best_double_results(m)[0] for m in model_names]
    # graph_ce_loss(model_names, directions)
