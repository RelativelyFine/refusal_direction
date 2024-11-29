import json
from pathlib import Path
import matplotlib
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
    # baseline_path = model_path / Path("0/completions/jailbreakbench_baseline_evaluations.json")
    single_path = model_path / Path("0/completions/jailbreakbench_ablation_evaluations.json")

    # baseline_results = parse_evaluations(baseline_path)
    single_results = parse_evaluations(single_path)
    double_direction, double_results = get_best_double_results(model_name)
    print(model_name, double_direction)

    results = {
        # "Refusal score (No intervention)": baseline_results[0],
        # "Safety score (No intervention)": baseline_results[1],
        "Refusal score (Single direction ablation)": single_results[0],
        "Safety score (Single direction ablation)": single_results[1],
        "Refusal score (Best double direction ablation)": double_results[0],
        "Safety score (Best double direction ablation)": double_results[1],
    }
    return results


def graph_results(model_names: list[str]):
    x = np.arange(len(model_names))
    width = 1 / 5
    multiplier = 0

    fig, ax = plt.subplots()

    agg_results = {
        # "Refusal score (No intervention)": [],
        # "Safety score (No intervention)": [],
        "Refusal score (Single direction ablation)": [],
        "Safety score (Single direction ablation)": [],
        "Refusal score (Best double direction ablation)": [],
        "Safety score (Best double direction ablation)": [],
    }
    patterns = ["", "", "//", "//"]
    orange = "#f5b942"
    blue = "#6fb7f2"
    colors = [orange, blue, orange, blue]

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
    ax.set_xticks(x + 1.5 * width, model_names)

    refusal_handle = patches.Patch(facecolor=orange, edgecolor="black", label="Refusal score (1 - refusal_benchmark)")
    safety_handle = patches.Patch(facecolor=blue, edgecolor="black", label="Safety score (1 - safety_benchmark)")
    single_handle = patches.Patch(facecolor="white", alpha=1.0, edgecolor="black", hatch="", label="Single direction ablation")
    double_handle = patches.Patch(facecolor="white", alpha=1.0, edgecolor="black", hatch="//", label="Best double direction ablation")

    ax.legend(loc="upper right", handles=[refusal_handle, safety_handle, single_handle, double_handle])
    ax.set_ylim(0.0, 0.45)
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
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("CE Loss")
    ax.set_xticks(x + 1 * width, model_names)
    ax.legend(loc="lower right")
    plt.show()


# def heatmap(data, row_labels, col_labels, ax=None,
#             cbar_kw=None, cbarlabel="", **kwargs):
#     """
#     Create a heatmap from a numpy array and two lists of labels.

#     Parameters
#     ----------
#     data
#         A 2D numpy array of shape (M, N).
#     row_labels
#         A list or array of length M with the labels for the rows.
#     col_labels
#         A list or array of length N with the labels for the columns.
#     ax
#         A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
#         not provided, use current Axes or create a new one.  Optional.
#     cbar_kw
#         A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
#     cbarlabel
#         The label for the colorbar.  Optional.
#     **kwargs
#         All other arguments are forwarded to `imshow`.
#     """

#     if ax is None:
#         ax = plt.gca()

#     if cbar_kw is None:
#         cbar_kw = {}

#     # Plot the heatmap
#     im = ax.imshow(data, **kwargs)

#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

#     # Show all ticks and label them with the respective list entries.
#     ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
#     ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

#     # Let the horizontal axes labeling appear on top.
#     ax.tick_params(top=True, bottom=False,
#                    labeltop=True, labelbottom=False)

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
#              rotation_mode="anchor")

#     # Turn spines off and create white grid.
#     ax.spines[:].set_visible(False)

#     ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
#     ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
#     ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
#     ax.tick_params(which="minor", bottom=False, left=False)

#     return im, cbar


# def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
#                      textcolors=("black", "white"),
#                      threshold=None, **textkw):
#     """
#     A function to annotate a heatmap.

#     Parameters
#     ----------
#     im
#         The AxesImage to be labeled.
#     data
#         Data used to annotate.  If None, the image's data is used.  Optional.
#     valfmt
#         The format of the annotations inside the heatmap.  This should either
#         use the string format method, e.g. "$ {x:.2f}", or be a
#         `matplotlib.ticker.Formatter`.  Optional.
#     textcolors
#         A pair of colors.  The first is used for values below a threshold,
#         the second for those above.  Optional.
#     threshold
#         Value in data units according to which the colors from textcolors are
#         applied.  If None (the default) uses the middle of the colormap as
#         separation.  Optional.
#     **kwargs
#         All other arguments are forwarded to each call to `text` used to create
#         the text labels.
#     """

#     if not isinstance(data, (list, np.ndarray)):
#         data = im.get_array()

#     # Normalize the threshold to the images color range.
#     if threshold is not None:
#         threshold = im.norm(threshold)
#     else:
#         threshold = im.norm(data.max())/2.

#     # Set default alignment to center, but allow it to be
#     # overwritten by textkw.
#     kw = dict(horizontalalignment="center",
#               verticalalignment="center")
#     kw.update(textkw)

#     # Get the formatter in case a string is supplied
#     if isinstance(valfmt, str):
#         valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

#     # Loop over the data and create a `Text` for each "pixel".
#     # Change the text's color depending on the data.
#     texts = []
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
#             text = im.axes.text(j, i, valfmt(data[i, j], (i, j)), **kw)
#             texts.append(text)

#     return texts


# def graph_double_direction_results(model_names, directions):
#     all_directions = []
#     for model_name in model_names:
#         model_path = Path("pipeline/runs") / Path(model_name)
#         double_paths = model_path.glob("*-*/completions/jailbreakbench_ablation_evaluations.json")

#         for path in double_paths:
#             directions = tuple(int(d) for d in path.parents[1].name.split("-"))
#             all_directions.append(directions)
#     all_directions = sorted(set(all_directions))

#     data = np.zeros((len(model_names), len(all_directions)))
#     data.fill(np.nan)
#     row_labels = model_names
#     col_labels = [",".join(map(str, d)) for d in all_directions]

#     for row, (model_name, ds) in enumerate(zip(model_names, directions)):
#         model_path = Path("pipeline/runs") / Path(model_name)
#         single_path = model_path / Path("0/completions/jailbreakbench_ablation_evaluations.json")
#         double_path = model_path / Path(f"{ds[0]}-{ds[1]}/completions/jailbreakbench_ablation_evaluations.json")

#         _, single_safety = parse_evaluations(single_path)
#         _, double_safety

#         for path in double_paths:
#             directions = tuple(int(d) for d in path.parents[1].name.split("-"))

#             col = all_directions.index(directions)

#             _, double_safety = parse_evaluations(path)
#             data[row, col] = double_safety - single_safety

#     fig, ax = plt.subplots()
#     im, cbar = heatmap(data, row_labels, col_labels, ax=ax)

#     def func(x, pos):
#         if np.isfinite(x):
#             return f"{x:.2f}"
#         if pos in [(1, 4), (1, 5), (2, 4), (2, 5)]:
#             return "X"
#         return ""

#     annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
#     plt.show()


if __name__ == "__main__":
    model_names = [
        "Llama-3.2-3B-Instruct",
        # "Llama-3.2-1B-Instruct",
        # "Llama-3.1-8B-Instruct",
        # "Yi-6B-Chat",
        # "Phi-3.5-mini-instruct",
        "gemma-2-2b-it",
    ]
    # graph_results(model_names)

    directions = [get_best_double_results(m)[0] for m in model_names]

    # graph_double_direction_results(model_names)
    graph_ce_loss(model_names, directions)
