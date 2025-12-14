# classification_error_plots.py
# Multi-panel prediction error plots for ordinal 3-class classification
# (Low=0, Medium=1, High=2).

import numpy as np
import matplotlib.pyplot as plt


def _add_jitter(values, scale=0.08):
    values = np.asarray(values, dtype=float)
    noise = (np.random.rand(len(values)) - 0.5) * 2 * scale
    return values + noise


def prediction_error_plot_ordinal(ax, y_true, y_pred, model_name="Model"):
    """
    One panel: Actual vs Predicted (ordinal) with error bands.
    Error = absolute difference in class index (0,1,2).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Error in "number of classes off"
    err = np.abs(y_true - y_pred)

    mask0 = err == 0          # perfect
    mask1 = err == 1          # one class off
    mask2 = err >= 2          # two classes off (Low vs High)

    # Add small jitter so points do not sit exactly on integers
    x0 = _add_jitter(y_pred[mask0])
    y0 = _add_jitter(y_true[mask0])
    x1 = _add_jitter(y_pred[mask1])
    y1 = _add_jitter(y_true[mask1])
    x2 = _add_jitter(y_pred[mask2])
    y2 = _add_jitter(y_true[mask2])

    ax.scatter(x0, y0, s=12, c="#2ca02c", alpha=0.7, label="Error 0 classes")
    ax.scatter(x1, y1, s=12, c="#ff7f0e", alpha=0.7, label="Error 1 class")
    ax.scatter(x2, y2, s=12, c="#d62728", alpha=0.7, label="Error 2 classes")

    # Perfect prediction diagonal
    ax.plot([-0.5, 2.5], [-0.5, 2.5], "k--", lw=1.5, label="Perfect prediction")

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Prediction Error Plot for {model_name}", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=7, frameon=True)


def multi_model_class_error_plots(y_true, preds_by_model,
                                  ncols=3,
                                  figsize=(12, 8),
                                  save_path="figures/prediction_error_plots_multimodel.png",
                                  suptitle="Prediction Error Plots for Health-Risk Models"):
    """
    Create a grid of prediction error plots for several models.

    preds_by_model: dict like
        {
          'log_reg': y_pred_logreg,
          'knn': y_pred_knn,
          'decision_tree': y_pred_dt,
          'random_forest': y_pred_rf,
          'gbm': y_pred_gbm
        }
    """
    model_names = list(preds_by_model.keys())
    n = len(model_names)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(model_names):
        prediction_error_plot_ordinal(
            ax=axes[i],
            y_true=y_true,
            y_pred=preds_by_model[name],
            model_name=name.replace("_", " ").title()
        )

    # Hide extra axes if grid has unused cells
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return save_path
