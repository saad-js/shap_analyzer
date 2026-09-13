"""
Plotting module built on top of SHAP visual outputs.
"""

import matplotlib.pyplot as plt
import numpy as np
import shap


def plot_summary(explainer_obj, max_display: int = 10):
    """Generates a SHAP beeswarm summary plot showing global feature impact."""
    plt.figure()
    shap.plots.beeswarm(explainer_obj.shap_values, max_display=max_display)
    plt.tight_layout()
    plt.show()


def plot_bar(explainer_obj, max_display: int = 10):
    """Generates a standard bar plot of mean absolute SHAP values."""
    plt.figure()
    shap.plots.bar(explainer_obj.shap_values, max_display=max_display)
    plt.tight_layout()
    plt.show()


def plot_local(explainer_obj, index: int = 0):
    """Generates a waterfall plot for explaining a specific single row prediction."""
    plt.figure()
    shap.plots.waterfall(explainer_obj.shap_values[index])
    plt.tight_layout()
    plt.show()

def plot_dependence(explainer_obj, feature_name: str):
    """
    Shows how a single feature's value affects SHAP values 
    and auto-colors by its strongest interacting feature.
    """
    plt.figure()
    shap.plots.scatter(explainer_obj.shap_values[:, feature_name], color=explainer_obj.shap_values)
    plt.tight_layout()
    plt.show()


def plot_heatmap(explainer_obj, max_display: int = 10, num_samples: int = 100):
    """
    Visualizes SHAP values across multiple rows simultaneously as a heatmap.
    """
    plt.figure()
    shap.plots.heatmap(explainer_obj.shap_values[:num_samples], max_display=max_display)
    plt.tight_layout()
    plt.show()


def plot_decision(explainer_obj, index: int = 0):
    """
    Shows the step-by-step decision trajectory from base value to final model prediction.
    """
    plt.figure()
    expected_val = explainer_obj.explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = expected_val[1] if len(expected_val) > 1 else expected_val[0]

    shap.decision_plot(
        expected_val, 
        explainer_obj.shap_values.values[index], 
        feature_names=explainer_obj.feature_names
    )
    plt.tight_layout()
    plt.show()