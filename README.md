
# shap-analyzer 🔍

[![PyPI version](https://img.shields.io/pypi/v/shap_analyzer.svg)](https://pypi.org/project/shap_analyzer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

A lightweight, high-level Python wrapper designed to simplify **SHAP (SHapley Additive exPlanations)**. Run full model explainability, handle multi-class 3D SHAP outputs effortlessly, extract top feature drivers, and plot interactive visual insights with minimal code.

---

## ⚡ Why `shap-analyzer`?

Explaining machine learning models with native SHAP often requires tedious array manipulation, manual dynamic dimensional slicing, and repetitive visualization setup. `shap-analyzer` removes the friction.

### Standard SHAP vs. `shap-analyzer`

| Task | Native `shap` Library | `shap-analyzer` |
| :--- | :--- | :--- |
| **Multi-class Handling** | Requires manual slicing of 3D array outputs (`shap_values[:, :, class_idx]`) | Automatic slicing with a simple parameter: `analyze(model, X, class_index=2)` |
| **Setup Boilerplate** | 15–20 lines of explainer setup, data extraction, and indexing | 2 lines of code to get a complete explanation object |
| **Local Feature Drivers** | Manual iteration over arrays to find top positive/negative contributions | Built-in helper: `exp.get_top_drivers(index=0, top_n=3)` |
| **Plot Customization** | Requires writing custom Matplotlib figure wrappers for every chart | Ready-to-use, clean visualizer modules for sequential or grouped display |

---
## 🛠️ API & Function Reference

| Method / Function | Signature | Description |
| :--- | :--- | :--- |
| **`analyze`** | `analyze(model, X_data, sample_size=None, class_index=0)` | Factory function that initializes `ModelExplainer` and computes SHAP values with automatic multi-class slicing. |
| **`.get_feature_importance()`** | `exp.get_feature_importance()` | Returns a pandas DataFrame of global mean absolute SHAP values sorted by feature importance. |
| **`.get_instance_explanation()`** | `exp.get_instance_explanation(index=0)` | Returns individual SHAP contribution scores and raw feature values for a single specified row. |
| **`.get_top_drivers()`** | `exp.get_top_drivers(index=0, top_n=3)` | Returns a dictionary containing the top N positive and negative driving features for a specific row prediction. |
| **`plot_summary`** | `plot_summary(explainer_obj)` | Renders a Beeswarm summary plot showing global feature impact and value distributions. |
| **`plot_bar`** | `plot_bar(explainer_obj)` | Renders a global bar chart of mean absolute SHAP values ranked by feature importance. |
| **`plot_local`** | `plot_local(explainer_obj, index=0)` | Renders a local Waterfall plot breaking down prediction drivers for a single target row. |
| **`plot_dependence`** | `plot_dependence(explainer_obj, feature_name)` | Renders a feature interaction plot to analyze non-linear dependencies. |
| **`plot_heatmap`** | `plot_heatmap(explainer_obj, num_samples=50)` | Renders a SHAP heatmap across multiple sample rows. |
| **`plot_decision`** | `plot_decision(explainer_obj, index=0)` | Renders a decision trajectory plot showing cumulative feature contribution paths for a row prediction. |

---
---

## 📦 Installation

Install directly via `pip` (**Recommended**):

```bash
pip install shap_analyzer

```

Or install locally in editable mode for development:

```bash
git clone https://github.com/saad-js/shap_analyzer.git
cd shap_analyzer
py -m pip install -e .

```

---

## 🚀 Quickstart

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from shap_analyzer import analyze, plot_summary, plot_local

# 1. Prepare data & train model
X, y = make_classification(n_samples=250, n_features=6, n_classes=3, random_state=42)
X_df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(6)])

model = RandomForestClassifier(random_state=42)
model.fit(X_df, y)

# 2. Initialize Explainer (target Class 2 for multi-class)
exp = analyze(model, X_df, class_index=2)

# 3. Extract Global Feature Importance & Local Drivers
print(exp.get_feature_importance())
drivers = exp.get_top_drivers(index=5, top_n=3)
print("Top Positive Drivers:", drivers["positive_drivers"])

# 4. Render Visualizations
plot_summary(exp)
plot_local(exp, index=5)

```

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create! Any contributions you make are **greatly appreciated**.

### How to Contribute

1. **Fork the Repository**
Click the **Fork** button at the top right of this repository page.
2. **Clone your Fork**
```bash

git clone https://github.com/saad-js/shap_analyzer.git
cd shap_analyzer

```


3. **Create a Feature Branch**
```bash
git checkout -b feature/AmazingFeature

```


4. **Make Changes & Test**
Make your code additions or bug fixes, then run the test suite:
```bash
py -m pytest

```


5. **Commit Your Changes**
```bash
git commit -m "Add AmazingFeature: Brief description of changes"

```


6. **Push to Your Branch**
```bash
git push origin feature/AmazingFeature

```


7. **Open a Pull Request**
Go to the original repository on GitHub and click **Compare & pull request** to submit your changes for review.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## ✉️ Contact & Support

* **Author:** ***ABU SAAD*** - [[abusaadd44@gmail.com](https://www.google.com/search?q=mailto%3Aabusaadd44@gmail.com)]
* **Project Link:** [[https://github.com/saad-js/shap_analyzer](https://www.google.com/search?q=https://github.com/saad-js/shap_analyzer)]

