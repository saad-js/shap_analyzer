from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. Generate Synthetic Dataset
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
feature_names = [f"Feature_{i}" for i in range(5)]
X_df = pd.DataFrame(X, columns=feature_names)

# 2. Train a Model
model = RandomForestClassifier(random_state=42)
model.fit(X_df, y)

# 3. Use Custom Library
from shap_analyzer import analyze, plot_summary, plot_local

# Run analysis
exp = analyze(model, X_df)

# Print Feature Importance Table
print("--- Global Feature Importance ---")
print(exp.get_feature_importance())

# Print Single Instance Explanation
print("\n--- Explanation for Row 0 ---")
print(exp.get_instance_explanation(index=0))

# Display Plots
plot_summary(exp)
plot_local(exp, index=0)