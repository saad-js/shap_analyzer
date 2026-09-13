import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from shap_analyzer import (
    analyze,
    plot_bar,
    plot_decision,
    plot_dependence,
    plot_heatmap,
    plot_local,
    plot_summary,
)

# 1. Setup synthetic 3-class dataset
X, y = make_classification(
    n_samples=250,
    n_features=6,
    n_informative=4,
    n_classes=3,
    random_state=42,
)
X_df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(6)])

# 2. Train multi-class model
model = RandomForestClassifier(random_state=42)
model.fit(X_df, y)

# 3. Analyze Class 2 specifically using class_index=2
exp = analyze(model, X_df, class_index=2)


print("1. GLOBAL FEATURE IMPORTANCE")

importance_df = exp.get_feature_importance()
print(importance_df)
print("\n")


print("2. MULTI-ROW LOCAL DRIVER EVALUATION")

# Evaluate specific rows: Row 0, Row 5, and Row 12
rows_to_evaluate = [0, 5, 12]

for r_idx in rows_to_evaluate:
    drivers = exp.get_top_drivers(index=r_idx, top_n=2)
    print(f"--- Top Drivers for Row #{r_idx} ---")
    
    print(" Positive Drivers:")
    for d in drivers["positive_drivers"]:
        print(f"    • {d['feature']}: {d['feature_value']:.2f} (Impact: +{d['shap_impact']:.4f})")
        
    print(" Negative Drivers:")
    for d in drivers["negative_drivers"]:
        print(f"    • {d['feature']}: {d['feature_value']:.2f} (Impact: {d['shap_impact']:.4f})")
    print()


print("3. TESTING VISUALIZATIONS")

# Global overview plots
print("Displaying Summary (Beeswarm) Plot...")
plot_summary(exp)

print("Displaying Global Bar Plot...")
plot_bar(exp)

print("Displaying Heatmap Plot (across first 50 rows)...")
plot_heatmap(exp, num_samples=50)

print("Displaying Feature Dependence Plot for Feature_0...")
plot_dependence(exp, feature_name="Feature_0")

# Row-specific local plots for Row #5
print("Displaying Local Waterfall Plot for Row #5...")
plot_local(exp, index=5)

print("Displaying Decision Trajectory Plot for Row #5...")
plot_decision(exp, index=5)
