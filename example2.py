import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from shap_analyzer import analyze

# 1. Train model on sample data
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
X_df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(5)])

model = RandomForestClassifier(random_state=42)
model.fit(X_df, y)

# 2. Run analysis
exp = analyze(model, X_df)

# 3. Test get_top_drivers for Row 0
drivers = exp.get_top_drivers(index=0, top_n=2)

print("--- Top Drivers for Row 0 ---\n")

print("Positive Drivers:")
if drivers["positive_drivers"]:
    for d in drivers["positive_drivers"]:
        print(f"  • {d['feature']}: Value = {d['feature_value']:.2f} | Impact = +{d['shap_impact']:.4f}")
else:
    print("  None")

print("\n Negative Drivers:")
if drivers["negative_drivers"]:
    for d in drivers["negative_drivers"]:
        print(f"  • {d['feature']}: Value = {d['feature_value']:.2f} | Impact = {d['shap_impact']:.4f}")
else:
    print("  None")