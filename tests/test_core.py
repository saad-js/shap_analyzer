import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from shap_analyzer import analyze, ModelExplainer


@pytest.fixture
def binary_data_and_model():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
    model = RandomForestClassifier(random_state=42).fit(X_df, y)
    return model, X_df


@pytest.fixture
def multiclass_data_and_model():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, n_informative=3, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
    model = RandomForestClassifier(random_state=42).fit(X_df, y)
    return model, X_df


def test_explainer_initialization(binary_data_and_model):
    model, X_df = binary_data_and_model
    exp = analyze(model, X_df)
    assert isinstance(exp, ModelExplainer)
    assert len(exp.feature_names) == 4


def test_feature_importance(binary_data_and_model):
    model, X_df = binary_data_and_model
    exp = analyze(model, X_df)
    df_imp = exp.get_feature_importance()
    
    assert isinstance(df_imp, pd.DataFrame)
    assert list(df_imp.columns) == ["feature", "mean_shap_value"]
    assert len(df_imp) == 4


def test_top_drivers(binary_data_and_model):
    model, X_df = binary_data_and_model
    exp = analyze(model, X_df)
    drivers = exp.get_top_drivers(index=0, top_n=2)
    
    assert "positive_drivers" in drivers
    assert "negative_drivers" in drivers
    assert len(drivers["positive_drivers"]) <= 2



# Currently working on multiclass Models
# def test_multiclass_handling(multiclass_data_and_model):
#     model, X_df = multiclass_data_and_model
#     exp = analyze(model, X_df, class_index=2)
    
#     assert exp.class_index == 2
#     assert exp.shap_values.values.shape[0] == 100


# def test_out_of_bounds_class_index(multiclass_data_and_model):
    # model, X_df = multiclass_data_and_model
    # with pytest.raises(ValueError):
    #     analyze(model, X_df, class_index=99)