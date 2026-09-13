# # """
# # Core analysis engine wrapping SHAP explainers.
# # """

# # from typing import Union, Optional
# # import pandas as pd
# # import numpy as np
# # import shap


# # class ModelExplainer:
# #     """
# #     Main wrapper class for initializing SHAP analysis on trained models.
# #     """

# #     def __init__(self, model, X_data: pd.DataFrame, sample_size: Optional[int] = None):
# #         """
# #         Parameters
# #         ----------
# #         model : object
# #             Trained Machine Learning model (Scikit-Learn, XGBoost, LightGBM, etc.).
# #         X_data : pd.DataFrame
# #             Dataset used for generating explanations.
# #         sample_size : int, optional
# #             If provided, subsamples X_data for faster computation on large datasets.
# #         """
# #         self.model = model
# #         self.feature_names = list(X_data.columns)

# #         if sample_size and sample_size < len(X_data):
# #             self.X_data = X_data.sample(n=sample_size, random_state=42)
# #         else:
# #             self.X_data = X_data.copy()

# #         # Automatically picks the best explainer (TreeExplainer, LinearExplainer, etc.)
# #         self.explainer = shap.Explainer(self.model, self.X_data)
# #         self.shap_values = self.explainer(self.X_data)

# #     def get_feature_importance(self) -> pd.DataFrame:
# #         """
# #         Calculates global feature importance based on mean absolute SHAP values.

# #         Returns
# #         -------
# #         pd.DataFrame
# #             DataFrame with 'feature' and 'mean_shap_value' sorted in descending order.
# #         """
# #         # Multi-class output handling
# #         if len(self.shap_values.values.shape) == 3:
# #             vals = np.abs(self.shap_values.values).mean(axis=(0, 2))
# #         else:
# #             vals = np.abs(self.shap_values.values).mean(axis=0)

# #         df = pd.DataFrame({
# #             'feature': self.feature_names,
# #             'mean_shap_value': vals
# #         })
# #         return df.sort_values(by='mean_shap_value', ascending=False).reset_index(drop=True)

# #     def get_instance_explanation(self, index: int) -> pd.DataFrame:
# #         """
# #         Extracts feature contributions for a single observation/row.

# #         Parameters
# #         ----------
# #         index : int
# #             Row index position in X_data.

# #         Returns
# #         -------
# #         pd.DataFrame
# #             DataFrame containing feature names, raw values, and instance SHAP impact.
# #         """
# #         if index < 0 or index >= len(self.X_data):
# #             raise IndexError(f"Index {index} is out of bounds for data length {len(self.X_data)}")

# #         single_shap = self.shap_values.values[index]
# #         single_val = self.X_data.iloc[index].values

# #         # If multi-class, select first class for simplicity
# #         if len(single_shap.shape) > 1:
# #             single_shap = single_shap[:, 0]

# #         df = pd.DataFrame({
# #             'feature': self.feature_names,
# #             'feature_value': single_val,
# #             'shap_impact': single_shap
# #         })
# #         return df.sort_values(by='shap_impact', key=abs, ascending=False).reset_index(drop=True)

# from typing import Optional
# import pandas as pd
# import numpy as np
# import shap


# class ModelExplainer:
#     def __init__(self, model, X_data: pd.DataFrame, sample_size: Optional[int] = None):
#         self.model = model
#         self.feature_names = list(X_data.columns)

#         if sample_size and sample_size < len(X_data):
#             self.X_data = X_data.sample(n=sample_size, random_state=42)
#         else:
#             self.X_data = X_data.copy()

#         self.explainer = shap.Explainer(self.model, self.X_data)
#         raw_shap = self.explainer(self.X_data)

#         # Fix: For classification models (3D matrix [samples, features, classes]), select Class 1
#         if len(raw_shap.shape) == 3:
#             self.shap_values = raw_shap[:, :, 1]
#         else:
#             self.shap_values = raw_shap

#     def get_feature_importance(self) -> pd.DataFrame:
#         vals = np.abs(self.shap_values.values).mean(axis=0)
#         return pd.DataFrame({
#             'feature': self.feature_names,
#             'mean_shap_value': vals
#         }).sort_values(by='mean_shap_value', ascending=False).reset_index(drop=True)



#     def get_instance_explanation(self, index: int) -> pd.DataFrame:
#         if index < 0 or index >= len(self.X_data):
#             raise IndexError(f"Index {index} out of bounds.")

#         return pd.DataFrame({
#             'feature': self.feature_names,
#             'feature_value': self.X_data.iloc[index].values,
#             'shap_impact': self.shap_values.values[index]
#         }).sort_values(by='shap_impact', key=abs, ascending=False).reset_index(drop=True)

#     def get_top_drivers(self, index: int, top_n: int = 3) -> dict:
#         df = self.get_instance_explanation(index)
        
#         # 1. Filter and sort positive drivers (largest positive impacts first)
#         pos_drivers = (
#             df[df['shap_impact'] > 0]
#             .sort_values(by='shap_impact', ascending=False)
#             .head(top_n)
#         )
        
#         # 2. Filter and sort negative drivers (most negative impacts first)
#         neg_drivers = (
#             df[df['shap_impact'] < 0]
#             .sort_values(by='shap_impact', ascending=True)
#             .head(top_n)
#         )
        
#         # 3. Missing Return Block
#         return {
#             "positive_drivers": pos_drivers[['feature', 'feature_value', 'shap_impact']].to_dict(orient='records'),
#             "negative_drivers": neg_drivers[['feature', 'feature_value', 'shap_impact']].to_dict(orient='records')
#         }

"""
Core explainability module for shap_analyzer.
"""

from typing import Optional
import pandas as pd
import numpy as np
import shap


class ModelExplainer:
    def __init__(
        self, 
        model, 
        X_data: pd.DataFrame, 
        sample_size: Optional[int] = None,
        class_index: int = 1
    ):
        self.model = model
        self.class_index = class_index
        self.feature_names = list(X_data.columns)

        if sample_size and sample_size < len(X_data):
            self.X_data = X_data.sample(n=sample_size, random_state=42)
        else:
            self.X_data = X_data.copy()

        self.explainer = shap.Explainer(self.model, self.X_data)
        raw_shap = self.explainer(self.X_data)

        # Handle 3D SHAP outputs (multi-class or binary classification)
        if len(raw_shap.shape) == 3:
            num_classes = raw_shap.shape[2]
            if self.class_index < 0 or self.class_index >= num_classes:
                raise ValueError(
                    f"class_index={self.class_index} is out of bounds. "
                    f"This model has {num_classes} classes (indexes 0 to {num_classes - 1})."
                )
            self.shap_values = raw_shap[:, :, self.class_index]
        else:
            self.shap_values = raw_shap

    def get_feature_importance(self) -> pd.DataFrame:
        """Returns global feature importance sorted by mean absolute SHAP value."""
        vals = np.abs(self.shap_values.values).mean(axis=0)
        return pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap_value': vals
        }).sort_values(by='mean_shap_value', ascending=False).reset_index(drop=True)

    def get_instance_explanation(self, index: int) -> pd.DataFrame:
        """Returns local feature impacts for a specific instance row."""
        if index < 0 or index >= len(self.X_data):
            raise IndexError(f"Index {index} out of bounds.")

        return pd.DataFrame({
            'feature': self.feature_names,
            'feature_value': self.X_data.iloc[index].values,
            'shap_impact': self.shap_values.values[index]
        }).sort_values(by='shap_impact', key=abs, ascending=False).reset_index(drop=True)

    def get_top_drivers(self, index: int, top_n: int = 3) -> dict:
        """Returns top N positive and negative features influencing a single prediction."""
        df = self.get_instance_explanation(index)
        
        pos_drivers = (
            df[df['shap_impact'] > 0]
            .sort_values(by='shap_impact', ascending=False)
            .head(top_n)
        )
        neg_drivers = (
            df[df['shap_impact'] < 0]
            .sort_values(by='shap_impact', ascending=True)
            .head(top_n)
        )
        
        return {
            "positive_drivers": pos_drivers[['feature', 'feature_value', 'shap_impact']].to_dict(orient='records'),
            "negative_drivers": neg_drivers[['feature', 'feature_value', 'shap_impact']].to_dict(orient='records')
        }


def analyze(
    model, 
    X_data: pd.DataFrame, 
    sample_size: Optional[int] = None, 
    class_index: int = 1
) -> ModelExplainer:
    """Helper function to initialize ModelExplainer directly."""
    return ModelExplainer(
        model, 
        X_data, 
        sample_size=sample_size, 
        class_index=class_index
    )