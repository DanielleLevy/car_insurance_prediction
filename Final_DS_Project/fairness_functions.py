# fairness_pipeline.py

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.preprocessing import CorrelationRemover
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Set seeds for reproducibility.
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

##############################
# Pre-processing Candidates
##############################

def pre_none(X, y, sensitive):
    """No pre-processing: return X, y, and sensitive unchanged."""
    return X.copy(), y.copy(), sensitive.copy()

def pre_correlation_remover(X, y, sensitive):
    """
    Apply Fairlearn's CorrelationRemover to decorrelate features from the sensitive attribute.
    Assumes X is a pandas DataFrame and sensitive is a pandas Series.
    
    The sensitive column is converted to numeric if needed.
    """
    X_temp = X.copy()
    X_temp["sensitive"] = sensitive.values

    # Convert sensitive to numeric if not already.
    if not pd.api.types.is_numeric_dtype(X_temp["sensitive"]):
        X_temp["sensitive"] = pd.factorize(X_temp["sensitive"])[0]

    corr_remover = CorrelationRemover(sensitive_feature_ids=["sensitive"])
    X_transformed = corr_remover.fit_transform(X_temp)
    
    # Convert back to DataFrame using original column names (except "sensitive").
    desired_cols = X_temp.columns.drop("sensitive")
    X_transformed_df = pd.DataFrame(X_transformed, columns=desired_cols)
    
    # In this case, sensitive remains unchanged.
    return X_transformed_df, y.copy(), sensitive.copy()

def pre_sensitive_resampling(X, y, sensitive, method='oversample'):
    """
    Resample the dataset to balance the sensitive attribute distribution.
    Splits the data by the sensitive attribute, then oversamples or undersamples each group
    so that all groups have equal representation.
    
    Returns:
      X_res, y_res, sens_res: Resampled features, targets, and sensitive attribute.
    """
    df = X.copy()
    df["target"] = y
    df["sensitive"] = sensitive.values
    
    group_counts = df["sensitive"].value_counts()
    if method == 'oversample':
        target_count = group_counts.max()
    elif method == 'undersample':
        target_count = group_counts.min()
    else:
        raise ValueError("Method must be either 'oversample' or 'undersample'.")
    
    resampled_dfs = []
    for group_val, group_df in df.groupby("sensitive"):
        if method == 'oversample':
            resampled_group = group_df.sample(n=target_count, replace=True, random_state=42)
        else: ## for undersamlpling
            resampled_group = group_df.sample(n=target_count, replace=False, random_state=42)
        resampled_dfs.append(resampled_group)
    
    df_resampled = pd.concat(resampled_dfs)
    X_res = df_resampled.drop(columns=["target", "sensitive"])
    y_res = df_resampled["target"]
    sens_res = df_resampled["sensitive"]
    
    # Reset indices.
    X_res.reset_index(drop=True, inplace=True)
    y_res = y_res.reset_index(drop=True)
    sens_res = sens_res.reset_index(drop=True)

    return X_res, y_res, sens_res

##############################
# In-processing Candidates (Model Training)
##############################

def in_baseline(X_train, y_train, sensitive):
    """Train a baseline Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def in_reweighting(X_train, y_train, sensitive):
    """Train Logistic Regression using reweighting based on sensitive group frequencies."""
    # Here, 'sensitive' is the preprocessed sensitive Series.
    group_counts = sensitive.value_counts().to_dict()
    sample_weights = sensitive.map(lambda x: 1.0 / group_counts[x])
    sample_weights = sample_weights / sample_weights.mean()
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

def in_expgrad_dp(X_train, y_train, sensitive):
    """Train using ExponentiatedGradient with Demographic Parity constraint."""
    base_estimator = LogisticRegression(random_state=42, max_iter=1000)
    constraint = DemographicParity()
    eg_model = ExponentiatedGradient(estimator=base_estimator, constraints=constraint, eps=0.01)
    eg_model.fit(X_train, y_train, sensitive_features=sensitive)
    return eg_model

def in_expgrad_eo(X_train, y_train, sensitive):
    """Train using ExponentiatedGradient with Equalized Odds constraint."""
    base_estimator = LogisticRegression(random_state=42, max_iter=1000)
    constraint = EqualizedOdds()
    eg_model = ExponentiatedGradient(estimator=base_estimator, constraints=constraint, eps=0.01)
    eg_model.fit(X_train, y_train, sensitive_features=sensitive)
    return eg_model

##############################
# Post-processing Candidates
##############################

def post_none(model, X_test, y_test, sensitive):
    """No post-processing: return the model as-is."""
    return model

def post_threshold_dp(model, X_test, y_test, sensitive):
    """Apply ThresholdOptimizer with Demographic Parity constraint."""
    postproc = ThresholdOptimizer(estimator=model, constraints="demographic_parity", prefit=True)
    postproc.fit(X_test, y_test, sensitive_features=sensitive)
    return postproc

def post_threshold_eo(model, X_test, y_test, sensitive):
    """Apply ThresholdOptimizer with Equalized Odds constraint."""
    postproc = ThresholdOptimizer(estimator=model, constraints="equalized_odds", prefit=True)
    postproc.fit(X_test, y_test, sensitive_features=sensitive)
    return postproc

##############################
# Evaluation Function
##############################

def evaluate_model(model, X_test, y_test, sensitive, method_label=""):
    """
    Evaluate a model using standard performance metrics and fairness metrics.
    Returns a dictionary of metrics.
    """
    try:
        y_pred = model.predict(X_test, sensitive_features=sensitive)
    except TypeError:
        y_pred = model.predict(X_test)
        
    metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "Demographic_parity": float(demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive)),
    "Equalized_odds": equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive)
    }
    

    return metrics


def run_experiments(pre_methods, in_methods, post_methods, X_train, y_train, sens_train,
                    X_test, y_test, sens_test):
    """
    Runs experiments over combinations of pre-processing, in-processing, and post-processing methods.
    Returns a dictionary mapping configuration names to evaluation metrics.
    
    Each pre-processing method now returns (X_transformed, y_transformed, sensitive_transformed).
    For test data, if the pre-processing method changes the number of samples (e.g. SensitiveResampling),
    we instead use the identity transformation (pre_none) so that test set sizes remain consistent.
    """
    results = {}
    
    for pre_name, pre_func in pre_methods.items():
        # Apply pre-processing to training data.
        X_train_pre, y_train_pre, sens_train_pre = pre_func(X_train, y_train, sens_train)
        
        # For the test set, if the pre-processing method is sensitive resampling,
        # then use pre_none to leave the test set unchanged.
        if pre_name.lower() == "sensitiveresampling":
            X_test_pre, _, sens_test_pre = pre_none(X_test, y_test, sens_test)
        else:
            X_test_pre, _, sens_test_pre = pre_func(X_test, y_test, sens_test)
        
        for in_name, in_func in in_methods.items():
            # Train the model using the in-processing method.
            model = in_func(X_train_pre, y_train_pre, sens_train_pre)
            
            for post_name, post_func in post_methods.items():
                # Apply post-processing.
                if post_name.lower() == "none":
                    final_model = model
                else:
                    final_model = post_func(model, X_test_pre, y_test, sens_test_pre)
                    
                config_name = f"Pre-processing: {pre_name}. In-training: {in_name}. Post-processing:{post_name}"
                metrics = evaluate_model(final_model, X_test_pre, y_test, sens_test_pre, method_label=config_name)
                results[config_name] = metrics
                
    return results


def filter_results(results, f1_threshold=0.90, dp_threshold=0.10, eo_threshold=0.10):
    """
    Filters experiment results based on user-defined thresholds.
    
    Parameters:
        results (dict): Dictionary mapping configuration names to metric dictionaries.
        accuracy_threshold (float): Minimum acceptable accuracy.
        dp_threshold (float): Maximum acceptable Demographic Parity Difference.
        eo_threshold (float): Maximum acceptable Equalized Odds Difference.
    
    Returns:
        dict: Filtered dictionary of results that satisfy the thresholds.
    """
    filtered = {}
    for config, metrics in results.items():
        if (metrics["f1_score"] >= f1_threshold and
            metrics["Demographic_parity"] <= dp_threshold and
            metrics["Equalized_odds"] <= eo_threshold):
            filtered[config] = metrics
    return filtered

def select_best_by_metric(filtered_results, metric="dp_diff"):
    """
    Selects the best configuration from filtered results based on a specific metric.
    
    Parameters:
        filtered_results (dict): Filtered dictionary of experiment results.
        metric (str): The metric to optimize ('dp_diff' or 'eo_diff'). Lower is better.
        
    Returns:
        tuple: (best_config, metrics) where best_config is the configuration name and
               metrics is the metric dictionary for that configuration.
    """
    best_config = None
    best_value = None
    for config, metrics in filtered_results.items():
        value = metrics.get(metric, None)
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_config = config
    return best_config, filtered_results.get(best_config, None)

def pareto_frontier(results, objectives):
    """
    Compute the Pareto frontier from a dictionary of results.
    
    Parameters:
      results (dict): A dictionary mapping configuration names to metrics.
                      Each value is a dict, e.g.:
                      {"accuracy": 0.95, "dp_diff": 0.1, "eo_diff": 0.12}
      objectives (dict): A dictionary specifying the optimization direction
                         for each metric.
                         For example, {"accuracy": True, "dp_diff": False, "eo_diff": False}
                         means we want to maximize accuracy and minimize both dp_diff and eo_diff.
    
    Returns:
      dict: A dictionary containing only the configurations that are Pareto optimal.
    """
    frontier = {}
    for config, metrics in results.items():
        dominated = False
        for other_config, other_metrics in results.items():
            if other_config == config:
                continue
            
            # Assume other_config "dominates" config if for every objective the performance is at least as good,
            # and for at least one objective it is strictly better.
            better_or_equal = True
            strictly_better = False
            for obj, maximize in objectives.items():
                if maximize:
                    # For a metric we want to maximize, other should be >= current.
                    if other_metrics[obj] < metrics[obj]:
                        better_or_equal = False
                        break
                    if other_metrics[obj] > metrics[obj]:
                        strictly_better = True
                else:
                    # For a metric we want to minimize, other should be <= current.
                    if other_metrics[obj] > metrics[obj]:
                        better_or_equal = False
                        break
                    if other_metrics[obj] < metrics[obj]:
                        strictly_better = True
            
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier[config] = metrics
    return frontier

