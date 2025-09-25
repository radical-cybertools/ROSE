import numpy as np
from pathlib import Path

# ---- Decorator registry for UQ algorithms ----
UQ_REGISTRY = {}


def register_uq(name):
    """Decorator to register a UQ metric"""

    def decorator(func):
        UQ_REGISTRY[name] = func
        return func

    return decorator


class UQScorer:
    def __init__(self, task_type):
        """
        task_type: 'classification' or 'regression'
        """
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be classification or regression")
        self.task_type = task_type

    #
    # *****************************
    def _validate_inputs(self, mc_preds, y_true=None):
        """Safeguard to check input dimensions"""
        if not isinstance(mc_preds, np.ndarray):
            try:
                mc_preds = np.array(mc_preds)
            except Exception as err:
                raise TypeError(
                    f"Fail to convert {type(mc_preds)} mc_preds to numpy array"
                ) from err

        if self.task_type == "classification":
            # Expected: [n_mc_samples, n_instances, n_classes]
            if mc_preds.ndim != 3:
                mc_preds = np.squeeze(mc_preds)
                if mc_preds.ndim != 3:
                    raise ValueError(
                        f"For classification, mc_preds must have 3 dimensions"
                        f" [n_mc_samples, n_instances, n_classes],"
                        f" got shape {mc_preds.shape}"
                    )
        else:
            # Expected: [n_mc_samples, n_instances] (regression outputs)
            if mc_preds.ndim != 2:
                mc_preds = np.squeeze(mc_preds)
                if mc_preds.ndim != 2:
                    raise ValueError(
                        f"For regression, mc_preds must have 2 dimensions "
                        f"[n_mc_samples, n_instances], "
                        f"got shape {mc_preds.shape}"
                    )
        if y_true is not None:
            try:
                y_true = np.array(y_true)
            except Exception as err:
                raise TypeError(
                    f"Fail to convert {type(y_true)} y_true to numpy array"
                ) from err
            if self.task_type == "classification":
                if y_true.ndim > 2:
                    y_true = np.squeeze(y_true)
                    if y_true.ndim > 2:
                        raise ValueError(
                            f"For classification, y_true must have "
                            f"2 dimensions [n_instances, n_classes], "
                            f"got shape {y_true.shape}"
                        )
            else:
                if y_true.ndim > 1:
                    y_true = np.squeeze(y_true)
                    if y_true.ndim > 1:
                        raise ValueError(
                            f"For regression, y_true must have 1 "
                            f"dimension [n_instances], "
                            f"y_true shape is {y_true.shape}"
                        )
        return mc_preds, y_true

    #
    # *****************************
    # ---- Classification metrics ----
    @register_uq("predictive_entropy")
    def predictive_entropy(self, mc_preds):
        mc_preds, _ = self._validate_inputs(mc_preds)
        mean_probs = np.mean(mc_preds, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
        return entropy

    #
    # *****************************
    @register_uq("mutual_information")
    def mutual_information(self, mc_preds):
        mc_preds, _ = self._validate_inputs(mc_preds)
        mean_probs = np.mean(mc_preds, axis=0)
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs 
                                                + 1e-8), axis=1)
        mean_entropies = -np.sum(mc_preds * np.log(mc_preds + 1e-8), axis=2)
        expected_entropy = np.mean(mean_entropies, axis=0)
        mi = predictive_entropy - expected_entropy
        return mi

    #
    # *****************************
    @register_uq("variation_ratio")
    def variation_ratio(self, mc_preds):
        mc_preds, _ = self._validate_inputs(mc_preds)
        n_mc_samples = mc_preds.shape[0]
        votes = np.argmax(mc_preds, axis=2)  # [n_mc_samples, N]
        mode_vote = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=votes
        )
        mode_count = np.sum(votes == mode_vote, axis=0)
        vr = 1.0 - mode_count / n_mc_samples
        return vr

    #
    # *****************************
    @register_uq("margin")
    def margin(self, mc_preds):
        mc_preds, _ = self._validate_inputs(mc_preds)
        mean_probs = np.mean(mc_preds, axis=0)
        part = np.partition(-mean_probs, 1, axis=1)
        top1 = -part[:, 0]
        top2 = -part[:, 1]
        margin = top1 - top2
        return margin

    #
    # *****************************
    # ---- Regression metrics ----
    @register_uq("predictive_variance")
    def predictive_variance(self, mc_preds):
        mc_preds, _ = self._validate_inputs(mc_preds)
        return np.var(mc_preds, axis=0).squeeze()

    #
    # *****************************
    @register_uq("predictive_interval_width")
    def predictive_interval_width(self, mc_preds, quantile=0.95):
        mc_preds, _ = self._validate_inputs(mc_preds)
        lower = np.percentile(mc_preds, (1 - quantile) / 2 * 100, axis=0)
        upper = np.percentile(mc_preds, (1 + quantile) / 2 * 100, axis=0)
        return (upper - lower).squeeze()

    #
    # *****************************
    @register_uq("negative_log_likelihood")
    def negative_log_likelihood(self, mc_preds, y_true):
        mc_preds, y_true = self._validate_inputs(mc_preds, y_true)
        if self.task_type == "classification":
            mean_probs = np.mean(mc_preds, axis=0)
            nll = -np.log(mean_probs[np.arange(len(y_true)), y_true] + 1e-8)
        else:
            mean_pred = np.mean(mc_preds, axis=0).squeeze()
            var_pred = np.var(mc_preds, axis=0).squeeze() + 1e-8
            nll = (
                0.5 * np.log(2 * np.pi * var_pred)
                + 0.5 * ((y_true - mean_pred) ** 2) / var_pred
            )

        return nll

    #
    # *****************************
    def compute_uncertainty(self, mc_preds, y_true=None):
        """Compute all registered UQ metrics"""
        mc_preds, y_true = self._validate_inputs(mc_preds, y_true)

        results = {}
        for name, func in UQ_REGISTRY.items():
            try:
                if name == "negative_log_likelihood" and y_true is None:
                    continue
                results[name] = (
                    func(self, mc_preds)
                    if name != "negative_log_likelihood"
                    else func(self, mc_preds, y_true)
                )
            except Exception as e:
                results[name] = f"Error: {e}"
        return results

    #
    # *****************************
    def select_top_uncertain(self, mc_preds, k=10, metric=None, y_true=None):
        """
        Select top-k most uncertain samples according to a registered metric.

        Args:
            mc_preds: numpy array of MC predictions
            k: number of samples to select
            metric: string, one of UQ_REGISTRY.keys()
            (default depends on task type)
            y_true: required only if metric == 'negative_log_likelihood'

        Returns:
            indices of top-k most uncertain samples, and their scores
        """
        mc_preds, y_true = self._validate_inputs(mc_preds, y_true)

        # Default metric
        if metric is None:
            metric = (
                "predictive_entropy"
                if self.task_type == "classification"
                else "predictive_variance"
            )

        if metric not in UQ_REGISTRY:
            raise ValueError(
                f"Metric '{metric}' is not registered. "
                f"Available: {list(UQ_REGISTRY.keys())}"
            )

        func = UQ_REGISTRY[metric]
        scores = (
            func(self, mc_preds)
            if metric != "negative_log_likelihood"
            else func(self, mc_preds, y_true)
        )

        top_indices = np.argsort(-scores)[:k]

        return top_indices, scores[top_indices]
