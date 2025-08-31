import numpy as np
import matplotlib.pyplot as plt

# ---- Decorator registry for UQ algorithms ----
UQ_REGISTRY = {}

def register_uq(name):
    """Decorator to register a UQ metric"""
    def decorator(func):
        UQ_REGISTRY[name] = func
        return func
    return decorator


class UQMetrics:
    def __init__(self, task_type):
        """
        task_type: 'classification' or 'regression'
        """
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type

#
#*****************************  
    def _validate_inputs(self, mc_preds):
        """Safeguard to check input dimensions"""
        if not isinstance(mc_preds, np.ndarray):
            raise TypeError("mc_preds must be a numpy array")
        
        if self.task_type == 'classification':
            # Expected: [n_mc_samples, n_instances, n_classes]
            if mc_preds.ndim != 3:
                raise ValueError(
                    f"For classification, mc_preds must have 3 dimensions "
                    f"[n_mc_samples, n_instances, n_classes], got shape {mc_preds.shape}"
                )
        else:
            # Expected: [n_mc_samples, n_instances] (regression outputs)
            if mc_preds.ndim != 2:
                raise ValueError(
                    f"For regression, mc_preds must have 2 dimensions "
                    f"[n_mc_samples, n_instances], got shape {mc_preds.shape}"
                )
        return True

#
#*****************************  
    # ---- Classification metrics ----
    @register_uq("predictive_entropy")
    def predictive_entropy(self, mc_preds):
        self._validate_inputs(mc_preds)
        mean_probs = np.mean(mc_preds, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
        return entropy

#
#*****************************  
    @register_uq("mutual_information")
    def mutual_information(self, mc_preds):
        self._validate_inputs(mc_preds)
        mean_probs = np.mean(mc_preds, axis=0)
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
        mean_entropies = -np.sum(mc_preds * np.log(mc_preds + 1e-8), axis=2)
        expected_entropy = np.mean(mean_entropies, axis=0)
        mi = predictive_entropy - expected_entropy
        return mi

#
#*****************************  
    @register_uq("variation_ratio")
    def variation_ratio(self, mc_preds):
        self._validate_inputs(mc_preds)
        n_mc_samples = mc_preds.shape[0]
        votes = np.argmax(mc_preds, axis=2)  # [n_mc_samples, N]
        mode_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
        mode_count = np.sum(votes == mode_vote, axis=0)
        vr = 1.0 - mode_count / n_mc_samples
        return vr

#
#*****************************  
    @register_uq("margin")
    def margin(self, mc_preds):
        self._validate_inputs(mc_preds)
        mean_probs = np.mean(mc_preds, axis=0)
        part = np.partition(-mean_probs, 1, axis=1)
        top1 = -part[:, 0]
        top2 = -part[:, 1]
        margin = top1 - top2
        return margin

#
#*****************************  
    # ---- Regression metrics ----
    @register_uq("predictive_variance")
    def predictive_variance(self, mc_preds):
        self._validate_inputs(mc_preds)
        return np.var(mc_preds, axis=0).squeeze()

#
#*****************************  
    @register_uq("predictive_interval_width")
    def predictive_interval_width(self, mc_preds, quantile=0.95):
        self._validate_inputs(mc_preds)
        lower = np.percentile(mc_preds, (1 - quantile) / 2 * 100, axis=0)
        upper = np.percentile(mc_preds, (1 + quantile) / 2 * 100, axis=0)
        return (upper - lower).squeeze()

#
#*****************************  
    @register_uq("negative_log_likelihood")
    def negative_log_likelihood(self, mc_preds, y_true):
        self._validate_inputs(mc_preds)
        if self.task_type == 'classification':
            mean_probs = np.mean(mc_preds, axis=0)
            nll = -np.log(mean_probs[np.arange(len(y_true)), y_true] + 1e-8)
        else:
            mean_pred = np.mean(mc_preds, axis=0).squeeze()
            var_pred = np.var(mc_preds, axis=0).squeeze() + 1e-8
            nll = 0.5 * np.log(2 * np.pi * var_pred) + 0.5 * ((y_true - mean_pred) ** 2) / var_pred
        return nll

#
#*****************************  
    def compute_uncertainty(self, mc_preds, y_true=None):
        """Compute all registered UQ metrics"""
        self._validate_inputs(mc_preds)

        results = {}
        for name, func in UQ_REGISTRY.items():
            try:
                if name == "negative_log_likelihood" and y_true is None:
                    continue
                results[name] = func(self, mc_preds) if name != "negative_log_likelihood" else func(self, mc_preds, y_true)
            except Exception as e:
                results[name] = f"Error: {e}"
        return results

#
#*****************************  
    def select_top_uncertain(self, mc_preds, k=10, metric=None, y_true=None, plot=None):
        """
        Select top-k most uncertain samples according to a registered metric.

        Args:
            mc_preds: numpy array of MC predictions
            k: number of samples to select
            metric: string, one of UQ_REGISTRY.keys() (default depends on task type)
            y_true: required only if metric == 'negative_log_likelihood'

        Returns:
            indices of top-k most uncertain samples, and their scores
        """
        self._validate_inputs(mc_preds)

        # Default metric
        if metric is None:
            metric = "predictive_entropy" if self.task_type == "classification" else "predictive_variance"

        if metric not in UQ_REGISTRY:
            raise ValueError(f"Metric '{metric}' is not registered. Available: {list(UQ_REGISTRY.keys())}")

        func = UQ_REGISTRY[metric]
        scores = func(self, mc_preds) if metric != "negative_log_likelihood" else func(self, mc_preds, y_true)

        top_indices = np.argsort(-scores)[:k]

        if plot == 'plot_top_uncertain':
            self.plot_top_uncertain(mc_preds, k=k, metric=metric, y_true=y_true, save_path='uncertain_plot.png', show_error=True)
        elif plot == 'plot_scatter_uncertainty':
            self.scatter_uncertainty(mc_preds, metric=metric, y_true=y_true, save_path='scatter_plot.png')
        return top_indices, scores[top_indices]

#
#*****************************  
    def plot_top_uncertain(
        self, mc_preds, k=10, metric=None, y_true=None, save_path=None, show_error=False
    ):
        """
        Plot the top-k most uncertain samples in a scientific style.

        Args:
            mc_preds: numpy array of MC predictions
            k: number of samples
            metric: metric name (defaults depend on task type)
            y_true: true labels (optional, for annotation and NLL)
            save_path: optional, if provided saves plot to file
            show_error: if True, also plots error bars (std across MC samples)
        """
        top_indices, top_scores = self.select_top_uncertain(
            mc_preds, k=k, metric=metric, y_true=y_true
        )

        # Error estimates (std of MC scores) if requested
        errors = None
        if show_error:
            if self.task_type == "classification":
                # Example: std of entropy across MC draws
                per_mc_scores = []
                if metric == "predictive_entropy":
                    for i in range(mc_preds.shape[0]):
                        mean_probs = mc_preds[i]
                        per_mc_scores.append(
                            -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
                        )
                if per_mc_scores:
                    errors = np.std(per_mc_scores, axis=0)[top_indices]
            elif self.task_type == "regression":
                errors = np.std(mc_preds, axis=0).squeeze()[top_indices]

        plt.figure(figsize=(8, 6))
        y_pos = np.arange(len(top_scores))

        bars = plt.barh(
            y_pos,
            top_scores,
            xerr=errors if errors is not None else None,
            color="steelblue",
            alpha=0.85,
            capsize=4,
        )
        plt.yticks(y_pos, [f"Sample {i}" for i in top_indices], fontsize=11)
        plt.xlabel("Uncertainty Score", fontsize=12)
        plt.ylabel("Sample Index", fontsize=12)
        plt.title(f"Top-{k} Most Uncertain Samples ({metric})", fontsize=14)
        plt.gca().invert_yaxis()  # Highest uncertainty at top

        # Overlay labels if classification
        if self.task_type == "classification":
            mean_probs = np.mean(mc_preds, axis=0)
            pred_labels = np.argmax(mean_probs, axis=1)

            for bar, idx in zip(bars, top_indices):
                txt = f"pred={pred_labels[idx]}"
                if y_true is not None:
                    txt += f", true={y_true[idx]}"
                plt.text(
                    bar.get_width() + 0.01 * max(top_scores),
                    bar.get_y() + bar.get_height() / 2,
                    txt,
                    va="center",
                    ha="left",
                    fontsize=10,
                    color="darkred" if y_true is not None and pred_labels[idx] != y_true[idx] else "black",
                )

        # For regression: annotate true values if available
        elif self.task_type == "regression" and y_true is not None:
            mean_preds = np.mean(mc_preds, axis=0).squeeze()
            for bar, idx in zip(bars, top_indices):
                txt = f"pred={mean_preds[idx]:.3f}, true={y_true[idx]:.3f}"
                plt.text(
                    bar.get_width() + 0.01 * max(top_scores),
                    bar.get_y() + bar.get_height() / 2,
                    txt,
                    va="center",
                    ha="left",
                    fontsize=10,
                    color="darkred",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

#
#*****************************            
    def scatter_uncertainty(
        self, mc_preds, metric=None, y_true=None, save_path=None
    ):
        """
        Scatter plot of uncertainty vs. predictions (or residuals).

        Args:
            mc_preds: numpy array of MC predictions
            metric: metric name (defaults depend on task type)
            y_true: true labels (optional, for residuals/calibration)
            save_path: optional, if provided saves plot to file
        """
        # Compute uncertainty scores
        #nll, entropy, variance = self.compute_metrics(mc_preds, y_true)
        if metric is None:
            metric = "predictive_entropy" if self.task_type == "classification" else "variance"

        try:
            func = UQ_REGISTRY[metric]
            scores = func(self, mc_preds) if metric != "negative_log_likelihood" else func(self, mc_preds, y_true)
        except:
            raise ValueError(f"Unsupported metric {metric}")

        plt.figure(figsize=(7, 6))

        if self.task_type == "classification":
            mean_probs = np.mean(mc_preds, axis=0)
            pred_labels = np.argmax(mean_probs, axis=1)

            if y_true is not None:
                correct = pred_labels == y_true
                plt.scatter(scores[correct], np.max(mean_probs, axis=1)[correct],
                            c="green", alpha=0.6, label="Correct")
                plt.scatter(scores[~correct], np.max(mean_probs, axis=1)[~correct],
                            c="red", alpha=0.6, label="Incorrect")
                plt.ylabel("Max Predicted Probability", fontsize=12)
            else:
                plt.scatter(scores, np.max(mean_probs, axis=1), c="steelblue", alpha=0.6)
                plt.ylabel("Max Predicted Probability", fontsize=12)

            plt.xlabel(f"Uncertainty ({metric})", fontsize=12)
            plt.title("Uncertainty vs Confidence", fontsize=14)
            if y_true is not None:
                plt.legend()

        elif self.task_type == "regression":
            mean_preds = np.mean(mc_preds, axis=0).squeeze()

            if y_true is not None:
                residuals = np.abs(mean_preds - y_true)
                plt.scatter(scores, residuals, c="purple", alpha=0.6)
                plt.ylabel("Residual |y - ŷ|", fontsize=12)
                plt.title("Uncertainty vs Error", fontsize=14)
            else:
                plt.scatter(scores, mean_preds, c="steelblue", alpha=0.6)
                plt.ylabel("Predicted Mean", fontsize=12)
                plt.title("Uncertainty vs Prediction", fontsize=14)

            plt.xlabel(f"Uncertainty ({metric})", fontsize=12)

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()