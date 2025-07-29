import torch.nn.functional as F
import numpy as np

class UQMetrics:
    def __init__(self, task_type):
        """
        task_type: 'classification' or 'regression'
        """, 
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type
    
    # ---- Classification metrics ----
    def predictive_entropy(self, mc_preds):
        mean_probs = np.mean(mc_preds, axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
        return entropy

    def mutual_information(self, mc_preds):
        mean_probs = np.mean(mc_preds, axis=0)
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
        mean_entropies = -np.sum(mc_preds * np.log(mc_preds + 1e-8), axis=2)
        expected_entropy = np.mean(mean_entropies, axis=0)
        mi = predictive_entropy - expected_entropy
        return mi

    def variation_ratio(self, mc_preds):
        #n_mc_samples: Number of MC samples
        n_mc_samples = mc_preds.shape[0]
        votes = np.argmax(mc_preds, axis=2)  # [n_mc_samples, N]
        mode_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
        mode_count = np.sum(votes == mode_vote, axis=0)
        vr = 1.0 - mode_count / n_mc_samples
        return vr

    def margin(self, mc_preds):
        mean_probs = np.mean(mc_preds, axis=0)
        part = np.partition(-mean_probs, 1, axis=1)
        top1 = -part[:, 0]
        top2 = -part[:, 1]
        margin = top1 - top2
        return margin
    
    # ---- Regression metrics ----
    
    def predictive_variance(self, mc_preds):
        return np.var(mc_preds, axis=0).squeeze()

    def predictive_interval_width(self, mc_preds, quantile=0.95):
        lower = np.percentile(mc_preds, (1 - quantile) / 2 * 100, axis=0)
        upper = np.percentile(mc_preds, (1 + quantile) / 2 * 100, axis=0)
        return (upper - lower).squeeze()

    def negative_log_likelihood(self, mc_preds, y_true):
        if self.task_type == 'classification':
            mean_probs = np.mean(mc_preds, axis=0)
            nll = -np.log(mean_probs[np.arange(len(y_true)), y_true] + 1e-8)
        else:
            mean_pred = np.mean(mc_preds, axis=0).squeeze()
            var_pred = np.var(mc_preds, axis=0).squeeze() + 1e-8
            nll = 0.5 * np.log(2 * np.pi * var_pred) + 0.5 * ((y_true - mean_pred) ** 2) / var_pred
        return nll

    def compute_uncertainty(self, mc_preds, y_true=None):
        """ Compute uncertainty metrics """
        if self.task_type == 'classification':

            # Predictive Entropy
            entropy = self.predictive_entropy(mc_preds)

            # Mutual Information (BALD)
            mi = self.mutual_information(mc_preds)

            # Variation ratio
            var_ratio = self.variation_ratio(mc_preds)
            
            # Margin
            margin = self.margin(mc_preds)

            # Negative Log Likelihood
            if y_true is not None:
                nll = self.negative_log_likelihood(mc_preds, y_true)

                return {
                    'predictive_entropy': entropy,
                    'mutual_information': mi,
                    'variation_ratio': var_ratio,
                    'margin': margin,
                    'negative_log_likelihood': nll
                }
            else:
                return {
                    'predictive_entropy': entropy,
                    'mutual_information': mi,
                    'variation_ratio': var_ratio,
                    'margin': margin
                }
        else:

            var = self.predictive_variance(mc_preds)
            interval_width = self.predictive_interval_width(mc_preds, quantile=0.95)
                        # Negative Log Likelihood
            if y_true is not None:
                nll = self.negative_log_likelihood(mc_preds, y_true)
                return {
                    'negative_log_likelihood': nll,
                    'predictive_variance': var,
                    'predictive_interval_width': interval_width
                }
            else:
                return {
                    'predictive_variance': var,
                    'predictive_interval_width': interval_width
                }

    def select_top_uncertain(self, mc_preds, y_true=None, n_instances=10, strategy='entropy'):
        """ Select samples based on a query strategy """
        uq = self.compute_uncertainty(mc_preds, y_true)

        if self.task_type == 'classification':
            if strategy == 'entropy':
                scores = uq['predictive_entropy']
                query_idx = np.argsort(-scores)[:n_instances]
            elif strategy == 'mutual_information':
                scores = uq['mutual_information']
                query_idx = np.argsort(-scores)[:n_instances]
            elif strategy == 'variation_ratio':
                scores = uq['variation_ratio']
                query_idx = np.argsort(-scores)[:n_instances]
            elif strategy == 'margin':
                scores = uq['margin']
                query_idx = np.argsort(scores)[:n_instances]  # smallest margin = most uncertain
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            if strategy == 'variance':
                scores = uq['predictive_variance']
                query_idx = np.argsort(-scores)[:n_instances]
            elif strategy == 'interval_width':
                scores = uq['predictive_interval_width']
                query_idx = np.argsort(-scores)[:n_instances]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return query_idx, scores[query_idx]
