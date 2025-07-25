from enum import Enum

MODEL_ACCURACY = "model_accuracy"
QUERY_EFFICIENCY = "query_efficiency"
LABELING_COST = "labeling_cost"
DIVERSITY_OF_SELECTED_SAMPLES = "diversity_of_selected_samples"
UNCERTAINTY = "uncertainty"
QUERY_REDUCTION = "query_reduction"
AREA_UNDER_LEARNING_CURVE_AUC = "area_under_learning_curve_auc"
EXPECTED_MODEL_IMPROVEMENT_EMI = "expected_model_improvement_emi"
SAMPLING_BIAS = "sampling_bias"
PREDICTION_CONFIDENCE = "prediction_confidence"
CLASS_DISTRIBUTION_SHIFT = "class_distribution_shift"
CUMULATIVE_ERROR_REDUCTION = "cumulative_error_reduction"
QUERY_SELECTION_STRATEGY_PERFORMANCE = "query_selection_strategy_performance"
NUMBER_OF_ITERATIONS = "number_of_iterations"
F1_SCORE = "f1_score"
PRECISION = "precision"
RECALL = "recall"
INFORMATION_GAIN = "information_gain"
TIME_TO_CONVERGENCE = "time_to_convergence"
MODEL_COMPLEXITY = "model_complexity"
DATA_LABELING_REDUNDANCY = "data_labeling_redundancy"
GENERALIZATION_ERROR = "generalization_error"
LEARNING_RATE = "learning_rate"
TRAINING_TIME = "training_time"
CONFIDENCE_INTERVAL_WIDTH = "confidence_interval_width"
OVERFITTING = "overfitting"
ACTIVE_LEARNING_EFFICIENCY = "active_learning_efficiency"
EXPLORATION_VS_EXPLOITATION = "exploration_vs_exploitation"
MEAN_SQUARED_ERROR_MSE = "mean_squared_error_mse"

# Metrics Operators
LESS_THAN_THRESHOLD = "<"
GREATER_THAN_THRESHOLD = ">"
EQUAL_TO_THRESHOLD = "="
LESS_THAN_OR_EQUAL_TO_THRESHOLD = "<="
GREATER_THAN_OR_EQUAL_TO_THRESHOLD = ">="


class LearningMetrics(Enum):
    MODEL_ACCURACY = "model_accuracy"
    QUERY_EFFICIENCY = "query_efficiency"
    LABELING_COST = "labeling_cost"
    DIVERSITY_OF_SELECTED_SAMPLES = "diversity_of_selected_samples"
    UNCERTAINTY = "uncertainty"
    QUERY_REDUCTION = "query_reduction"
    AREA_UNDER_LEARNING_CURVE_AUC = "area_under_learning_curve_auc"
    EXPECTED_MODEL_IMPROVEMENT_EMI = "expected_model_improvement_emi"
    SAMPLING_BIAS = "sampling_bias"
    PREDICTION_CONFIDENCE = "prediction_confidence"
    CLASS_DISTRIBUTION_SHIFT = "class_distribution_shift"
    CUMULATIVE_ERROR_REDUCTION = "cumulative_error_reduction"
    QUERY_SELECTION_STRATEGY_PERFORMANCE = "query_selection_strategy_performance"
    NUMBER_OF_ITERATIONS = "number_of_iterations"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    INFORMATION_GAIN = "information_gain"
    TIME_TO_CONVERGENCE = "time_to_convergence"
    MODEL_COMPLEXITY = "model_complexity"
    DATA_LABELING_REDUNDANCY = "data_labeling_redundancy"
    GENERALIZATION_ERROR = "generalization_error"
    LEARNING_RATE = "learning_rate"
    TRAINING_TIME = "training_time"
    CONFIDENCE_INTERVAL_WIDTH = "confidence_interval_width"
    OVERFITTING = "overfitting"
    ACTIVE_LEARNING_EFFICIENCY = "active_learning_efficiency"
    EXPLORATION_VS_EXPLOITATION = "exploration_vs_exploitation"
    MEAN_SQUARED_ERROR_MSE = "mean_squared_error_mse"

    LESS_THAN_THRESHOLD = "<"
    GREATER_THAN_THRESHOLD = ">"
    EQUAL_TO_THRESHOLD = "="
    LESS_THAN_OR_EQUAL_TO_THRESHOLD = "<="
    GREATER_THAN_OR_EQUAL_TO_THRESHOLD = ">="

    # Dictionary mapping each metric to its corresponding operator
    OPERATORS = {
        MODEL_ACCURACY: GREATER_THAN_OR_EQUAL_TO_THRESHOLD,            # Accuracy should be greater than or equal to a threshold
        QUERY_EFFICIENCY: GREATER_THAN_THRESHOLD,                      # Efficiency should be greater than a threshold
        LABELING_COST: LESS_THAN_THRESHOLD,                            # Cost should be less than a threshold
        DIVERSITY_OF_SELECTED_SAMPLES: GREATER_THAN_THRESHOLD,         # Diversity should be greater than a threshold
        UNCERTAINTY: LESS_THAN_THRESHOLD,                              # Uncertainty should be less than a threshold
        QUERY_REDUCTION: GREATER_THAN_THRESHOLD,                       # Query reduction should be greater than a threshold
        AREA_UNDER_LEARNING_CURVE_AUC: GREATER_THAN_THRESHOLD,         # AUC should be greater than a threshold
        EXPECTED_MODEL_IMPROVEMENT_EMI: GREATER_THAN_THRESHOLD,        # Improvement should be greater than a threshold
        SAMPLING_BIAS: LESS_THAN_THRESHOLD,                            # Bias should be less than a threshold
        PREDICTION_CONFIDENCE: GREATER_THAN_THRESHOLD,                 # Confidence should be greater than a threshold
        CLASS_DISTRIBUTION_SHIFT: LESS_THAN_THRESHOLD,                 # Distribution shift should be less than a threshold
        CUMULATIVE_ERROR_REDUCTION: GREATER_THAN_THRESHOLD,            # Error reduction should be greater than a threshold
        QUERY_SELECTION_STRATEGY_PERFORMANCE: GREATER_THAN_THRESHOLD,  # Strategy performance should be greater than a threshold
        NUMBER_OF_ITERATIONS: LESS_THAN_THRESHOLD,                     # Iterations should be less than a threshold
        F1_SCORE: GREATER_THAN_THRESHOLD,                              # F1 Score should be greater than a threshold
        PRECISION: GREATER_THAN_THRESHOLD,                             # Precision should be greater than a threshold
        RECALL: GREATER_THAN_THRESHOLD,                                # Recall should be greater than a threshold
        INFORMATION_GAIN: GREATER_THAN_THRESHOLD,                      # Information gain should be greater than a threshold
        TIME_TO_CONVERGENCE: LESS_THAN_THRESHOLD,                      # Time to convergence should be less than a threshold
        MODEL_COMPLEXITY: LESS_THAN_THRESHOLD,                         # Model complexity should be minimized
        DATA_LABELING_REDUNDANCY: LESS_THAN_THRESHOLD,                 # Redundancy in labeled data should be minimized
        GENERALIZATION_ERROR: LESS_THAN_THRESHOLD,                     # Generalization error should be minimized
        LEARNING_RATE: LESS_THAN_THRESHOLD,                            # Learning rate should be minimized
        TRAINING_TIME: LESS_THAN_THRESHOLD,                            # Training time should be minimized
        CONFIDENCE_INTERVAL_WIDTH: LESS_THAN_THRESHOLD,                # Confidence intervals should be narrow
        OVERFITTING: LESS_THAN_THRESHOLD,                              # Overfitting should be minimized
        ACTIVE_LEARNING_EFFICIENCY: GREATER_THAN_THRESHOLD,            # Efficiency of the active learning loop should increase
        EXPLORATION_VS_EXPLOITATION: GREATER_THAN_THRESHOLD,           # Balance exploration and exploitation (higher value preferred)
        MEAN_SQUARED_ERROR_MSE: LESS_THAN_THRESHOLD,                   # MSE should be minimized
    }

    @classmethod
    def is_supported_metric(cls, metric):
        return metric in cls.OPERATORS.value

    @classmethod
    def get_operator(cls, metric):
        return cls.OPERATORS.value.get(metric)
