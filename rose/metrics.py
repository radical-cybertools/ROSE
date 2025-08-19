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
        MODEL_ACCURACY: GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
        QUERY_EFFICIENCY: GREATER_THAN_THRESHOLD,
        LABELING_COST: LESS_THAN_THRESHOLD,
        DIVERSITY_OF_SELECTED_SAMPLES: GREATER_THAN_THRESHOLD,
        UNCERTAINTY: LESS_THAN_THRESHOLD,
        QUERY_REDUCTION: GREATER_THAN_THRESHOLD,
        AREA_UNDER_LEARNING_CURVE_AUC: GREATER_THAN_THRESHOLD,
        EXPECTED_MODEL_IMPROVEMENT_EMI: GREATER_THAN_THRESHOLD,
        SAMPLING_BIAS: LESS_THAN_THRESHOLD,
        PREDICTION_CONFIDENCE: GREATER_THAN_THRESHOLD,
        CLASS_DISTRIBUTION_SHIFT: LESS_THAN_THRESHOLD,
        CUMULATIVE_ERROR_REDUCTION: GREATER_THAN_THRESHOLD,
        QUERY_SELECTION_STRATEGY_PERFORMANCE: GREATER_THAN_THRESHOLD,
        NUMBER_OF_ITERATIONS: LESS_THAN_THRESHOLD,
        F1_SCORE: GREATER_THAN_THRESHOLD,
        PRECISION: GREATER_THAN_THRESHOLD,
        RECALL: GREATER_THAN_THRESHOLD,
        INFORMATION_GAIN: GREATER_THAN_THRESHOLD,
        TIME_TO_CONVERGENCE: LESS_THAN_THRESHOLD,
        MODEL_COMPLEXITY: LESS_THAN_THRESHOLD,
        DATA_LABELING_REDUNDANCY: LESS_THAN_THRESHOLD,
        GENERALIZATION_ERROR: LESS_THAN_THRESHOLD,
        LEARNING_RATE: LESS_THAN_THRESHOLD,
        TRAINING_TIME: LESS_THAN_THRESHOLD,
        CONFIDENCE_INTERVAL_WIDTH: LESS_THAN_THRESHOLD,
        OVERFITTING: LESS_THAN_THRESHOLD,
        ACTIVE_LEARNING_EFFICIENCY: GREATER_THAN_THRESHOLD,
        EXPLORATION_VS_EXPLOITATION: GREATER_THAN_THRESHOLD,
        MEAN_SQUARED_ERROR_MSE: LESS_THAN_THRESHOLD,
    }

    @classmethod
    def is_supported_metric(cls, metric):
        return metric in cls.OPERATORS.value

    @classmethod
    def get_operator(cls, metric):
        return cls.OPERATORS.value.get(metric)
