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


class ActiveLearningMetrics(Enum):
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


    # Dictionary mapping each metric to its corresponding operator
    OPERATORS = {
        MODEL_ACCURACY: ">=",            # Accuracy should be greater than or equal to a threshold
        QUERY_EFFICIENCY: ">",           # Efficiency should be greater than a threshold
        LABELING_COST: "<",              # Cost should be less than a threshold
        DIVERSITY_OF_SELECTED_SAMPLES: ">",  # Diversity should be greater than a threshold
        UNCERTAINTY: "<",                # Uncertainty should be less than a threshold
        QUERY_REDUCTION: ">",            # Query reduction should be greater than a threshold
        AREA_UNDER_LEARNING_CURVE_AUC: ">",  # AUC should be greater than a threshold
        EXPECTED_MODEL_IMPROVEMENT_EMI: ">", # Improvement should be greater than a threshold
        SAMPLING_BIAS: "<",              # Bias should be less than a threshold
        PREDICTION_CONFIDENCE: ">",      # Confidence should be greater than a threshold
        CLASS_DISTRIBUTION_SHIFT: "<",   # Distribution shift should be less than a threshold
        CUMULATIVE_ERROR_REDUCTION: ">", # Error reduction should be greater than a threshold
        QUERY_SELECTION_STRATEGY_PERFORMANCE: ">",  # Strategy performance should be greater than a threshold
        NUMBER_OF_ITERATIONS: "<",       # Iterations should be less than a threshold
        F1_SCORE: ">",                   # F1 Score should be greater than a threshold
        PRECISION: ">",                  # Precision should be greater than a threshold
        RECALL: ">",                     # Recall should be greater than a threshold
        INFORMATION_GAIN: ">",           # Information gain should be greater than a threshold
        TIME_TO_CONVERGENCE: "<",        # Time to convergence should be less than a threshold
        MODEL_COMPLEXITY: "<",           # Model complexity should be minimized
        DATA_LABELING_REDUNDANCY: "<",   # Redundancy in labeled data should be minimized
        GENERALIZATION_ERROR: "<",       # Generalization error should be minimized
        LEARNING_RATE: "<",              # Learning rate should be minimized
        TRAINING_TIME: "<",              # Training time should be minimized
        CONFIDENCE_INTERVAL_WIDTH: "<",  # Confidence intervals should be narrow
        OVERFITTING: "<",                # Overfitting should be minimized
        ACTIVE_LEARNING_EFFICIENCY: ">", # Efficiency of the active learning loop should increase
        EXPLORATION_VS_EXPLOITATION: ">",# Balance exploration and exploitation (higher value preferred)
        MEAN_SQUARED_ERROR_MSE: "<",     # MSE should be minimized
    }

    @classmethod
    def is_supported_metric(cls, metric):
        return metric in cls.OPERATORS.value

    @classmethod
    def get_operator(cls, metric):
        return cls.OPERATORS.value.get(metric)
