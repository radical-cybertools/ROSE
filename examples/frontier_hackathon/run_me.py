import os
import sys

from rose.learner import ActiveLearner
from radical.flow import RadicalExecutionBackend, Task

engine = RadicalExecutionBackend({'runtime': 30,
                                  'resource': 'local.localhost'})

custom_acl = ActiveLearner(engine)
code_path = f"{sys.executable} {os.getcwd()}"

# ============================
# Define all utility tasks for the workflow
# ============================

@custom_acl.utility_task
def pre_process_modes(*args):
    """
    Preprocess the raw XGC .bp files by:
      - Loading the specified 14 variables
      - Separating n=0 and n≠0 modes
      - Computing flux-surface averages
    Outputs intermediate processed data for further feature engineering.
    """
    return Task(executable=f'{code_path}/01_preprocess_modes.py')

@custom_acl.utility_task
def normalize_and_log(*args):
    """
    Normalize the preprocessed variables:
      - Normalize variables by their corresponding flux-surface averages
      - Compute the logarithm of flux-surface averages
    Prepares normalized and log-transformed features for dataset assembly.
    """
    return Task(executable=f'{code_path}/02_normalize_and_log.py')

@custom_acl.utility_task
def prepare_dataset(*args):
    """
    Prepare the final machine learning dataset:
      - Assemble 56 input features + 5 constant features
      - Build input-target pairs for training
      - Split or organize data appropriately for model training
    """
    return Task(executable=f'{code_path}/03_prepare_dataset.py')

@custom_acl.utility_task
def train_model(*args):
    """
    Train a neural network model:
      - Defines and trains a simple MLP with 4 layers
      - Uses prepared input features to predict dA||/dt
      - Saves the trained model artifacts for evaluation
    """
    return Task(executable=f'{code_path}/04_train_model.py')

@custom_acl.utility_task
def evaluate_model(*args):
    """
    Evaluate the trained model's performance:
      - Tests the model on held-out validation or test set
      - Computes model accuracy or other relevant metrics
      - Determines if the model meets performance thresholds
    """
    return Task(executable=f'{code_path}/05_evaluate_model.py')

# ============================
# Define and run the full workflow
# ============================

def run_workflow():
    """
    Execute the complete end-to-end workflow:
      1. Preprocess raw simulation data
      2. Normalize and log-transform features
      3. Prepare training and testing datasets
      4. Train the neural network model
      5. Evaluate the trained model
    """
    step1 = pre_process_modes()
    step2 = normalize_and_log(step1)
    step3 = prepare_dataset(step2)
    step4 = train_model(step3)
    step5 = evaluate_model(step4)

run_workflow()
engine.shutdown()
