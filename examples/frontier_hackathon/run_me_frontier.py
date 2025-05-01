import os
import sys

from rose.learner import ActiveLearner
from radical.flow import RadicalExecutionBackend, Task

engine = RadicalExecutionBackend({"resource": "ornl.frontier",
                                  "runtime": 15,
                                  "exit_on_error" : True,
                                  "project"  : "chm155_003",
                                  "queue"    : "batch",
                                  "schema"   : "local",
                                  "cores"    : 10,
                                  "gpus"     : 8})

acl = ActiveLearner(engine)
code_path = f"{sys.executable} {os.getcwd()}"

# ============================
# Define all utility tasks for the workflow
# ============================

@acl.utility_task
def pre_process_modes(*args):
    """
    Preprocess the raw XGC .bp files by:
      - Loading the specified 14 variables
      - Separating n=0 and n≠0 modes
      - Computing flux-surface averages
    Outputs intermediate processed data for further feature engineering.
    """
    print('Preprocessing starting 1st')
    return Task(executable=f'/bin/sleep 5 && /bin/date')

@acl.utility_task
def normalize_and_log(*args):
    """
    Normalize the preprocessed variables:
      - Normalize variables by their corresponding flux-surface averages
      - Compute the logarithm of flux-surface averages
    Prepares normalized and log-transformed features for dataset assembly.
    """
    print('Normalizing starting 2nd')
    return Task(executable=f'/bin/sleep 5 && /bin/date')

@acl.utility_task
def prepare_dataset(*args):
    """
    Prepare the final machine learning dataset:
      - Assemble 56 input features + 5 constant features
      - Build input-target pairs for training
      - Split or organize data appropriately for model training
    """
    print('Preparing dataset starting 3rd')
    return Task(executable=f'/bin/sleep 5 && /bin/date')

@acl.utility_task
def train_model(*args):
    """
    Train a neural network model:
      - Defines and trains a simple MLP with 4 layers
      - Uses prepared input features to predict dA||/dt
      - Saves the trained model artifacts for evaluation
    """
    print('Training model starting 4th')
    return Task(executable=f'/bin/sleep 5 && /bin/date')

@acl.utility_task
def evaluate_model(*args):
    """
    Evaluate the trained model's performance:
      - Tests the model on held-out validation or test set
      - Computes model accuracy or other relevant metrics
      - Determines if the model meets performance thresholds
    """
    print('Evaluating model starting 5th')
    return Task(executable=f'/bin/sleep 5 && /bin/date')

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

    print('Results:')
    print([s.result() for s in [step1, step2, step3, step4, step5]])

    print('Workflow completed successfully!')

run_workflow()
engine.shutdown()
