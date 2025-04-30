import os
import sys

from rose.learner import ActiveLearner
from radical.flow import RadicalExecutionBackend, Task

backend = RadicalExecutionBackend({'runtime': 30,
                                  'resource': 'local.localhost'})

custom_acl = ActiveLearner(backend)
code_path = f"{sys.executable} {os.getcwd()}"

# ============================
# Define all utility tasks for the workflow
# ============================

@custom_acl.utility_task
def ml_service(*args):
    """
    ml service that exposes the model to be used in the workflow
    """
    return Task(executable=f'{code_path}/service_task.py', mode='task.service')


@custom_acl.utility_task
def ml_inference1(*args):
    """
    Perform service requests
    """
    return Task(executable=f'curl http://localhost:8000/status && /bin/time')


@custom_acl.utility_task
def ml_inference2(*args):
    """
    Perform service requests
    """
    return Task(executable=f'curl http://localhost:8000/status && /bin/time')


service = ml_service()
infer1 = ml_inference1(service)
infer2 = ml_inference2(service)

print([infer.result() for infer in [infer1, infer2]])

backend.shutdown()
