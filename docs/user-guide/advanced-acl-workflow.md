To support the rapid advancement of AL techniques, ROSE offers an additional approach to building and executing complex AL workflows.

In this example, we demonstrate how to express an AL workflow with different levels of parallelism. What does that mean?

In some cases, AL workflows may require the execution of N simulation or training tasks **concurrently**. But not only that—additionally, they may also require the submission of M AL workflows concurrently. This introduces two levels of parallelism: one at the task level and another at the AL workflow level. Such an approach is possible and can be easily expressed and executed using ROSE's **custom AL policy**.

```sh            
                             (N AL WFs in Parallel)
          +-------------------+               +-------------------+
          |      AL WF 1      |               |      AL WF 2      |   
          +-------------------+               +-------------------+
                   │                                    │
  +----------------+-----------------+  +----------------+-----------------+
  |       (N tasks Parallel)         |  |       (N AL tasks Parallel)      |
  +---------------+  +---------------+  +---------------+  +---------------+
  | Simulation 1  |  | Simulation 2  |  | Simulation 1  |  | Simulation 2  |
  +---------------+  +---------------+  +---------------+  +---------------+
          |                |                    |                 |
  +---------------+  +---------------+  +---------------+  +---------------+
  |  Training 1   |  |  Training 2   |  |  Training 1   |  |  Training 2   |
  +---------------+  +---------------+  +---------------+  +---------------+
          |                |                    |                 |
        (...)            (...)                (...)             (...)    
```

Since we have already learned how to deploy and load ROSE, and how to instruct it to use different resources, we will skip this part and focus only on expressing the AL workflow.

First, let's express our tasks:

```python
code_path = f'{sys.executable} {os.getcwd()}'

# Define and register the simulation task
@custom_acl.simulation_task
def simulation(*args):
    return Task(executable=f'{code_path}/simulation.py')

# Define and register the training task
@custom_acl.training_task
def training(*args):
    return Task(executable=f'{code_path}/training.py')

# Define and register the active learning task
@custom_acl.active_learn_task
def active_learn(*args):
    return Task(executable=f'{code_path}/active_learn.py')

# Defining the stop criterion with a metric (MSE in this case)
@custom_acl.as_stop_criterion(metric_name=MODEL_ACCURACY, threshold=0.99)
def check_accuracy(*args):
    return Task(executable=f'{code_path}/check_accuracy.py')

# Special task that can perform different operation (example post-processing)
@custom_acl.utility_task()
def post_process_simulation(*args):
    return Task(executable=f'{code_path}/post_process_simulation.py')
```


Now, lets express the core custom AL policy logic. The example below will:


* Submits 5 AL workflows in parallel using `as_async` decorator. (Workflow parallelism)
* Each workflow will run for 10 iterations sequentially. 
* Each iteration will submit 3 simulation tasks in parallel. (task parallelism)

!!! tip
    `as_async` allows the `blocking` workflows to be `non-blocking` while keep managing their execution via ROSE in the background.

```python
@custom_acl.as_async
def teach():
    # 10 iterations of active learning
    for acl_iter in range(10):
        print(f'Starting Iteration-{acl_iter}')
        simulations = []
        for i in range(3):
            # run 3 simulations in parallel
            simulations.append(simulation())

        post_process_simulation(*simulations)

        # Now run training and active_learn
        train = training(*simulations)
        active = active_learn(simulations, train)

        if check_accuracy(active):
            print('Accuracy met the threshold')
            break
```

Now, lets submit 5 AL workflows for execution:

```python
# invoke the custom/user-defined teach() method
al_workflows = []
for i in range(5):
    al_workflows.append(teach())

[wf.result() for wf in al_workflows]
```