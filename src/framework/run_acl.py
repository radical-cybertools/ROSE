from active_learner import ActiveLearner
from rose import Task, ResourceEngine

engine = ResourceEngine({'resource': 'local.localhost'})
acl = ActiveLearner(engine)


# Define and register the simulation task
@acl.simulation_task
def task_a(*args):
    return Task(executable=f'/bin/echo SIMULATION')

# Define and register the training task
@acl.training_task
def task_b(*args):
    return Task(executable=f'/bin/echo TRAIN')

# Define and register the active learning task
@acl.active_learn_task
def task_c(*args):
    return Task(executable=f'/bin/echo ACL')

# Now, call the tasks and teach
t1 = task_a()
t2 = task_b()
t3 = task_c()

# Start the teaching process
acl.teach(max_iter=10)
engine.shutdown()
