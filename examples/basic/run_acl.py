from rose.learner import ActiveLearner
from rose.engine import Task, ResourceEngine

engine = ResourceEngine({'resource': 'local.localhost'})
acl = ActiveLearner(engine)


# Define and register the simulation task
@acl.simulation_task
def simulation(*args):
    return Task(executable=f'/bin/echo SIMULATION')

# Define and register the training task
@acl.training_task
def training(*args):
    return Task(executable=f'/bin/echo TRAIN')

# Define and register the active learning task
@acl.active_learn_task
def active_learn(*args):
    return Task(executable=f'/bin/echo ACL')

# Now, call the tasks and teach
t1 = simulation()
t2 = training()
t3 = active_learn()

# Start the teaching process
acl.teach(max_iter=10)
engine.shutdown()
