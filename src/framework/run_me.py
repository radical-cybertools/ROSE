from roseee import RoseWorkflow, RoseEngine
from rose import SimulationTask, ActiveLearnTask, TrainingTask

engine = RoseEngine({'resource': 'local.localhost'})

flow = RoseWorkflow(engine=engine)

@flow
def pi1(*args):
    return SimulationTask(executable='/bin/date && /bin/sleep 2 && /bin/echo $RP_TASK_NAME')

@flow
def pi2(*args):
    return ActiveLearnTask(executable='/bin/date && /bin/sleep 2 && /bin/echo $RP_TASK_NAME')

@flow
def pi3(*args):
    return ActiveLearnTask(executable='/bin/date && /bin/sleep 2 && /bin/echo $RP_TASK_NAME')

@flow
def mean(*args):
    return TrainingTask(executable='/bin/date && /bin/sleep 2 && /bin/echo $RP_TASK_NAME')

@flow
def decision(*args):
    return TrainingTask(executable='/bin/echo I AM DONE')

# Running the same set of workflows 5 time
# decision(mean(pi1(), pi2(), pi3()))
# flow.run()

# flow.clear()


#mean(pi1(),
#     pi2(),
#     pi3(pi1(),
#         pi2()))

#flow.run()

#mean(pi1(),
#     pi2(pi1()),
#     pi3(pi1(),
#         pi2()))


# case-1 basic
#  pi1   pi2
#   \     /
#     pi3
#      |
#     mean
# mean(pi3(pi1(), pi2()))


# case-2
mean(pi3(mean(pi1()), mean(pi2())))

flow.run()
engine.shutdown()
