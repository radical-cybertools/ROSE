from rosee import RoseWorkflow, RoseEngine
from rose import SimulationTask, ActiveLearnTask, TrainingTask

engine = RoseEngine({'resource': 'local.localhost'})

flow = RoseWorkflow(engine=engine)

@flow
def pi1():
    return SimulationTask(executable='/bin/date && /bin/sleep 2 && /bin/echo $RP_TASK_NAME')

@flow
def pi2():
    return ActiveLearnTask(executable='/bin/date && /bin/sleep 2 && /bin/echo $RP_TASK_NAME')

@flow
def pi3():
    return ActiveLearnTask(executable='/bin/date && /bin/sleep 2 && /bin/echo $RP_TASK_NAME')

@flow
def mean(pi1, pi2, pi3):
    return TrainingTask(executable='/bin/date && /bin/sleep 2 && /bin/echo $RP_TASK_NAME')

@flow
def decision(mean):
    return TrainingTask(executable='/bin/echo I AM DONE')

# Running the same set of workflows 5 time
decision(mean(pi1(), pi2(), pi3()))
flow.run()

flow.clear()


mean(pi1(), pi2(), pi3(pi1, pi2))
flow.run()

engine.shutdown()
