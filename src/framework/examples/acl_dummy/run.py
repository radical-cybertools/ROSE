from data import InputFile, OutputFile
from rose import ResourceEngine, WorkflowEngine, Task

engine = ResourceEngine({'resource': 'local.localhost'})
flow = WorkflowEngine(engine=engine)
code_path = 'python3 /home/aymen/RADICAL/ROSE/src/framework/'

@flow
def t1(*args):
    return Task(executable=code_path+'prepare_data.py')

@flow
def t2(*args):
    return Task(executable=code_path+'train_model.py')

@flow
def t3(*args):
    return Task(executable=code_path+'query_selection.py') # this is one example of ACL (there are many)

@flow
def t4(*args):
    return Task(executable=code_path+'update_data.py')

@flow
def t5(*args):
    return Task(executable=code_path+'evaluate_model.py')


# Parameters
max_iterations = 1
target_accuracy = 0.95 # terminate cond (overall performance)

tt1 = t1()

results = []

@flow.as_async
def run_wf():
    ttx = t1()
    tt2 = t2(tt1) 
    tt3 = t3(tt2, tt1)
    tt4 = t4(tt1, tt3)
    t4_res = tt4.result()

    if tt4.done():
        print('adding another task')
        tt5 = t5(tt4, tt2)

# run 10 non-blocking workflows in parallel
for i in range(10):
    results.append(run_wf())

[t.result() for t in results]

engine.shutdown()
