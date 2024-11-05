from rose import ResourceEngine, WorkflowEngine, Task

engine = ResourceEngine({'resource': 'local.localhost'})

flow = WorkflowEngine(engine=engine)

@flow
def task1(*args):
    return Task(executable='/bin/echo $RP_TASK_NAME')

@flow
def task2(*args):
    return Task(executable='/bin/echo $RP_TASK_NAME')

@flow
def task3(*args):
    return Task(executable='/bin/echo $RP_TASK_NAME')

# ====================================================
# Workflow-1: 1-layer funnel DAG
print('Running 1-layer funnel DAG workflow\n')
print("Shape:")
print("""
  task1      task2
     \\       /
       task3
""")
task3(task1(), task2())
flow.run()

# ====================================================
# Workflow-2: 2-layer funnel DAG
print('Running 2-layer funnel DAG workflow\n')
print("Shape:")
print("""
   task1      task2
     |          |
   task2      task1
     \\        /
       task3
""")
task3(task2(task1()), task1(task2()))
flow.run()

# ====================================================
# Workflow-3: Sequential Pipelines (Repeated Twice)
print('Running sequential pipelines\n')
print("Shape:")
print("""
   Run 1          Run 2
   task1          task1
      |             |
   task2          task2
      |             |
   task3          task3
""")
for i in range(2):
    task3(task2(task1()))
    flow.run()

# ====================================================
# Workflow-4: Concurrent Pipelines
print('Running concurrent pipelines\n')
print("Shape:")
print("""
   task1       task1
      |           |
   task2       task2
      |           |
   task3       task3
""")
for i in range(2):
    task3(task2(task1()))

flow.run()

engine.shutdown()
