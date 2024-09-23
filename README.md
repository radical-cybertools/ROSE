# ROSE
What is ROSE:

ROSE is a framework that
1). Integrate AL-coupled HPC workflow with active learning to improve the performance
2). Provide a list of pre-build AL policy and allow user to customize their own active learning policy
3). Can be used to compare the performance of various active learning policy that user are interested
4). Provide a number of execution pattern for running active learning workflow with different bottleneck



How to build:

Simply install radical on top of the simulation and training environment



How to run two examples:

For the exalearn example, it is implemented using bash script instead of rct, so following the follow steps:
1). Run "prepare_data_dir.py" to setup experiment directory
2). In workflow directory, choose the workflow that you want to execute
If files in this directory fails to execute correctly, please use this repo:
https://github.com/GKNB/AL-Exalearn-phase-2
and use the scripts in branch AL-two-uncertainty-v2 for serial workflow, and branch AL-two-uncertainty-stream for stream workflow

For the diffusion solver example, it is implemented using rct, but only support serial workflow with a single instance, and parallel workflow with different seed. To run them, just do
python launch_workflow_basic.py
