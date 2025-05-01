# Instructions to configure RADICAL-Pilot

## Setup environment

```shell
export ACCOUNT=phy122    # chm155
cd $MEMBERWORK/$ACCOUNT  # working space for the virtual environment and repos

export PYTHONNOUSERSITE=True
module load cray-python
python3 -m venv ve.rp
. ve.rp/bin/activate

pip install git+https://github.com/radical-cybertools/radical.utils.git@devel
pip install git+https://github.com/radical-cybertools/radical.gtod.git
pip install git+https://github.com/radical-cybertools/radical.pilot.git@feature/hackathon
pip install git+https://github.com/radical-cybertools/radical.analytics.git

pip install git+https://github.com/radical-cybertools/radical.flow.git@feature/hackathon
pip install git+https://github.com/radical-cybertools/ROSE.git@feature/hackathon
```
NOTE: This environment is for RADICAL tools only, environment for scientific 
executables could be kept separately.

```shell
git clone --single-branch --branch feature/hackathon https://github.com/radical-cybertools/radical.pilot.git
git clone --single-branch --branch feature/hackathon https://github.com/radical-cybertools/ROSE.git
```

## Access resources (interactive job)

```shell
# NOTE: $ACCOUNT here may need an extra `_001` etc. for sub allocations
salloc -A $ACCOUNT -p batch --reservation=hackathon2 -t 2:00:00 -N 2
```

## Run test examples

- RADICAL-Pilot test example
```shell
cd $MEMBERWORK/$ACCOUNT/radical.pilot
vim examples/config.sh
# search for 'ornl.frontier'
# change 'chm155_003' to the value of your $ACCOUNT

. $MEMBERWORK/$ACCOUNT/ve.rp/bin/activate
./examples/00_getting_started.py ornl.frontier
```

- ROSE test example
```shell
cd $MEMBERWORK/$ACCOUNT/ROSE/examples/frontier_hackathon

. $MEMBERWORK/$ACCOUNT/ve.rp/bin/activate
python3 run_me_frontier.py
```