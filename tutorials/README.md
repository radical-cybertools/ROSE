# ROSE Tutorials

This folder contains tutorial environments for **ROSE**.

---

## Available Tutorials

- **00-active-learning** – Active learning / Gaussian Process tutorials
- **01-reinforcement-learning** – Reinforcement learning / PyTorch + Gym tutorials
- **03-highly-parallel-surrogates** – Highly parallel surrogate training with ensemble GP models
- **04-al-algorithm-selector** – Running multiple AL algorithms in parallel and selecting the best

Each tutorial has its own optional dependencies. The common dependencies are required for all tutorials.

---

## Installation

It is recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv rose_tutorial
source rose_tutorial/bin/activate
```

## Install dependencies for a specific tutorial
To install the corresponding requirement for any tutorial use the `pip install -e` + `the tutorial id` for example:
```bash
# Tutorial 00al (00-active-learning)
pip install -e ".[00al]"

# Tutorial 01rl (01-reinforcement-learning)
pip install -e ".[01rl]"

# Tutorial 03hp (03-highly-parallel-surrogates)
pip install -e ".[03hp]"

# Tutorial 04als (04-al-algorithm-selector)
pip install -e ".[04als]"
```

## Usage
Once installed, you can open the Jupyter notebooks in this folder or run the Python scripts corresponding to each tutorial.
