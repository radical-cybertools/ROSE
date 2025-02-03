# Installation

To get started with the ROSE, you'll first need to install it on your machine or the targeted cluster.
Below are the steps to install it on different operating systems. For a full list of the supported HPC
machines, please refer to the following link: [RADICAL-Pilot Supported HPC Machines](https://radicalpilot.readthedocs.io/en/stable/supported.html)

## For Linux and macOS 🐧

1. Clone the latest version from the [official website](https://github.com/radical-cybertools/ROSE).
   ```
   git clone https://github.com/radical-cybertools/ROSE.git
   ```
2. Run the following commands to install ROSE and its dependencies:
    ```bash
    cd ROSE
    pip install .
    ```

## For Windows Machines 🖥️

1. Download the Windows WSL installer from the [official website](https://apps.microsoft.com/detail/9pdxgncfsczv?hl=en-US&gl=US).
2. Setup you WSL user name and password.
3. Make sure you have Python 3.9 or higher in your WSL as follows:
    ```
    python --version
    ```
4. create new pip virtual env: 
    ```
    python3 -m venv rose_env
    ```
5. Activate the env:
   ```
   source rose_env/bin/activate
   ```
6. Clone the latest version from the [official website](https://github.com/radical-cybertools/ROSE).
   ```
   git clone https://github.com/radical-cybertools/ROSE.git
   ```
7. Run the following commands to install ROSE and its dependencies:
    ```bash
    cd ROSE
    pip install .
    ```

If you encounter any issues, refer to the [Issues Section](https://github.com/radical-cybertools/ROSE/issues).
