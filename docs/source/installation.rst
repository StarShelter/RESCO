====================
Installation
====================


Retrieve the source code from the `GitHub repository <https://github.com/Pi-Star-Lab/RESCO.git>`_.

Python's package manager, `pip <https://pip.pypa.io/en/stable/>`_, is used to install RESCO. The following will install
RESCO in editable mode. The `Simulation of Urban Mobility <https://eclipse.dev/sumo/>`_ (SUMO) will be installed via pip as well.

.. code-block:: bash

    cd RESCO
    pip install -e .

Optionally install the dependencies for hyper-parameter tuning [optuna], building this documentation [docs], or
executing the FMA2C/MA2C algorithms [fma2c]. The option [fma2c] requires a compatible python version for the outdated
version of tensorflow used, suggested is version 3.6.

.. code-block:: bash

    pip install -e .[optuna,docs,fma2c,cl]

Installation isolation is recommended via
`miniconda <https://docs.anaconda.com/miniconda/#quick-command-line-install>`_. After installing miniconda, create a new
environment and install the dependencies using pip with the following commands.

.. code-block:: bash

    conda create -n resco python=3.12
    conda activate resco
    python -m pip install -e .[optuna,docs]


------------------
Nvidia GPU Support
------------------
If you want to use the GPU for the neural network based algorithms and you have a current version of CUDA installed you
do not need to install anything else. If you do not have a current version of CUDA installed you will need to reference
the `torch installation instructions <https://pytorch.org/get-started/locally/>`_ for your CUDA version. An example is
given below for CUDA 11.X.


.. code-block:: bash

    pip install torch --index-url https://download.pytorch.org/whl/cu118

---------------------------------
Texas A&M University HPRC Install
---------------------------------
Given the slowness of the login nodes you might want to chain these commands and
nohup & the process and come back in a day. Make sure your private key is in ~/.ssh
and doesn't need a password. Clone the repo in your home directory, not on scratch.
Scratch is extremely slow, but home directories don't have space for dependency installation.
A result of this is that loading dependencies will be slow when you run main.py initially.
Uninstalling the benchmark at th end will allow SLURM to use fast local storage.

.. code-block:: bash

    ml Miniconda3/24.11.1 GCCcore/13.3.0 git/2.45.1
    git clone git@github.com:jault/RESCO.git
    git clone git@github.com:hyintell/COOM.git
    conda create --yes -p $SCRATCH/conda/resco python==3.10
    source activate $SCRATCH/conda/resco
    python -m pip install --yes ~/RESCO/.[cl]
    python -m pip install --yes ~/COOM/.
    conda clean --all --yes
    python -m pip uninstall --yes resco_benchmark
    python -m pip uninstall --yes COOM
    ml Miniconda3/24.11.1 GCCcore/12.3.0 p7zip/17.04
    rm -rf ~/RESCO/.git ~/COOM/.git ~/COOM/assets
    7za a -mx=9 RESCO.7z RESCO COOM
    rm -rf ~/RESCO ~/COOM
    python -m pip install conda-pack
    conda pack -p $SCRATCH/conda/resco

Use the SLURM scheduler to run many jobs. Create a shell script using one of the experiment_runners (e.g. germany.sh).
Copy the script and experiment_runner/resco.slurm to your home directory on GRACE. Then on GRACE run:

.. code-block:: bash

    dos2unix germany.sh
    nohup sh germany.sh &

You can use `squeue -u <your-username>` to check the status of your jobs. Use 'squeue -j <job-id>' if you have many running.
In the queue you can see the node running the job in NODELIST. You can use ssh to connect to the node (e.g. ssh c261) then
use 'cd /tmp/job.<job-id>' to go to the job directory.