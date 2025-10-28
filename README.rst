Reinforcement Signal Control (RESCO) Benchmark
==============================================
.. figure:: docs/source/_static/maps.png
  :alt: Six real-world traffic scenarios in RESCO.

|

RESCO provides an interface for traffic network simulation and includes a set of algorithms for traffic signal control.

**Quick start**

The command below will run the IDQN algorithm on the Cologne Single Signal scenario without using libsumo. Libsumo is faster but
requires linux.

.. code-block:: bash

    cd RESCO/resco_benchmark && pip install -e ../[pfrl,torch]
    python main.py @cologne1 @IDQN libsumo:False save_console_log:False gui:True

Features
--------
- Real-world traffic scenarios from three cities: Cologne, Luxembourg, and Salt Lake City.
- Observed and modeled demands from real-world data.
- Salt Lake City scenarios simulate one year of recorded traffic.
- Multiple state, reward, and action space configurations.
- StateBuilder and RewardBuilder interfaces allow for parameterized state and reward functions.
- Three static (non-learning) traffic signal control algorithms: Fixed Time, Max Pressure, and Max Wave.
- Integration with continuous learning benchmark algorithms from COOM.
- Integration with PFRL library algorithms.
- Interfaces with SUMO via subscription API for faster simulation.

Installation
------------

Please visit the `installation docs <https://github.com/Pi-Star-Lab/RESCO/blob/master/docs/source/installation.rst>`_.


Versions
--------
Release v1.0 - RESCO benchmark as of NeurIPS 2021

Release v2.0 - Refactored codebase, more algorithms, saltlake scenarios

------------
Citing RESCO
------------
If you use RESCO in your research, please cite the following paper:
`Reinforcement Learning Benchmarks for Traffic Signal Control <https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/f0935e4cd5920aa6c7c996a5ee53a70f-Abstract-round1.html>`_

.. code-block:: latex

    @inproceedings{ault2021reinforcement,
      title={Reinforcement Learning Benchmarks for Traffic Signal Control},
      author={James Ault and Guni Sharon},
      booktitle={Proceedings of the Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021) Datasets and Benchmarks Track},
      month={December},
      year={2021}
    }

