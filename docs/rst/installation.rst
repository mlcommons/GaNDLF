=========================
Installation Instructions
=========================

*************
Prerequisites
*************

- Python3 with a preference for `conda <https://www.anaconda.com/>`_
- `CUDA <https://developer.nvidia.com/cuda-download>`_ and a compatible `cuDNN version <https://developer.nvidia.com/cudnn>`_ installed system-wide

*************
Instructions
*************

Follow these instructions (written assuming `Anaconda <https://www.anaconda.com/>`_ distribution in mind):

.. code-block:: powershell
  :linenos:

  conda create -p ./venv python=3.6.5 -y
  conda activate ./venv
  # install torch according to your cuda version 
  # https://pytorch.org/get-started/locally/
  conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y  
  pip install -e .