=========================
Installation Instructions
=========================

Follow these instructions (written assuming `Anaconda <https://www.anaconda.com/>`_ distribution in mind):

.. code-block:: powershell
  :linenos:

  conda create -p ./venv python=3.6.5 -y
  conda activate ./venv
  # install torch according to your cuda version 
  # https://pytorch.org/get-started/locally/
  conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y  
  pip install -e .