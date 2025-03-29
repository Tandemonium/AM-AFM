#!/bin/bash

if [ ! -d "$2/$3" ]; then
    # install python
    echo ">>> install python3.12"
    sudo apt-get install python3.12
    alias python=python3.12

    # install virtual environment library and create venv
    echo ">>> install virtual environment"
    sudo apt install python3.12-venv
    python -m venv "$2/$3"

    # activate venv
    source "$2/$3/bin/activate"

    # upgrade pip
    echo ">>> upgrade pip"
    pip install --upgrade pip

    # create jupyter-kernel
    echo ">>> install jupyter-kernel"
    pip install ipykernel
    python -m ipykernel install --user --name=$3

    # install python libraries
    echo ">>> install requirements"
    pip install -r src/setup/requirements.txt

    echo ">>> install pyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$1
    #pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+$1.html
fi

echo ">>> All set - the shell can be closed now"
echo ">>> Restart the IDE and select the new kernel !"

$SHELL