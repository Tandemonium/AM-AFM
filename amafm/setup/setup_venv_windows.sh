#!/bin/bash

if [ ! -d "$2/$3" ]; then
    # install virtual environment library and create venv
    printf ">>> install virtual environment\n"
    if [ $# -eq 4 ]
    then
        $4/python.exe -m pip install --upgrade pip
        $4/Scripts/pip.exe install virtualenv
        $4/python.exe -m venv "$2/$3"
    else
        python -m pip install --upgrade pip
        pip install virtualenv
        python -m venv "$2/$3"
    fi

    # activate venv
    source "$2/$3/Scripts/activate"
    
    # upgrade pip
    printf "\n>>> upgrade pip\n"
    python -m pip install --upgrade pip
    
    # create jupyter-kernel
    printf "\n>>> install jupyter-kernel\n"
    pip install ipykernel
    python -m ipykernel install --user --name=$3

    # install python libraries
    printf "\n>>> install requirements\n"
    pip install -r setup/requirements.txt

    #printf "\n>>> install pyTorch\n"
    #pip install torch==2.2.2+$1 torchvision==0.17.2+$1 torchaudio==2.2.2+$1 --index-url https://download.pytorch.org/whl/$1
    #pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+$1.html
fi

printf "\n>>> All set - the shell can be closed now\n"
printf "\n>>> Restart the IDE and select the new kernel !\n"

$SHELL