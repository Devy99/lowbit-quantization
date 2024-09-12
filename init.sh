#!/bin/bash

# ======== INIT AQLM REPOSITORY ========
dirs=("1-calibration-datasets" "2-quantize-models" "3-finetuning")
for dir in "${dirs[@]}"; do
    cd ./$dir
    if [ ! -d "AQLM" ]; then
        echo "Cloning AQLM repository in $dir..."
        git clone https://github.com/Vahe1994/AQLM
        cd AQLM
        git checkout f13ca200bf8595c549867e082d75869f44193db5
        cd ..
    fi
    cd ..

    # Set up the custom data utils
    rm ./$dir/AQML/src/datautils.py
    cp ./1-calibration-datasets/custom_datautils.py ./$dir/AQLM/src/datautils.py

    if [ $dir == "1-calibration-datasets" ]; then
        rm ./1-calibration-datasets/AQML/src/datautils.py
        cp ./1-calibration-datasets/custom_datautils.py ./1-calibration-datasets/AQLM/src/custom_datautils.py
        rm ./1-calibration-datasets/AQLM/fetch_calibration_dataset.py
        cp ./1-calibration-datasets/fetch_calibration_dataset.py ./1-calibration-datasets/AQLM/fetch_calibration_dataset.py
    fi
done

# ======== INIT MULTIPL-E REPOSITORY ========
cd ./utils
if [ ! -d "MultiPL-E" ]; then
    echo "Cloning MultiPL-E repository..."
    git clone https://github.com/nuprl/MultiPL-E.git
    cd MultiPL-E
    git checkout 19a25675e6df678945a6e3da0dca9473265b0055
    cd ..
fi

# Set up the custom automodel.py and automodel_aqlm.py
rm ./MultiPL-E/automodel.py
cp ./automodel.py ./MultiPL-E
cp ./automodel_aqlm.py ./MultiPL-E

cd ..

