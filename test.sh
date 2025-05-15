#!/bin/bash

clear

if [ ! -d "$(pwd)/libtorch" ]; then
    echo "libtorch directory not found. Downloading libtorch..."
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip -O libtorch.zip
    echo "Unzipping libtorch..."
    unzip libtorch.zip
    rm libtorch.zip
    echo "libtorch downloaded and extracted."
fi

# create the .net file
echo "Creating the .net file with python..."
echo "----- Python --------------------------------------------------------------------"
python create_net.py
echo "---------------------------------------------------------------------------------"

# compile the C++ code
echo "Compiling net_forward_mex.cpp..."
libtorch_path="$(pwd)/libtorch"
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="$libtorch_path"
make

cd ..

# test standalone C++ version
echo "Testing standalone C++ version..."
echo "----- C++ -----------------------------------------------------------------------"
./build/net_forward
echo "---------------------------------------------------------------------------------"
echo "Standalone version test completed."

# test MATLAB version
# # for now this is forced:
# export LD_PRELOAD="$(pwd)/libtorch/lib/libtorch_cpu.so:$(pwd)/libtorch/lib/libtorch.so"
export LD_PRELOAD="$(pwd)/libtorch/lib/libtorch.so"
echo "----- Matlab --------------------------------------------------------------------"
# start MATLAB -> run the script forward_test.m -> exit
matlab -nodisplay -nosplash -nodesktop -r "run('forward_test.m'); exit;"
echo "---------------------------------------------------------------------------------"
echo "MATLAB version test completed."


