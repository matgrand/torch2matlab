#!/bin/bash

clear


if [ ! -d "$(pwd)/onnxruntime-linux-x64-1.22.0" ]; then
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
    tar -xzf onnxruntime-linux-x64-1.22.0.tgz
    rm onnxruntime-linux-x64-1.22.0.tgz
    echo "onnxruntime downloaded and extracted."
fi

export onnx_dir="$(pwd)/onnxruntime-linux-x64-1.22.0"

echo "onnxruntime directory: $onnx_dir"

# compile the C++ code
echo "Compiling..."
cp onnx_cmake/CMakeLists.txt CMakeLists.txt  # copy the CMakeLists.txt file to the current directory
rm -rf build  # remove the build directory if it exists
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="$onnx_dir"
make

cd ..
rm CMakeLists.txt  # remove the copied CMakeLists.txt file
echo "Compilation completed."

# create the .net file with python
echo "Creating the .net file with python..."
echo "----- Python --------------------------------------------------------------------"
python create_net.py
echo "---------------------------------------------------------------------------------"

# test standalone C++ version
echo "Testing standalone C++ version..."
echo "----- C++ -----------------------------------------------------------------------"
./build/net_forward
echo "---------------------------------------------------------------------------------"
echo "Standalone version test completed."

# # test MATLAB version
echo "----- Matlab --------------------------------------------------------------------"
# start MATLAB -> run the script forward_test.m -> exit
matlab -nodisplay -nosplash -nodesktop -r "run('forward_test.m'); exit;"
echo "---------------------------------------------------------------------------------"
echo "MATLAB version test completed."


