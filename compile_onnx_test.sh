#!/bin/bash

clear


if [ ! -d "$(pwd)/onnxruntime-linux-x64-1.22.0" ]; then
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
    tar -xzf onnxruntime-linux-x64-1.22.0.tgz
    rm onnxruntime-linux-x64-1.22.0.tgz
    echo "onnxruntime downloaded and extracted."
fi

export ONNX_DIR="$(pwd)/onnxruntime-linux-x64-1.22.0"

echo "onnxruntime directory: $ONNX_DIR"

echo "compiling onnx_test.cpp..."

g++ onnx_test.cpp -o onnx_test \
  -I$ONNX_DIR/include \
  -L$ONNX_DIR/lib \
  -Wl,-rpath=$ONNX_DIR/lib \
  -lonnxruntime \
  -std=c++17

# create the .net file with python
echo "Creating the .net file with python..."
echo "----- Python --------------------------------------------------------------------"
python create_net.py
echo "---------------------------------------------------------------------------------"

# test standalone C++ version
echo "Testing standalone C++ version..."
echo "----- C++ -----------------------------------------------------------------------"
./onnx_test
echo "---------------------------------------------------------------------------------"
echo "Standalone version test completed."

