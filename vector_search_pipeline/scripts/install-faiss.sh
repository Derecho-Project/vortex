#!/bin/bash

# This line should be set in env variables
export PATH=/usr/local/cuda/bin:$PATH

git clone https://github.com/facebookresearch/faiss.git
cd faiss
git checkout v1.9.0
build_path="build"
if [ -z $FAISS_INSTALL_PREFIX ]; then
    install_prefix="/usr/local"
else
    install_prefix=$FAISS_INSTALL_PREFIX
fi
echo "FAISS library will be installed in $install_prefix"

if [ -z $CUDAToolKitRoot ]; then
    cudatoolkit_dir="/usr/local/cuda-12.3"
else
    cudatoolkit_dir=$CUDAToolKitRoot
fi

# Python interface is currently not needed for vortex, but it is needed to buildthe GIST dataset with clusters. So need to install it on the server that handle the data processing
read -p "Install FAISS Python interface? (y/n): " response
response=${response,,}

if [[ "$response" == "y" ]]; then
    enable_python="ON"
elif [[ "$response" == "n" ]]; then
    enable_python="OFF"
else
    echo "Invalid response. Please enter 'y' or 'n'."
fi

# CUDA_ARCHITECTURES: specify the target CUDA architectures. 
# Tesla T4 on fractus is 75
# (When changing GPU machine, also check to see if this needs to be changed)
cuda_archs="75" 

# More about the flags setting checkout https://github.com/facebookresearch/faiss/blob/main/INSTALL.md 
cmake_defs="-DCMAKE_BUILD_TYPE=${build_type} \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=${install_prefix} \
          -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=${enable_python} -DFAISS_ENABLE_RAFT=OFF \
          -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_C_API=ON \
          -DCUDAToolkit_ROOT=${cudatoolkit_dir} -DCMAKE_CUDA_ARCHITECTURES=${cuda_archs}"

rm -rf ${install_prefix}/include/faiss ${install_prefix}/lib/libfaiss* ${install_prefix}/lib/cmake/faiss
rm -rf ${build_path} 2>/dev/null
mkdir ${build_path}
cd ${build_path}
cmake ${cmake_defs} ..
NPROC=`nproc`
if [ $NPROC -lt 2 ]; then
    NPROC=2
fi
make faiss -j `expr $NPROC - 1` 2>err.log
make install

if [[ "$response" == "y" ]]; then
    echo "Installing FAISS Python interface..."
    make -j swigfaiss
    cd faiss/python
    python setup.py install --user
    cd ../../..
fi

echo "FAISS installed successfully."
# clean up
rm -rf faiss
