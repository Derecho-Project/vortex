#!/bin/bash

build_type=$1
with_vector=$2

# Set build type (default to Release if not specified)
if [ -z "$build_type" ]; then
    build_type="Release"
fi

# Set build path
build_path="build-${build_type}"

# Set install prefix
if [ -z "$VORTEX_INSTALL_PREFIX" ]; then
    install_prefix="/usr/local"
else
    install_prefix=$VORTEX_INSTALL_PREFIX
fi

# Configure vector search pipeline option
if [ "$with_vector" == "vector" ] || [ "$with_vector" == "--with-vector" ]; then
    vector_flag="-DVORTEX_BUILD_VECTOR_SEARCH_PIPELINE=ON"
    echo "Building with vector search pipeline enabled"
else
    vector_flag="-DVORTEX_BUILD_VECTOR_SEARCH_PIPELINE=OFF"
    echo "Building without vector search pipeline"
fi

# CMake definitions
cmake_defs="-DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=${install_prefix} ${vector_flag}"

# Begin building...
echo "Build type: ${build_type}"
echo "Build path: ${build_path}"
echo "Install prefix: ${install_prefix}"

rm -rf ${build_path}
mkdir ${build_path}
cd ${build_path}

cmake ${cmake_defs} ..

NPROC=`nproc`
if [ $NPROC -lt 2 ]; then
    NPROC=2
fi

make -j `expr $NPROC - 1` 2>err.log

cd ..
