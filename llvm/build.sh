#!/bin/bash

build() {
    SRC_PATH=$(pwd)/
    if [ ! -d './build' ]; then
        mkdir ./build
    fi
    cd ./build
    echo "Entering build directory: $(realpath $(pwd))"
    cmake -DLT_LLVM_INSTALL_DIR=/usr/lib/llvm-15 $SRC_PATH
    cmake --build .

    echo "Leaving build directory, back to $SRC_PATH"
    cd $SRC_PATH
}

clean() {
    if [ -d './build' ]; then
        echo "remove build directory"
        rm -r ./build
    fi
}

if [[ $# != 0 ]] && [[ "$1" == "clean" ]]; then
    clean
else
    build
fi
