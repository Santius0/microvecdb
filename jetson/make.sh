#!/bin/bash

# Define the build directory
BUILD_DIR=build

# Function to check if the build directory exists and remove it if requested
clean_build() {
    if [ -d "$BUILD_DIR" ]; then
        echo "Removing existing build directory..."
        rm -rf $BUILD_DIR
    fi
}

# Check for command line options
if [ "$1" == "--clean" ]; then
    clean_build
fi

# Create the build directory if it doesn't exist, and navigate into it
if [ ! -d "$BUILD_DIR" ]; then
    mkdir $BUILD_DIR
    echo "Created build directory."
fi
# shellcheck disable=SC2164
cd $BUILD_DIR

# Run CMake and Make
cmake ..
make

# Return to the original directory
# shellcheck disable=SC2103
cd ..

echo "Build process completed."
