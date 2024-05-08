#!/bin/bash

# Define the build directory
BUILD_DIR=build

# Function to check if the build directory exists and remove it if requested
clean_build() {
    if [ -d "$BUILD_DIR" ]; then
        echo "Removing existing build directory..."
        rm -rf "$BUILD_DIR"
    fi
}

# Check for command line options
case "$1" in
    -c|--clean)
        clean_build
        ;;
esac

# Create the build directory if it doesn't exist, and navigate into it
if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
    echo "Created build directory."
fi
cd "$BUILD_DIR" || exit

# Run CMake and Make
cmake ..
make

# Return to the original directory
cd ..

echo "Build process completed."
