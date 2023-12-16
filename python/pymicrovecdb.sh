#!/bin/bash

# Stop execution if any command fails
set -e

# Run python setup.py bdist_wheel
echo "Building the wheel package..."
python3 setup.py bdist_wheel

# Install the package
echo "Installing the package..."
pip install .

echo "Build and installation completed successfully!"
