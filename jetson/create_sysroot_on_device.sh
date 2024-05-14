#!/bin/bash

# Define variables
DEST_DIR="/media/67C9-117A/jetson-nano-sysroot"  # Destination directory on the local machine

# Create the destination directory if it does not already exist
mkdir -p "$DEST_DIR"

# List of directories to copy
declare -a directories=(
    "/lib"
    "/usr/lib"
    "/usr/include"
    "/bin"
    "/sbin"
    "/etc"
    "/opt"
    "/usr/local"
    "/usr/bin"
    "/usr/sbin"
)

# Loop through directories and copy each one
for dir in "${directories[@]}"; do
    # Check if the source directory exists
    if [ ! -d "$dir" ]; then
        echo "Directory $dir does not exist."
        continue
    fi
    # Define destination path for current directory
    dest_path="$DEST_DIR$(dirname $dir)"
    # Create destination directory if it does not exist
    mkdir -p "$dest_path"
    # Use cp command to copy directory contents
    cp -a "$dir/." "$dest_path/"
done

echo "Copying complete."
