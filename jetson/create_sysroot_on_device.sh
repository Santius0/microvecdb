#!/bin/bash

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

# Destination directory on the flash drive
destination="/media/67C9-117A/jetson-nano-sysroot"

mkdir -p "$destination"

# Loop through each directory and copy it using rsync
for dir in "${directories[@]}"; do
  echo "Copying $dir to $destination$(dirname $dir)"
  mkdir -p "$destination$(dirname $dir)"
  rsync -avzHAX --exclude='*.lock' --exclude='/var/run/' "$dir" "$destination$(dirname $dir)"
done

echo "All directories have been copied."
