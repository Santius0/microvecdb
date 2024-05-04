#!/bin/bash

# Define variables
JETSON_IP="192.168.55.1"  # IP address of the Jetson Nano
USERNAME="santius"        # Username on the Jetson Nano
DEST_DIR="/home/santius/microvecdb/jetson/jetson-nano-sysroot"  # Destination directory on the local machine
PASSWORD_FILE="/home/santius/microvecdb/jetson/password.txt"  # Path to the password file

# Check if the password file exists
if [ ! -f "$PASSWORD_FILE" ]; then
    echo "Password file does not exist."
    exit 1
fi

# Read password from file
PASSWORD=$(cat "$PASSWORD_FILE")

# Create the destination directory if it does not already exist
mkdir -p "$DEST_DIR"

# rsync options
RSYNC_OPTS="-avzHAX"

# List of directories to sync
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

# Loop through directories and sync each one
for dir in "${directories[@]}"; do
    sshpass -p "$PASSWORD" rsync $RSYNC_OPTS -e "ssh -o StrictHostKeyChecking=no" "$USERNAME@$JETSON_IP:$dir" "$DEST_DIR$(dirname $dir)"
done

echo "Synchronization complete."
