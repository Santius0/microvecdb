#!/bin/bash

#Sysstat Utilities Overview:
#    mpstat: This tool reports individual or combined processor statistics.
#    iostat: It provides CPU and input/output statistics for devices and partitions.
#    pidstat: This is useful for monitoring individual tasks managed by the Linux kernel in real-time.
#    sar: It collects, reports, and saves system activity information (this can be particularly useful for historical data analysis).

# Check for NVIDIA device
is_nvidia_device() {
    if lspci | grep -i nvidia > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Ensure tegrastats is available for NVIDIA devices
check_tegrastats() {
    if ! command -v tegrastats &> /dev/null; then
        echo "Tegrastats is not found. Please ensure it is installed. Tegrastats typically comes pre-installed on NVIDIA Jetson devices."
        exit 1
    fi
}

# Install sysstat and powertop for non-NVIDIA devices
install_sysstat_powertop() {
    if ! command -v mpstat &> /dev/null || ! command -v powertop &> /dev/null; then
        echo "Sysstat or Powertop is not installed. Attempting to install both..."
        sudo apt-get update
        sudo apt-get install -y sysstat powertop
    else
        echo "Sysstat and Powertop are already installed."
    fi
}

# Function to check and install memory_profiler if needed
ensure_memory_profiler_installed() {
    if ! python3 -c "import memory_profiler" &> /dev/null; then
        echo "memory_profiler is not installed. Attempting to install..."
        pip3 install memory_profiler || { echo "Failed to install memory_profiler. Please install it manually." ; exit 1; }
    else
        echo "memory_profiler is already installed."
    fi
}

# Initialize flags
USE_MEMORY_PROFILER=false

# Process command-line options
while getopts 'p' flag; do
  case "${flag}" in
    p) USE_MEMORY_PROFILER=true ;;
    *) echo "Usage: $0 [-p] <command_to_run>"
       exit 1 ;;
  esac
done

# Remove the processed option from the parameters
shift $((OPTIND-1))

# Check if a command to run is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [-p] <command_to_run>"
    exit 1
fi

COMMAND="$1"

# Check device type and ensure required utilities are installed
if is_nvidia_device; then
    echo "NVIDIA device detected."
    check_tegrastats
    MONITOR_COMMAND="sudo tegrastats --interval 1000 > performance.log &"
else
    echo "Non-NVIDIA device detected."
    install_sysstat_powertop
    # Example monitoring command for non-NVIDIA devices; adjust as necessary.
    MONITOR_COMMAND="mpstat 1 > performance.log &"
fi

# Check and possibly install memory_profiler, if -p is provided
if [ "$USE_MEMORY_PROFILER" = true ]; then
    ensure_memory_profiler_installed
    COMMAND="python3 -m memory_profiler $COMMAND > profile.log"
fi

# Start performance monitoring
eval "$MONITOR_COMMAND"
MONITOR_PID=$!

echo "Running your program with performance logging in the background..."
echo "Monitor PID: $MONITOR_PID"

# Execute the provided command
eval $COMMAND

# Wait a bit for the last measurements to be logged
sleep 2

# Stop the performance monitoring
echo "Stopping performance monitoring..."
sudo kill -9 $MONITOR_PID

echo "Performance log saved to performance.log"
