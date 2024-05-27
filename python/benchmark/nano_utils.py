import subprocess
import os
import platform
import signal

TEGRATSTATS_PID_FILE = "/tmp/tegrastats.pid"

def is_jetson_nano():
    if platform.system() != 'Linux':
        return False
    try:
        with open('/proc/device-tree/model', 'r') as model_file:
            model = model_file.read().strip()
            if 'Jetson Nano' in model:
                return True
    except FileNotFoundError:
        return False

    return False

def start_tegrastats(logfile_name):
    # Build the command with the specified interval and logfile
    command = ['tegrastats', '--interval', '100', '--logfile', logfile_name]
    # Run tegrastats in the background
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Store the PID of the process
    with open(TEGRATSTATS_PID_FILE, 'w') as pid_file:
        pid_file.write(str(process.pid))

    print(f"tegrastats started with PID {process.pid}, logging to {logfile_name}.")

def stop_tegrastats():
    try:
        # Read the PID from the file
        with open(TEGRATSTATS_PID_FILE, 'r') as pid_file:
            pid = int(pid_file.read().strip())

        # Terminate the process
        os.kill(pid, signal.SIGTERM)
        print(f"tegrastats with PID {pid} stopped.")

        # Remove the PID file
        os.remove(TEGRATSTATS_PID_FILE)
    except FileNotFoundError:
        print("No running tegrastats process found.")
    except ProcessLookupError:
        print("The tegrastats process was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
