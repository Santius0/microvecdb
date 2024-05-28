import numpy as np
import pandas as pd
import subprocess
import psutil
import platform
import datetime
import struct
import mmap

def is_wsl():
    return 'microsoft' in platform.uname().release.lower()
def get_cpu_info():
    cpu_info = subprocess.check_output("lscpu", shell=True).decode('utf-8')
    cpu_info_dict = {}
    for line in cpu_info.split('\n'):
        if "Model name:" in line:
            cpu_info_dict['cpu_model_name'] = line.split(':')[1].strip()
        if "Architecture:" in line:
            cpu_info_dict['cpu_architecture'] = line.split(':')[1].strip()
        if "CPU(s):" in line:
            cpu_info_dict['cpu_core_count'] = line.split(':')[1].strip()
        if "CPU MHz:" in line:
            cpu_info_dict['cpu_clock_speed_(MHz)'] = line.split(':')[1].strip()
    return cpu_info_dict

def get_ram_info():
    ram_info = {}
    svmem = psutil.virtual_memory()
    ram_info['total_DRAM_(GB)'] = svmem.total / (1024 ** 3)

    # Using dmidecode to get RAM speed and type
    try:
        # Requires root access
        ram_details = subprocess.check_output("sudo dmidecode -t memory", shell=True).decode('utf-8')
        for item in ram_details.split("Memory Device\n"):
            if "Speed:" in item and "Type:" in item:
                ram_info['RAM_speed_(MHz)'] = item.split("Speed:")[1].split("MHz")[0].strip()
                ram_info['RAM_type'] = item.split("Type:")[1].split("\n")[0].strip()
                break  # Assuming all RAM sticks have the same type and speed
    except Exception as e:
        print(f"Failed to get detailed RAM info: {e}")

    return ram_info

def get_storage_info():
    storage_info = {}
    disk_io = psutil.disk_io_counters()
    storage_info['storage_read_count'] = disk_io.read_count
    storage_info['storage_write_count'] = disk_io.write_count
    storage_info['storage_read_bytes'] = disk_io.read_bytes
    storage_info['storage_write_bytes'] = disk_io.write_bytes
    result = subprocess.check_output("lsblk -o NAME,TYPE,SIZE", shell=True).decode('utf-8')
    lines = result.strip().split('\n')[1:]  # skip header
    data = [line.split() for line in lines]

    result = subprocess.check_output("lsblk -o NAME,TYPE,SIZE", shell=True).decode('utf-8')
    lines = result.strip().split('\n')[1:]  # skip header
    data = [line.split() for line in lines]
    if data:
        df = pd.DataFrame(data, columns=['Name', 'Type', 'Size'])
        storage_info['storage_devices'] = df.to_dict('records')
    # Use iostat for I/O statistics if available (install sysstat package)
    try:
        io_stats = subprocess.check_output("iostat -dx", shell=True).decode('utf-8')
        lines = io_stats.strip().split('\n')[3:-2]
        data = [line.split() for line in lines]
        columns = ['Device', 'r/s', 'rkB/s', 'rrqm/s', '%rrqm', 'r_await', 'rareq-sz', 'w/s', 'wkB/s', 'wrqm/s', '%wrqm', 'w_await', 'wareq-sz', 'd/s', 'dkB/s', 'drqm/s', '%drqm', 'd_await', 'dareq-sz', 'aqu-sz', '%util']
        df = pd.DataFrame(data, columns=columns)
        storage_info['IO Stats'] = df.to_dict('records')
    except Exception as e:
        print(f"Failed to get I/O stats: {e}")
    return storage_info

def get_battery_info():
    battery_info = {}
    if hasattr(psutil, 'sensors_battery'):
        battery = psutil.sensors_battery()
        if battery:
            battery_info['battery_capacity (%)'] = battery.percent
            battery_info['power_plugged'] = battery.power_plugged
    return battery_info

def get_system_uptime():
    # Calculate system uptime
    uptime_seconds = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    return str(uptime_seconds)

def get_concurrent_processes():
    # List all current processes
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        processes.append(proc.info)
    df = pd.DataFrame(processes)
    return df.to_dict("records")

def get_system_load():
    if platform.system() == 'Linux' or 'microsoft' in platform.uname().release.lower():
        # Use uptime command to get system load averages
        try:
            load_avg = subprocess.check_output("uptime", shell=True).decode('utf-8').split("load average: ")[1]
        except IndexError:
            load_avg = "Could not retrieve load average"
    elif platform.system() == 'Windows':
        # Windows does not provide load average directly; simulate it using CPU percent
        load_avg = f"CPU Usage Percent: {psutil.cpu_percent(interval=1)}"
    else:
        load_avg = "Unsupported platform"
    return load_avg


def read_vector_file(filename, n=-1):
    # Determine the type based on file extension
    file_type = 'f' if filename.endswith('.fvecs') else 'i'
    dtype = np.float32 if file_type == 'f' else np.int32
    element_size = np.dtype(dtype).itemsize  # 4 bytes for both float32 and int32

    vectors = []

    with open(filename, 'rb') as file:
        mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        offset = 0
        i = 0
        while offset < mmapped_file.size() and (n < 0 or i < n):
            # Read the dimensionality of the vector (first 4 bytes)
            dim_bytes = mmapped_file[offset:offset + 4]
            if not dim_bytes:
                break
            dimension = struct.unpack('i', dim_bytes)[0]
            offset += 4

            # Read the vector data
            vector_bytes = mmapped_file[offset:offset + element_size * dimension]
            offset += element_size * dimension

            if file_type == 'f':
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
            else:
                vector = np.frombuffer(vector_bytes, dtype=np.int32)

            vectors.append(vector)
            i += 1

        mmapped_file.close()

    return np.array(vectors)


def write_vector_file(arr, filename):
    """
    Write a NumPy array to a file in fvecs or ivecs format.

    Parameters:
    arr (np.array): Input array of shape (n_vectors, vector_dim)
    filename (str): Output filename, should end with .fvecs or .ivecs
    """
    assert filename.endswith(('.fvecs', '.ivecs')), "Filename must end with .fvecs or .ivecs"

    dtype = np.float32 if filename.endswith('.fvecs') else np.int32
    arr = arr.astype(dtype)

    with open(filename, 'wb') as f:
        for vec in arr:
            length = np.int32(len(vec)).tobytes()
            f.write(length)
            f.write(vec.tobytes())


def get_fvecs_dim_size(filename):
    """
    Get the dimension size of the first vector in an fvecs file.

    Parameters:
    filename (str): The path to the fvecs file.

    Returns:
    int: The dimension size of the vectors in the fvecs file.
    """
    with open(filename, 'rb') as f:
        # Read the first 4 bytes which contain the dimension of the first vector
        dim_size_bytes = f.read(4)
        # Convert the bytes to an integer
        dim_size = np.frombuffer(dim_size_bytes, dtype=np.int32)[0]

    return dim_size
