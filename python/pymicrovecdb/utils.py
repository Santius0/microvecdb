import numpy as np
import pandas as pd
import subprocess
import psutil
import platform
import datetime
import struct

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
    print("linux storage")
    result = subprocess.check_output("lsblk -o NAME,TYPE,SIZE", shell=True).decode('utf-8')
    lines = result.strip().split('\n')[1:]  # skip header
    data = [line.split() for line in lines]

    print("linux storage")
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
        df = pd.DataFrame(data, columns=['Device', 'tps', 'kB_read/s', 'kB_wrtn/s', 'kB_read', 'kB_wrtn'])
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
    return df

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


def read_vector_file(filename):
    """
    Read vectors from a file with .fvecs or .ivecs extension.

    Parameters:
    - filename (str): Path to the file to read.

    Returns:
    - numpy.ndarray: Array of vectors stored in the file.
    """
    # Determine the type based on file extension
    file_type = 'f' if filename.endswith('.fvecs') else 'i'
    dtype = np.float32 if file_type == 'f' else np.int32

    vectors = []
    with open(filename, 'rb') as file:
        while True:
            # Read the dimensionality of the vector (first 4 bytes)
            dim_bytes = file.read(4)
            if not dim_bytes:
                break
            dimension = struct.unpack('i', dim_bytes)[0]

            # Read the vector data
            count = dimension
            if file_type == 'f':
                vector_bytes = file.read(4 * count)  # Each float takes 4 bytes
                vector = struct.unpack(f'{count}f', vector_bytes)
            else:
                vector_bytes = file.read(4 * count)  # Each int takes 4 bytes
                vector = struct.unpack(f'{count}i', vector_bytes)

            vectors.append(vector)

    return np.array(vectors, dtype=dtype)

def save_ivecs(filename, data):
    """
    Save an array of integers in the .ivecs format.

    Parameters:
    - filename: The name of the file to save.
    - data: A numpy array of shape (n, d) containing the integer vectors.
    """
    with open(filename, 'wb') as f:
        for vec in data:
            # Write the dimension (d) as a 4-byte integer
            f.write(np.int32(len(vec)).tobytes())
            # Write the vector data as 4-byte integers
            f.write(np.int32(vec).tobytes())

def generate_groundtruth(query_set, topk_function, k=100, save_path='groundtruth.ivecs'):
    """
    Generate and save ground truth vectors for a given query set in the .ivecs format.

    Parameters:
    - query_set: A numpy array of shape (nq, d) containing the query vectors.
    - topk_function: A function that takes a query set and returns a tuple containing:
                     - an np.array of shape (nq * k,) with the IDs of the top k vectors.
                     - an np.array of shape (nq * k,) with the corresponding distances.
    - k: The number of top results to return for each query vector.
    - save_path: The file path to save the ground truth vectors.
    """

    query_set = np.array(query_set, dtype=np.float32)
    ids, distances = topk_function(query_set, k)
    ids = np.array(ids, dtype=np.int32)
    distances = np.array(distances, dtype=np.float32)

    nq = query_set.shape[0]
    ids = ids.reshape(nq, k)

    save_ivecs(save_path, ids)
    print(f"Ground truth vectors saved to {save_path}")