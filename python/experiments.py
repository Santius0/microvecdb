import numpy as np
import pandas as pd
import subprocess
import psutil
import platform
import datetime
import struct
import time
from memory_profiler import profile, memory_usage
from progress.bar import Bar
from pymicrovecdb import MVDB, DATA_TYPE, IndexType

def is_wsl():
    # Check if the script is running in the Windows Subsystem for Linux
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


def read_vector_file(filename, dtype = np.float32):
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


def l2_distance(a, b):
    return np.linalg.norm(a - b)

def top_k_measurement_wrapper(db_, q):
    results = db_.topk(1, q, len(q))
    return results

def run_single_query(db_, query, ground_truth):
    start_time = time.time()
    topk_res = memory_usage((top_k_measurement_wrapper, (db_, query)), retval=True, max_usage=True)
    query_time = time.time() - start_time
    found = 0
    for id in topk_res[1][0]:
        if id in ground_truth:
            found += 1
    recall = found / len(ground_truth)
    row_dict = cpu_env
    row_dict['peak_dram_(MB)'] = topk_res[0]
    row_dict['index'] = IndexType.FAISS_FLAT.value
    row_dict['latency_(s)'] = query_time
    row_dict['recall@100'] = recall
    return row_dict

# Gather information
cpu_info = get_cpu_info()
ram_info = get_ram_info()
storage_info = get_storage_info()
battery_info = get_battery_info()
# Gather information
uptime = get_system_uptime()
processes_df = get_concurrent_processes()
load_average = get_system_load()

# Print the system information
print(f"System Uptime: {uptime}")
print(f"System Load Average: {load_average}")
print("Processes running concurrently:")
print(processes_df.head())  # Print first few processes for brevity

# Create a DataFrame
cpu_env = {**cpu_info, **ram_info, **storage_info, **battery_info}
# cpu_end_df = pd.DataFrame([cpu_env])

db_ = MVDB(DATA_TYPE.FLOAT)

# delete_directory("./test_faiss_flat_db")
# db_.create(IndexType.FAISS_FLAT, 128, "./test_faiss_flat_db", "../SPTAG/datasets/sift/sift_base.fvecs")
db_.open("./test_faiss_flat_db")

queries = read_vector_file("../SPTAG/datasets/sift/sift_query.fvecs")
ground_truth = read_vector_file("../SPTAG/datasets/sift/sift_groundtruth.ivecs")


rows = []

bar = Bar(f"{IndexType.FAISS_FLAT}_{cpu_info['cpu_architecture']}: ", max=len(queries))
for i, q in enumerate(queries):
    rows.append(run_single_query(db_, q, ground_truth[i]))
    bar.next()
bar.finsh()

# start_time = time.time()
    # ids, distances = db_.topk(1, q, len(q))
    # query_time = time.time() - start_time
    #
    # found = 0
    # for id in ids:
    #     if id in ground_truth[i]:
    #         found += 1
    # recall = found / len(ground_truth[i])
    # row_dict = cpu_env
    # row_dict['index'] = IndexType.FAISS_FLAT.value
    # row_dict['latency'] = query_time
    # row_dict['recall'] = recall
    # rows.append(row_dict)


df = pd.DataFrame(rows)
df.to_csv('faissflat_x86_64_12c_8gb.csv')
print(df)
