import numpy as np
import os
import shutil
import pandas as pd
import subprocess
import psutil
import platform
import datetime
from memory_profiler import profile, memory_usage
from pymicrovecdb import MVDB, DataType, IndexType
import time
import struct
import pyJoules
from pyJoules.energy_meter import measure_energy


# System Information Functions
def get_cpu_info():
    # Fetch CPU information using subprocess and lscpu command
    cpu_info = subprocess.check_output("lscpu", shell=True).decode('utf-8')
    cpu_info_dict = {}
    for line in cpu_info.split('\n'):
        if "Model name:" in line:
            cpu_info_dict['CPU Model Name'] = line.split(':')[1].strip()
        if "Architecture:" in line:
            cpu_info_dict['CPU Architecture'] = line.split(':')[1].strip()
        if "CPU(s):" in line:
            cpu_info_dict['CPU Core Count'] = line.split(':')[1].strip()
        if "CPU MHz:" in line:
            cpu_info_dict['CPU Clock Speed (MHz)'] = line.split(':')[1].strip()
    return cpu_info_dict

def get_system_uptime():
    # Calculate system uptime
    uptime_seconds = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    return str(uptime_seconds)

def get_system_load():
    # Get system load average
    try:
        load_avg = subprocess.check_output("uptime", shell=True).decode('utf-8').split("load average: ")[1]
    except Exception as e:
        load_avg = f"Could not retrieve load average: {e}"
    return load_avg

def delete_directory(path):
    # Check if the directory exists and delete it
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Directory '{path}' deleted.")
    else:
        print(f"Directory '{path}' does not exist.")

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

# @measure_energy
def query(db_, query, ground_truth):
    start_time = time.time()
    results = db_.topk(1, query, len(ground_truth))
    result_ids, results_distances = results
    print(result_ids)
    print(results_distances)
    query_time = time.time() - start_time

    # Calculate recall
    correct_answers = 0
    for id in result_ids:
        if id in ground_truth:
            correct_answers += 1

    recall = correct_answers / len(ground_truth)
    print(recall)

# @profile
def main():
    # Initialize database
    db_ = MVDB(DataType.FLOAT)

    # delete_directory("./test_faiss_flat_db")
    # db_.create(IndexType.FAISS_FLAT, 128, "./test_faiss_flat_db", "../SPTAG/datasets/sift/sift_base.fvecs")
    db_.open("./test_faiss_flat_db")

    # Load query data
    # data = read_vector_file("../SPTAG/datasets/sift/sift_base.fvecs")
    queries = read_vector_file("../SPTAG/datasets/sift/sift_query.fvecs")
    ground_truth = read_vector_file("../SPTAG/datasets/sift/sift_groundtruth.ivecs")

    query(db_, queries[0], ground_truth[0])
    #
    # mem_usage = memory_usage((query, (db_, queries[0], ground_truth[0])), max_usage=True)
    # print('Memory usage (in MB):', mem_usage)
    # start_time = time.time()
    # results = db_.topk(1, queries[0], 100)
    # result_ids, results_distances = results
    # print(result_ids)
    # print(results_distances)
    # query_time = time.time() - start_time
    #
    # # Calculate recall
    # correct_answers = 0
    # for id in result_ids:
    #     if id in ground_truth[0]:
    #         correct_answers += 1
    #
    # recall = correct_answers / len(ground_truth[0])
    # print(recall)
    # print(queries[0])
    # print(ground_truth[0])
    # print(ground_truth[0][0])
    # print(data[ground_truth[0][0]])
    # print(data[ground_truth[0][1]])

    # print(len(queries[0]))
    # print(len(ground_truth))
    # print(len(ground_truth[0]))

    # print(l2_distance(queries[0], data[ground_truth[0][0]]))
    # print(l2_distance(queries[0], data[ground_truth[0][1]]))

    # # System information before starting queries
    # cpu_info = get_cpu_info()
    # uptime = get_system_uptime()
    # load_average = get_system_load()

    # Perform queries and record latencies

    # # System information after queries
    # post_query_load_average = get_system_load()
    #
    # # Output results
    # print(f"CPU Info: {cpu_info}")
    # print(f"System Uptime: {uptime}")
    # print(f"Pre-Query System Load Average: {load_average}")
    # print(f"Post-Query System Load Average: {post_query_load_average}")
    # print(f"Query Time: {query_time} seconds")
    # print(f"Recall: {recall * 100}%")

if __name__ == '__main__':
    main()
