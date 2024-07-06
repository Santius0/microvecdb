import os
import shutil
import numpy as np
import pickle
from PIL import Image
import io
from memory_profiler import profile

from pymicrovecdb import mvdb, utils


SIFT1M_BASE = "../data/sift1M/sift/sift_base.fvecs"
SIFT1M_QUERY = "../data/sift1M/sift/sift_query.fvecs"
SIFT1M_TRUTH = "../data/sift1M/sift/sift_groundtruth.ivecs"

def test_create_function(db_):
    initial_data = np.random.rand(100, 64).astype(np.float32)  # 100 random vectors of dimension 64
    vecs = utils.read_vector_file(SIFT1M_BASE)

    # FAISS_FLAT
    # db_.create(mvdb.IndexType.FAISS_FLAT, 64, "./indexes/faissflat_64_rand_direct_index", initial_data=initial_data)         # direct numpy array input
    # db_.create(mvdb.IndexType.FAISS_FLAT, 128, "./indexes/faissflat_128_sift1m_file_index", initial_data_path=SIFT1M_BASE)   # .xvec file input
    # db_.create(mvdb.IndexType.FAISS_FLAT, 128, "./indexes/faissflat_128_sift1m_direct_index", initial_data=vecs)             # loading from xvecs and then doing direct numpy array input

    # ANNOY
    # db_.create(mvdb.IndexType.ANNOY, 64, "./indexes/annoy_64_rand_direct_index", initial_data=initial_data, n_trees=10, n_threads = -1)
    # db_.create(mvdb.IndexType.ANNOY, 128, "./indexes/annoy_128_sift1m_file_index", initial_data_path=SIFT1M_BASE, n_trees=40, n_threads = -1)
    # db_.create(mvdb.IndexType.ANNOY, 128, "./indexes/annoy_128_sift1m_direct_index", initial_data=vecs, n_trees=25, n_threads = 6)

    # SPANN
    # db_.create(mvdb.IndexType.SPANN, 64, "./indexes/spann_64_rand_direct_index", initial_data=initial_data, build_config_path="/home/santius/microvecdb/SPTAG/buildconfig.ini")
    db_.create(mvdb.IndexType.SPANN, 128, "./indexes/spann_128_sift1m_file_index", initial_data_path=SIFT1M_BASE, build_config_path="/home/santius/microvecdb/SPTAG/buildconfig.ini")
    # db_.create(mvdb.IndexType.SPANN, 128, "./indexes/spann_128_sift1m_direct_index", initial_data=vecs, build_config_path="/home/santius/microvecdb/SPTAG/buildconfig.ini")

def test_topk_function(db_):
    rand_queries = np.random.rand(10, 64).astype(np.float32)  # 10 random vectors of dimension 64
    queries = utils.read_vector_file(SIFT1M_QUERY)

    # FAISS_FLAT
    # db_.open("./indexes/faissflat_64_rand_direct_index")
    # db_.topk(query=rand_queries, k = 10)
    # db_.topk(query=queries, k = 10)
    # db_.topk(query_file=SIFT1M_QUERY, k = 10)

    # SPANN
    db_.open("./indexes/spann_128_sift1m_file_index")
    # print(db_.dims)
    # res = db_.topk(query=rand_queries, k = 10)
    # print(res)
    res = db_.topk(query=queries, k = 100)
    print(res)
    # res = db_.topk(query_file=SIFT1M_QUERY, k = 100)
    # print(res)
    print(res[0].shape)
    print(res[1].shape)
    # print(res)


BASE_DATA_DIR = '/home/santius/ann_data'
BASE_INDEX_DIR = '/home/santius/ann_indices'

SIFT10K = f'{BASE_DATA_DIR}/sift10K_128D_float32/sift10K_128D_float32_base.fvecs'
queries = utils.read_vector_file(f"{BASE_DATA_DIR}/deep/deep1B_queries.fvecs")

# @profile
def main():
    # mvdb.process_image('./Screenshot 2024-05-24 074314.png')

    if os.path.exists("./test_annoy_idx"):
        shutil.rmtree("./test_annoy_idx")

    image_path = './Screenshot 2024-05-24 074314.png'
    audio_path = './file_example_WAV_10MG.wav'
    # text_path = 'path/to/text.txt'

    # Read binary data from the files
    image_data = mvdb.read_binary_file(image_path)
    audio_data = mvdb.read_binary_file(audio_path)
    # print(audio_data)
    # with open(text_path, 'r', encoding='utf-8') as f:
    #     text_data = f.read().encode('utf-8')

    # Create a list of mixed data types
    data_list = [
        # "asdasda",
        # "asassssssssssssssssssssss",
        {'type': 'image', 'data': image_data},
        {'type': 'audio', 'data': audio_data},
        None
        # {'type': 'text', 'data': text_data}
        # b'00000000',
        # b'10000000',
        # b'11000000',
        # image_data
    ]

    serialized_data_list = [pickle.dumps(data) for data in data_list]
    serialized_data = np.array(serialized_data_list, dtype=object)

    annoy_db = mvdb.MVDB(dtype=mvdb.DataType.INT8)
    annoy_db.create(
        index_type=mvdb.IndexType.ANNOY,
        dims=5,
        path="./test_annoy_idx",
        initial_data=np.array([
            [1, 2, 3, 4, 5],
            [0, 44, 55, 3, 1],
            [0, 44, 55, 3, 1],
        ], dtype=np.int8),
        initial_objs=serialized_data,
        n_trees=10,
        n_threads=12
    )

    keys = np.array([0, 1, 2])
    a = annoy_db.get(keys)

    for i in range(len(keys)):
        try:
            thing = pickle.loads(a[i])
            if thing is not None:
                print(thing['type'])
            else:
                print(f'{i} is None')
        except Exception as e:
            print(f"Error unpickling data for key {keys[i]}: {e}")

    # print(a)
    # image_stream = io.BytesIO(a[2])
    # image = Image.open(image_stream)
    # image.save("./test_image.png")


    # if a is not None:
    #     print("Data successfully retrieved from RocksDB.")
    #     for i in range(len(keys)):
    #         print(pickle.loads(a[i]))
    # else:
    #     print("Failed to retrieve data from RocksDB.")

#     print(annoy_db.num_items)
#     print(annoy_db.dims)
#     # print(queries)
#     # print(len(queries))
#     # print(len(queries[33]))
#     # db1 = mvdb.MVDB()
#     # db1.open("./indices/spann_sift10K_float32")
#     # print(f'python num items {db1.num_items}')
#
#     # db2 = mvdb.MVDB()
#     # db2.open("./indices/spann_sift1M_float32")
#     # print(f'python num items {db2.num_items}')
#     res = annoy_db.topk(query=queries.astype(np.int8), k=100)
#     print(res[0])
#     # print(utils.read_vector_file("../../ann_data/deep1M/deep1M_groundtruth.ivecs"))
# # test_create_function(db)
#     # test_topk_function(db)

if __name__ == '__main__':
    main()
