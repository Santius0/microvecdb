import os
import numpy as np
from progress.bar import Bar
from pymicrovecdb import mvdb, utils as mv_utils

dataset_configs = {
    'sift': {'base_path': '/home/santius/microvecdb/data/sift1M/sift/sift_base.fvecs', 'query_path': '/home/santius/microvecdb/data/sift1M/sift/sift_query.fvecs' , 'sizes': [10000, 100000, 500000]},
    'gist': {'base_path': '/home/santius/microvecdb/data/gist1M/gist/gist_base.fvecs', 'query_path': '/home/santius/microvecdb/data/gist1M/gist/gist_query.fvecs', 'sizes': [10000, 100000, 500000]},
    'deep': {'base_path': '/home/santius/microvecdb/data/deep10M/deep10M.fvecs', 'query_path': '/home/santius/microvecdb/data/deep10M/deep1B_queries.fvecs', 'sizes': [10000, 100000, 500000, 1000000, 5000000]},
}
def short_code(num):
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        short_num = num / 1_000
        suffix = "K"
    elif num < 1_000_000_000:
        short_num = num / 1_000_000
        suffix = "M"
    elif num < 1_000_000_000_000:
        short_num = num / 1_000_000_000
        suffix = "B"
    else:
        short_num = num / 1_000_000_000_000
        suffix = "T"

    if short_num == int(short_num):
        short_num = int(short_num)
    else:
        short_num = f"{short_num:.1f}".rstrip('0').rstrip('.')

    return f"{short_num}{suffix}"

def build_dataset(base_path, n, output_name):
    vecs = mv_utils.read_vector_file(base_path, n)
    mv_utils.write_vector_file(vecs, f'{output_name}_base.fvecs')

def build_all():
    bar = Bar("building datasets", max=1000)
    for conf_key in dataset_configs:
        bar.next()
        dataset_config = dataset_configs[conf_key]
        for size in dataset_config['sizes']:
            bar.next()
            name = f'{conf_key}{short_code(size)}'
            path = f'/home/santius/microvecdb/data/{name}'
            os.makedirs(path, exist_ok=True)
            build_dataset(dataset_config['base_path'], size, f'{path}/{name}')
            db = mvdb.MVDB()
            db.create(
                index_type=mvdb.IndexType.FAISS_FLAT,
                dims=mv_utils.get_fvecs_dim_size(f'{path}/{name}_base.fvecs'),
                path=f'./indices/{name}',
                initial_data_path=dataset_config['base_path']
            )
            bar.next()
            ids, dists, _ = db.topk(query=mv_utils.read_vector_file(dataset_config['query_path']), k=100)
            mv_utils.write_vector_file(ids, f'{path}/{name}_groundtruth.ivecs')
            mv_utils.write_vector_file(dists, f'{path}/{name}_groundtruth_dists.fvecs')
            bar.next()
    bar.finish()

def main():
    build_all()

if __name__ == '__main__':
    main()