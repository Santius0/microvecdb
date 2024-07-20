import os
import random
import shutil

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
from pymicrovecdb import mvdb, utils as mv_utils
import pickle
import time
import io
import csv

# Initialize face detection and recognition models
mtcnn = MTCNN(image_size=160, margin=0)
model = InceptionResnetV1(pretrained='vggface2').eval()

DB_PATH = './facesDB'
FACE_DIR = './lfw/lfw-deepfunneled/lfw-deepfunneled'  # LFW dataset

spann_index_params = {
    'build_config_path': "./buildconfig.ini",
    'BKTKmeansK': 8,
    'Samples': 4000,
    'TPTNumber': 112,
    'RefineIterations': 2,
    'NeighborhoodSize': 144,
    'CEF': 1800,
    'MaxCheckForRefineGraph': 7168,
    'NumberOfInitialDynamicPivots': 30,
    'GraphNeighborhoodScale': 2,
    'NumberOfOtherDynamicPivots': 2,
    'batch_size': 2000,
    'thread_num': 10,
}

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    face = mtcnn(img)
    return face


def get_embedding(image_path):
    face = preprocess_image(image_path)
    if face is not None:
        face = face.unsqueeze(0)
        with torch.no_grad():
            vec = model(face).numpy()
        return vec
    else:
        return None


embeddings = []
binary_data = []
test_images = []

for idx, person in enumerate(os.listdir(FACE_DIR)):
    person_dir = os.path.join(FACE_DIR, person)
    if os.path.isdir(person_dir):
        images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
        if images:
            selected_image = random.choice(images)
            test_images.extend(img for img in images if img != selected_image)

            print(f"{idx} => Generating embedding for {selected_image}...")
            embedding = get_embedding(selected_image)
            if embedding is not None:
                embeddings.append(embedding)
                with open(selected_image, 'rb') as img_file:
                    binary_image = img_file.read()
                person_data = {
                    "name": person,
                    "image_path": selected_image,
                    "image": binary_image
                }
                binary_data.append(pickle.dumps(person_data))
            print(f"{idx} => {selected_image} embedding generated.")

embeddings = np.vstack(embeddings)
binary_data = np.array(binary_data, dtype=object)

face_db = mvdb.MVDB(dtype=mvdb.DataType.FLOAT32)

if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH):
    shutil.rmtree(DB_PATH)

face_db.create(
    index_type=mvdb.IndexType.SPANN,
    dims=512,
    path=DB_PATH,
    initial_data=embeddings,
    initial_objs=binary_data,
    **spann_index_params,
)

def validate_performance(test_images_):
    total_latency = 0
    total_recall = 0
    num_images = len(test_images_)
    result_file = "./results.csv"

    for image_path in test_images_:
        query_embedding = get_embedding(image_path)
        if query_embedding is None:
            continue

        query_embedding = query_embedding.astype(np.float32)

        start_time = time.time()
        res = face_db.knn(query=query_embedding, k=1, **spann_index_params)
        end_time = time.time()

        retrieval_time = end_time - start_time
        total_latency += retrieval_time

        ids = res[0][0]
        dists = res[1][0]

        recall_at_1 = 0
        retrieved_path = ""
        retrieved_name = ""

        retrieved_objs = face_db.get(ids)

        # Calculate recall@1
        for data in retrieved_objs[:1]:
            retrieved_data = pickle.loads(data)
            retrieved_path = retrieved_data["image_path"]
            retrieved_name = retrieved_data["name"]
            # image_stream = io.BytesIO(retrieved_data["image"])
            # image = Image.open(image_stream)
            # image.save(f'./retrieved/{retrieved_data["name"]}.png')
            if retrieved_data["name"] in image_path:
                recall_at_1 = 1
            total_recall += recall_at_1

        row = {
            "image": image_path,
            "latency": retrieval_time,
            "recall@1": recall_at_1,
            "ids": ids,
            "distances": dists,
            "retrieved_name": retrieved_name,
            "retrieved_path": retrieved_path,
        }

        with open(result_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not os.path.isfile(result_file):
                writer.writeheader()
            writer.writerow(row)

    avg_latency_ = total_latency / num_images
    avg_recall_ = total_recall / num_images

    return avg_latency_, avg_recall_


# Validate performance
avg_latency, avg_recall = validate_performance(test_images)
print(f'Avg Retrieval Time (s): {avg_latency}')
print(f'Avg Recall@1: {avg_recall}')
