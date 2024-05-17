#include <Python.h>
#include <mvdb.h>
#include <faiss_flat_index.h>
#include <annoy_index.h>
#include <spann_index.h>
#include <iostream>
#include <utils.h>


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define NAMED_ARGS "mvdb::NamedArgs"
#define FAISS_FLAT_INDEX_NAMED_ARGS "mvdb::index::FaissFLatIndexNamedArgs"
#define ANNOY_INDEX_NAMED_ARGS "mvdb::index::AnnoyIndexNamedArgs"
#define SPANN_INDEX_NAMED_ARGS "mvdb::index::SPANNIndexNamedArgs"

#define MVDB_NAME_int8_t "mvdb::MVDB<int8_t>"
#define MVDB_NAME_int16_t "mvdb::MVDB<int16_t>"
#define MVDB_NAME_uint8_t "mvdb::MVDB<uint8_t>"
#define MVDB_NAME_float "mvdb::MVDB<float>"


enum DATA_TYPES : unsigned char {
    INT8 = 0,
    INT16 = 1,
    UINT8 = 2,
    FLOAT = 3,
};

static PyObject* hello(PyObject* self, PyObject* args) {
    int num;
    if (!PyArg_ParseTuple(args, "i", &num)) return nullptr;
    std::cout << "NPY_INT = " << NPY_INT << std::endl;
    std::cout << "hello with number = " << num << std::endl;
    Py_RETURN_NONE;
}

static void MVDB_delete_int8_t(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_int8_t));
}

static void MVDB_delete_int16_t(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_int16_t));
}


static void MVDB_delete_uint8_t(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_uint8_t));
}


static void MVDB_delete_float(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_float));
}


static PyObject* MVDB_init(PyObject* self, PyObject* args) {
    uint8_t data_type;
    if(!PyArg_ParseTuple(args, "B", &data_type)) return nullptr;
    if(data_type == INT8) {
        auto* mvdb_ = new mvdb::MVDB<int8_t>();
        return PyCapsule_New(mvdb_, MVDB_NAME_int8_t, MVDB_delete_int8_t);
    }
    else if(data_type == INT16){
        auto* mvdb_ = new mvdb::MVDB<int16_t>();
        return PyCapsule_New(mvdb_, MVDB_NAME_int16_t, MVDB_delete_int16_t);
    }
    else if(data_type == UINT8) {
        auto* mvdb_ = new mvdb::MVDB<uint8_t>();
        return PyCapsule_New(mvdb_, MVDB_NAME_uint8_t, MVDB_delete_uint8_t);
    }
    else if(data_type == FLOAT) {
        auto* mvdb_ = new mvdb::MVDB<float>();
        return PyCapsule_New(mvdb_, MVDB_NAME_float, MVDB_delete_float);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return nullptr;
    }
}

static PyObject* MVDB_get_dims(PyObject* self, PyObject* args) {
    unsigned char data_type;
    PyObject * mvdb_capsule;
    if (!PyArg_ParseTuple(args, "BO", &data_type, &mvdb_capsule)) return nullptr;
    if(data_type == INT8){
        auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == INT16) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int16_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == UINT8) {
        auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == FLOAT) {
        auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return nullptr;
    }
}

int8_t* extract_int8_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if(PyArray_TYPE(pyarray) != NPY_INT8){
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.int8");
        return nullptr;
    }
    auto *data_arr = (int8_t*)PyArray_DATA(pyarray);
    if(data_arr == nullptr){
        PyErr_SetString(PyExc_BufferError, "failed to extract np.uint8 array data");
        return nullptr;
    }
    return data_arr;
}

int16_t* extract_int16_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_INT16) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.int16");
        return nullptr;
    }
    auto *data_arr = (int16_t*)PyArray_DATA(pyarray);
    if(data_arr == nullptr){
        PyErr_SetString(PyExc_BufferError, "failed to extract np.uint8 array data");
        return nullptr;
    }
    return data_arr;
}

uint8_t* extract_uint8_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.uint8");
        return nullptr;
    }
    auto *data_arr =  (uint8_t*)PyArray_DATA(pyarray);
    if(data_arr == nullptr){
        PyErr_SetString(PyExc_BufferError, "failed to extract np.uint8 array data");
        return nullptr;
    }
    return data_arr;
}

float* extract_float_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.float");
        return nullptr;
    }
    auto *data_arr = (float *)PyArray_DATA(pyarray);
    if(data_arr == nullptr){
        PyErr_SetString(PyExc_BufferError, "failed to extract np.uint8 array data");
        return nullptr;
    }
    return data_arr;
}


mvdb::NamedArgs* extract_named_args(unsigned char index_type, PyObject* args_capsule) {
//    if(index_type == mvdb::index::IndexType::FAISS_FLAT)
//        return static_cast<mvdb::index::FaissFlatIndexNamedArgs*>(PyCapsule_GetPointer(args_capsule, FAISS_FLAT_INDEX_NAMED_ARGS));
//    else if(index_type == mvdb::index::IndexType::ANNOY)
    if(index_type == mvdb::index::IndexType::ANNOY)
        return static_cast<mvdb::index::AnnoyIndexNamedArgs*>(PyCapsule_GetPointer(args_capsule, ANNOY_INDEX_NAMED_ARGS));
//    else if(index_type == mvdb::index::IndexType::SPANN)
//        return static_cast<mvdb::index::SPANNIndexNamedArgs*>(PyCapsule_GetPointer(args_capsule, SPANN_INDEX_NAMED_ARGS));
    return nullptr;
}

static PyObject* MVDB_create(PyObject* self, PyObject* args) {
    unsigned char data_type, index_type;
    PyObject *mvdb_capsule, *initial_data, *create_args_capsule;
    uint64_t dims, initial_data_size;
    const char *path, *initial_data_path;
    if (!PyArg_ParseTuple(args, "BOBlssO!lO", &data_type, &mvdb_capsule, &index_type, &dims, &path, &initial_data_path, &PyArray_Type, &initial_data, &initial_data_size, &create_args_capsule)) return nullptr;

    void* extracted_data = nullptr;
    mvdb::NamedArgs* c_args = extract_named_args(index_type, create_args_capsule);
    if (!c_args) {
        PyErr_SetString(PyExc_TypeError, "Failed to extract NamedArgs");
        return nullptr;
    }
    switch (data_type) {
        case INT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
            if (initial_data_size > 0) extracted_data = extract_int8_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (int8_t*)extracted_data, initial_data_size, c_args);
            Py_RETURN_TRUE;
        }
        case INT16: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int16_t));
            if (initial_data_size > 0) extracted_data = extract_int16_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (int16_t*)extracted_data, initial_data_size, c_args);
            Py_RETURN_TRUE;
        }
        case UINT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
            if (initial_data_size > 0) extracted_data = extract_uint8_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (uint8_t*)extracted_data, initial_data_size, c_args);
            Py_RETURN_TRUE;
        }
        case FLOAT: {
            auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
            if (initial_data_size > 0) extracted_data = extract_float_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (float*)extracted_data, initial_data_size, c_args);
            Py_RETURN_TRUE;
        }
        default: {
            PyErr_SetString(PyExc_ValueError, "Unsupported data type");
            return nullptr;
        }
    }
}

static PyObject* MVDB_open(PyObject* self, PyObject* args) {
    uint8_t data_type;
    PyObject* mvdb_capsule;
    const char* path;

    if (!PyArg_ParseTuple(args, "BOs", &data_type, &mvdb_capsule, &path)) return nullptr;

    switch (data_type) {
        case INT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int8_t> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
            mvdb_->open(path);
            break;
        }
        case INT16: {
            auto * mvdb_ = static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int16_t));
            mvdb_->open(path);
            break;
        }
        case UINT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
            mvdb_->open(path);
            break;
        }
        case FLOAT: {
            auto *mvdb_ = static_cast<mvdb::MVDB<float> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
            mvdb_->open(path);
            break;
        }
        default: {
            PyErr_SetString(PyExc_ValueError, "Unsupported data type");
            return nullptr;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* MVDB_get_built(PyObject* self, PyObject* args) {
    unsigned char data_type;
    PyObject * mvdb_capsule;
    if (!PyArg_ParseTuple(args, "BO", &data_type, &mvdb_capsule)) return nullptr;
    if(data_type == INT8){
        auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
        if(mvdb_ != nullptr && mvdb_->get_db_() != nullptr && mvdb_->get_db_()->index() != nullptr)
            return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
        return PyLong_FromLong(false);
    } else if (data_type == INT16) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int16_t >*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int16_t));
        if(mvdb_ != nullptr && mvdb_->get_db_() != nullptr && mvdb_->get_db_()->index() != nullptr)
            return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
        return PyLong_FromLong(false);
    } else if (data_type == UINT8) {
        auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
        if(mvdb_ != nullptr && mvdb_->get_db_() != nullptr && mvdb_->get_db_()->index() != nullptr)
            return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
        return PyLong_FromLong(false);
    } else if (data_type == FLOAT) {
        auto *mvdb_ = static_cast<mvdb::MVDB<float> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        if(mvdb_ == nullptr) std::cout << "DONE\n";
        if(mvdb_ != nullptr && mvdb_->get_db_() != nullptr && mvdb_->get_db_()->index() != nullptr)
            return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
        return PyLong_FromLong(false);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return nullptr;
    }
}

static PyObject* MVDB_get_num_items(PyObject* self, PyObject* args) {
    unsigned char data_type;
    PyObject * mvdb_capsule;
    if (!PyArg_ParseTuple(args, "BO", &data_type, &mvdb_capsule)) return nullptr;
    if(data_type == INT8){
        auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == INT16) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int16_t >*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int16_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == UINT8) {
        auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == FLOAT) {
        auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return nullptr;
    }
}

static PyObject* MVDB_topk(PyObject* self, PyObject* args) {
    uint8_t data_type;
    PyObject *mvdb_capsule, *query_array_obj = nullptr;
    uint64_t nq = 0, k;
    double c;
    const char *query_path = "", *result_path = "";
    mvdb::index::DISTANCE_METRIC metric;

    if (!PyArg_ParseTuple(args, "BO!lO!ssBl", &data_type, &PyCapsule_Type, &mvdb_capsule, &nq, &PyArray_Type, &query_array_obj, &query_path, &result_path, &k, &metric, &c)) return nullptr;

    if (query_path && strlen(query_path) > 0) {
        nq = mvdb::xvecs_num_vecs<float>(query_path);
        query_array_obj = nullptr;
    }

    mvdb::idx_t *ids = nullptr;
    void *distances = nullptr;
    npy_intp return_arr_dims[2] = {static_cast<npy_intp>(nq), static_cast<npy_intp>(k)};

    if (result_path == nullptr || strlen(result_path) == 0) {
        ids = (mvdb::idx_t*)malloc(nq * k * sizeof(mvdb::idx_t));
        switch (data_type) {
            case INT8:
            case UINT8:
                distances = malloc(nq * k * sizeof(uint8_t));
                break;
            case INT16:
                distances = malloc(nq * k * sizeof(int16_t));
                break;
            case FLOAT:
                distances = malloc(nq * k * sizeof(float));
                break;
            default:
                PyErr_SetString(PyExc_ValueError, "Unsupported data type for distances");
                free(ids);
                return nullptr;
        }
        if (!ids || !distances) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for ids or distances");
            if (ids) free(ids);
            if (distances) free(distances);
            return nullptr;
        }
    }

    switch (data_type) {
        case INT8:
            ((mvdb::MVDB<int8_t>*)mvdb_capsule)->topk(nq, extract_int8_arr(query_array_obj), std::string(query_path), std::string(result_path), ids, (int8_t*)distances, k, metric, (float)c);
            break;
        case UINT8:
            ((mvdb::MVDB<uint8_t>*)mvdb_capsule)->topk(nq, extract_uint8_arr(query_array_obj), std::string(query_path), std::string(result_path), ids, (uint8_t*)distances, k, metric, (float)c);
            break;
        case INT16:
            ((mvdb::MVDB<int16_t>*)mvdb_capsule)->topk(nq, extract_int16_arr(query_array_obj), std::string(query_path), std::string(result_path), ids, (int16_t*)distances, k, metric, (float)c);
            break;
        case FLOAT:
            ((mvdb::MVDB<float>*)mvdb_capsule)->topk(nq, extract_float_arr(query_array_obj), std::string(query_path), std::string(result_path), ids, (float*)distances, k, metric, (float)c);
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Unsupported data type");
            if (ids) free(ids);
            if (distances) free(distances);
            return nullptr;
    }

//    if (!success) {
//        PyErr_SetString(PyExc_RuntimeError, "Top-k search failed");
//        if (ids) free(ids);
//        if (distances) free(distances);
//        return nullptr;
//    }

    if (result_path && strlen(result_path) > 0) {
//        std::cout << "Results written to " << result_path << std::endl;
        Py_RETURN_NONE;
    } else {
        PyObject *ids_npArray = PyArray_SimpleNewFromData(2, return_arr_dims, NPY_UINT64, ids);
        PyObject *distances_npArray = PyArray_SimpleNewFromData(2, return_arr_dims, data_type, distances);
        PyArray_ENABLEFLAGS((PyArrayObject*)ids_npArray, NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS((PyArrayObject*)distances_npArray, NPY_ARRAY_OWNDATA);

        PyObject *result_tuple = PyTuple_Pack(2, ids_npArray, distances_npArray);
        Py_DECREF(ids_npArray); // PyArray_SimpleNewFromData does not steal reference
        Py_DECREF(distances_npArray);
        return result_tuple;
    }
}

static PyObject* MVDB_add(PyObject *self, PyObject *args) {

    PyErr_SetString(PyExc_BaseException, "MVDB_add not yet implemented");
    return nullptr;

    unsigned char data_type;
    PyObject *mvdb_capsule, *input_array;
    int nv;

    if(!PyArg_ParseTuple(args, "BOO!i", &data_type, &mvdb_capsule, &PyArray_Type, &input_array, &nv)) return nullptr;

    npy_intp return_arr_dims[1] = {static_cast<npy_intp>(nv)};
    auto *ids = (mvdb::idx_t*)malloc(nv * sizeof(mvdb::idx_t));
//    auto* input_pyarray = (PyArrayObject*)input_array;
    PyObject *ids_npArray;

    switch (data_type) {
        case INT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
            break;
        }
        case INT16: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int16_t));
            break;
        }
        case UINT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
            break;
        }
        case FLOAT: {
            auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
            float *input = extract_float_arr(input_array);
//            idx_t* keys = mvdb_->insert();
            break;
        }
        default: {
            PyErr_SetString(PyExc_ValueError, "Unsupported data type");
            return nullptr;
        }
    }

//    auto* db = static_cast<mvdb::DB*>(PyCapsule_GetPointer(capsule, DB_NAME));
//    auto* input_pyarray = (PyArrayObject*)input_array;
//    npy_intp return_arr_dims[1] = {nv};
//    if(PyArray_TYPE(input_pyarray) != NPY_FLOAT){
//        PyErr_SetString(PyExc_TypeError, "input data must be of type float32");
//        return nullptr;
//    }
//    auto* v = (float*)PyArray_DATA(input_pyarray);
//    int64_t* keys = db->add_vector(nv, v);
//    if (!keys){
//        PyErr_SetString(PyExc_TypeError, "keys = nullptr => vector add failed");
//        return nullptr;
//    }
//    PyObject* keys_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_INT64, (void*)keys);
//    if (!keys_npArray){
//        PyErr_SetString(PyExc_TypeError, "keys_npArray = nullptr => failed to generate keys nparray");
//        return nullptr;
//    }
//    // If your data should not be freed by NumPy when the array is deleted,
//    // you should set the WRITEABLE flag to ensure Python code doesn't change the data.
//    // PyArray_CLEARFLAGS((PyArrayObject*)np_array, NPY_ARRAY_WRITEABLE);
//    return keys_npArray;
    Py_RETURN_NONE;
}
//
//static void FaissFlatIndexNamedArgs_delete(PyObject* capsule) {
//    delete static_cast<mvdb::index::FaissFlatIndexNamedArgs*>(PyCapsule_GetPointer(capsule, FAISS_FLAT_INDEX_NAMED_ARGS));
//}
//
//static PyObject* FaissFlatIndexNamedArgs_init(PyObject* self, PyObject* args) {
//    auto * na_ = new mvdb::index::FaissFlatIndexNamedArgs();
//    return PyCapsule_New(na_, FAISS_FLAT_INDEX_NAMED_ARGS, FaissFlatIndexNamedArgs_delete);
//}

static void AnnoyIndexNamedArgs_delete(PyObject* capsule) {
    delete static_cast<mvdb::index::AnnoyIndexNamedArgs*>(PyCapsule_GetPointer(capsule, ANNOY_INDEX_NAMED_ARGS));
}

static PyObject* AnnoyIndexNamedArgs_init(PyObject* self, PyObject* args) {
    int n_trees, n_threads;
    if (!PyArg_ParseTuple(args, "ii", &n_trees, &n_threads)) return nullptr;
    auto * na_ = new mvdb::index::AnnoyIndexNamedArgs();
    na_->n_trees = n_trees;
    na_->n_threads = n_threads;
    return PyCapsule_New(na_, ANNOY_INDEX_NAMED_ARGS, AnnoyIndexNamedArgs_delete);
}

//static void SPANNIndexNamedArgs_delete(PyObject* capsule) {
//    delete static_cast<mvdb::index::SPANNIndexNamedArgs*>(PyCapsule_GetPointer(capsule, SPANN_INDEX_NAMED_ARGS));
//}
//
//static PyObject* SPANNIndexNamedArgs_init(PyObject* self, PyObject* args) {
//    const char *build_config_path, *quantizer_path;
//    PyObject *meta_mapping, *normalized;
//    unsigned int thread_num;
//    if (!PyArg_ParseTuple(args, "ssOOp", &build_config_path, &quantizer_path, &meta_mapping, &normalized, &thread_num)) return nullptr;
//    auto * na_ = new mvdb::index::SPANNIndexNamedArgs();
//    std::string build_config_path_str = std::string(build_config_path);
//    if(!build_config_path_str.empty()) na_->build_config_path = build_config_path_str;
//    na_->quantizer_path = std::string(quantizer_path);
//    na_->meta_mapping = PyObject_IsTrue(meta_mapping);
//    na_->normalized = PyObject_IsTrue(normalized);
//    if(thread_num > 0) na_->thread_num = thread_num;
//    return PyCapsule_New(na_, SPANN_INDEX_NAMED_ARGS, SPANNIndexNamedArgs_delete);
//}

static PyMethodDef ExtensionMethods[] = {
//        { Python method name, C function to be called, arguments for this function, Docstring for this function },
        { "hello", hello, METH_VARARGS, "Say hello with a number" },
        { "MVDB_init", MVDB_init, METH_VARARGS, "Initialise an MVDB object" },
        { "MVDB_create", MVDB_create, METH_VARARGS, "Create an MVDB database" },
        { "MVDB_open", MVDB_open, METH_VARARGS, "Open an MVDB database" },
        { "MVDB_topk", MVDB_topk, METH_VARARGS, "Find topk results" },
        { "MVDB_add", MVDB_add, METH_VARARGS, "Add vectors to the vector database" },
        { "MVDB_get_dims", MVDB_get_dims, METH_VARARGS, "Returns number of dimensions in db index" },
        { "MVDB_get_built", MVDB_get_built, METH_VARARGS, "Returns built flag for db index" },
        { "MVDB_num_items", MVDB_get_num_items, METH_VARARGS, "Returns number of items in db index" },
//        { "FaissFlatIndexNamedArgs_init", FaissFlatIndexNamedArgs_init, METH_VARARGS, "Return new named args obj for faiss flat index" },
        { "AnnoyIndexNamedArgs_init", AnnoyIndexNamedArgs_init, METH_VARARGS, "Return new named args obj for ANNOY index" },
//        { "SPANNIndexNamedArgs_init", SPANNIndexNamedArgs_init, METH_VARARGS, "Return new named args obj for SPANN index" },
        { NULL, NULL, 0, NULL }  // Sentinel value ending the array
};

// Module definition
static struct PyModuleDef extensionmodule = {
    PyModuleDef_HEAD_INIT,
    "microvecdb",    // Name of the module
    NULL,            // Module documentation, NULL for none
    -1,              // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    ExtensionMethods
};

// Initialization function for this module
PyMODINIT_FUNC PyInit_microvecdb(void) {
    import_array(); // Initialize NumPy API
    return PyModule_Create(&extensionmodule);
}
