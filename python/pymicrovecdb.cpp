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
    FLOAT32 = 1,
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
    else if(data_type == FLOAT32) {
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
    } else if (data_type == FLOAT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return nullptr;
    }
}

int8_t* extract_int8_arr(PyObject* arr) {
    if(arr == nullptr) return nullptr;
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
    if(arr == nullptr) return nullptr;
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
    if(arr == nullptr) return nullptr;
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
    if(arr == nullptr) return nullptr;
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.float32");
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
    #ifdef FAISS
    if(index_type == mvdb::index::IndexType::FAISS_FLAT)
        return static_cast<mvdb::index::FaissFlatIndexNamedArgs*>(PyCapsule_GetPointer(args_capsule, FAISS_FLAT_INDEX_NAMED_ARGS));
    else
    #endif
    if(index_type == mvdb::index::IndexType::ANNOY)
        return static_cast<mvdb::index::AnnoyIndexNamedArgs*>(PyCapsule_GetPointer(args_capsule, ANNOY_INDEX_NAMED_ARGS));
    else if(index_type == mvdb::index::IndexType::SPANN)
        return static_cast<mvdb::index::SPANNIndexNamedArgs*>(PyCapsule_GetPointer(args_capsule, SPANN_INDEX_NAMED_ARGS));
    return nullptr;
}

static PyObject* MVDB_create(PyObject* self, PyObject* args) {
    unsigned char data_type, index_type;
    PyObject *mvdb_capsule, *initial_data, *named_args_capsule, *data_array;
    uint64_t dims, initial_data_size;
    const char *path;
    if (!PyArg_ParseTuple(args, "BBOlsO!OlO", &data_type, &index_type, &mvdb_capsule, &dims, &path, &PyArray_Type, &initial_data, &data_array, &initial_data_size, &named_args_capsule)) return nullptr;

    void* vector_data;
    mvdb::NamedArgs *c_args = extract_named_args(index_type, named_args_capsule);
    if (!c_args) {
        PyErr_SetString(PyExc_TypeError, "Failed to extract NamedArgs");
        return nullptr;
    }

    std::string binary_data;
    size_t binary_data_sizes[initial_data_size > 0 ? initial_data_size : 1]; // stop seg fault if no initial data is passed
    if(data_array != Py_None) {
        auto *binary_data_pyarray_obj = (PyArrayObject*)data_array;

        if (!PyArray_Check(binary_data_pyarray_obj)) {
            PyErr_SetString(PyExc_TypeError, "Initial binary data must be a NumPy array");
            return nullptr;
        }

        if (PyArray_TYPE(binary_data_pyarray_obj) != NPY_OBJECT) {
            PyErr_SetString(PyExc_TypeError, "Expected an array of objects for initial binary data");
            return nullptr;
        }

        for (int i = 0; i < initial_data_size; ++i) {
            PyObject *obj = PyArray_GETITEM(binary_data_pyarray_obj, (char *)PyArray_GETPTR1(binary_data_pyarray_obj, i));
            if (!obj) continue;

            PyObject *serialized_obj_bytes = PyBytes_FromObject(obj);
            const char *serialized_obj = PyBytes_AS_STRING(serialized_obj_bytes);
            std::cout << serialized_obj << std::endl;
            Py_ssize_t serialized_obj_size = PyBytes_GET_SIZE(serialized_obj_bytes);

            binary_data.append(serialized_obj, serialized_obj_size);
            binary_data_sizes[i] = serialized_obj_size;

            Py_DECREF(serialized_obj_bytes);
        }
        std::cout << binary_data.c_str() << std::endl;
    }

    switch (data_type) {
        case INT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
            if (initial_data_size > 0) vector_data = extract_int8_arr(initial_data);
            if (initial_data_size > 0 && !vector_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type),
                          dims,
                          std::string(path),
                          (int8_t*)vector_data,
                          binary_data,
                          binary_data_sizes,
                          initial_data_size,
                          c_args);
            Py_RETURN_TRUE;
        }
        case FLOAT32: {
            auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
            if (initial_data_size > 0) vector_data = extract_float_arr(initial_data);
            if (initial_data_size > 0 && !vector_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type),
                          dims,
                          std::string(path),
                          (float*)vector_data,
                          binary_data,
                          binary_data_sizes,
                          initial_data_size,
                          c_args);
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
        case FLOAT32: {
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

static PyObject* MVDB_get_index_type(PyObject* self, PyObject* args) {
    unsigned char data_type;
    PyObject * mvdb_capsule;
    if (!PyArg_ParseTuple(args, "BO", &data_type, &mvdb_capsule)) return nullptr;
    if(data_type == INT8){
        auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
        if(mvdb_ != nullptr && mvdb_->get_db_() != nullptr && mvdb_->get_db_()->index() != nullptr)
            return PyLong_FromUnsignedLong((unsigned long)mvdb_->get_db_()->index()->type());
        return PyLong_FromLong(false);
    } else if (data_type == FLOAT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<float> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        if(mvdb_ != nullptr && mvdb_->get_db_() != nullptr && mvdb_->get_db_()->index() != nullptr)
            return PyLong_FromUnsignedLong((unsigned long)mvdb_->get_db_()->index()->type());
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return nullptr;
    }
    PyErr_SetString(PyExc_ValueError, "Failed to retrieve index type. Internal index not properly initialized");
    return nullptr;
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
    } else if (data_type == FLOAT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<float> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
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
    } else if (data_type == FLOAT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return nullptr;
    }
}

static PyObject* MVDB_topk(PyObject* self, PyObject* args) {
    uint8_t data_type, index_type;
    PyObject *mvdb_capsule, *query_array_obj, *named_args_capsule;
    int64_t nq, k;

    if (!PyArg_ParseTuple(args, "BBOO!LLO", &index_type, &data_type, &mvdb_capsule, &PyArray_Type, &query_array_obj, &nq, &k, &named_args_capsule)) return nullptr;

    mvdb::NamedArgs* c_args = extract_named_args(index_type, named_args_capsule);

    double peak_wss_mb = -1.0;

    if(data_type == INT8) {
        auto* pyarray = (PyArrayObject*)query_array_obj;
        if (PyArray_TYPE(pyarray) != NPY_INT8) {
            PyErr_SetString(PyExc_TypeError, "input data must be of type np.float");
            return nullptr;
        }
        auto *data_arr = (int8_t*)PyArray_DATA(pyarray);
        if(data_arr == nullptr){
            PyErr_SetString(PyExc_BufferError, "failed to extract np.uint8 array data");
            return nullptr;
        }

        npy_intp return_arr_dims[1] = {static_cast<npy_intp>(nq * k)};

        auto *ids = (mvdb::idx_t*)malloc(nq * k * sizeof(mvdb::idx_t));
        if(ids == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for ids");
            return nullptr;
        }

        auto *distances = (int8_t*)malloc(nq * k * sizeof(int8_t));
        if(distances == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for distances");
            free(ids);
            return nullptr;
        }

        auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
        mvdb_->knn(nq, data_arr, "", "", ids, distances, peak_wss_mb, k, c_args);

        PyObject *ids_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_INT64, ids);
        if(ids_npArray == nullptr){
            PyErr_SetString(PyExc_RuntimeError, "Failed to create return NumPy array, 'ids'");
            free(ids);
            free(distances);
            return nullptr;
        }

        PyObject *distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_INT8, distances);
        if(distances_npArray == nullptr){
            PyErr_SetString(PyExc_RuntimeError, "Failed to create return NumPy array, 'distances'");
            Py_DECREF(ids_npArray);
            free(ids);
            free(distances);
            return nullptr;
        }

        PyArray_ENABLEFLAGS((PyArrayObject *) ids_npArray, NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS((PyArrayObject *) distances_npArray, NPY_ARRAY_OWNDATA);

        PyObject* tuple = PyTuple_New(3);
        if (!tuple) {
            Py_DECREF(ids_npArray);
            Py_DECREF(distances_npArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to create return tuple.");
            return nullptr;
        }

        PyObject *peak_wss_mb_py_obj = Py_BuildValue("d", peak_wss_mb);

        PyTuple_SetItem(tuple, 0, ids_npArray);
        PyTuple_SetItem(tuple, 1, distances_npArray);
        PyTuple_SetItem(tuple, 2, peak_wss_mb_py_obj);

        return tuple;
    } else if(data_type == FLOAT32){
        auto* pyarray = (PyArrayObject*)query_array_obj;
        if (PyArray_TYPE(pyarray) != NPY_FLOAT) {
            PyErr_SetString(PyExc_TypeError, "input data must be of type np.float");
            return nullptr;
        }
        auto *data_arr = (float *)PyArray_DATA(pyarray);
        if(data_arr == nullptr){
            PyErr_SetString(PyExc_BufferError, "failed to extract np.uint8 array data");
            return nullptr;
        }

        npy_intp return_arr_dims[1] = {static_cast<npy_intp>(nq * k)};

        auto *ids = (mvdb::idx_t*)malloc(nq * k * sizeof(mvdb::idx_t));
        if(ids == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for ids");
            return nullptr;
        }

        auto *distances = (float*)malloc(nq * k * sizeof(float));
        if(distances == nullptr) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for distances");
            free(ids);
            return nullptr;
        }

        auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        mvdb_->knn(nq, data_arr, "", "", ids, distances, peak_wss_mb, k, c_args);

        PyObject *ids_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_INT64, ids);
        if(ids_npArray == nullptr){
            PyErr_SetString(PyExc_RuntimeError, "Failed to create return NumPy array, 'ids'");
            free(ids);
            free(distances);
            return nullptr;
        }

        PyObject *distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_FLOAT, distances);
        if(distances_npArray == nullptr){
            PyErr_SetString(PyExc_RuntimeError, "Failed to create return NumPy array, 'distances'");
            Py_DECREF(ids_npArray);
            free(ids);
            free(distances);
            return nullptr;
        }

        PyArray_ENABLEFLAGS((PyArrayObject *) ids_npArray, NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS((PyArrayObject *) distances_npArray, NPY_ARRAY_OWNDATA);

        PyObject* tuple = PyTuple_New(3);
        if (!tuple) {
            Py_DECREF(ids_npArray);
            Py_DECREF(distances_npArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to create return tuple.");
            return nullptr;
        }

        PyObject *peak_wss_mb_py_obj = Py_BuildValue("d", peak_wss_mb);

        PyTuple_SetItem(tuple, 0, ids_npArray);
        PyTuple_SetItem(tuple, 1, distances_npArray);
        PyTuple_SetItem(tuple, 2, peak_wss_mb_py_obj);

        return tuple;
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return nullptr;
    }
}

static PyObject* MVDB_get(PyObject* self, PyObject* args) {
    PyObject *mvdb_capsule;
    PyArrayObject *array;
    uint8_t data_type;

    if (!PyArg_ParseTuple(args, "BOO!", &data_type, &mvdb_capsule, &PyArray_Type, &array)) return nullptr;

    if(PyArray_TYPE(array) != NPY_UINT64) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.uint64");
        return nullptr;
    }

    // Print the array contents
    npy_intp size = PyArray_SIZE(array);
    if(size == 0) Py_RETURN_NONE;

    auto *data = (uint64_t *)PyArray_DATA(array);

    printf("Array keys:\n");
    for (npy_intp i = 0; i < size; ++i) {
        std::cout << data[i] << std::endl;
    }
    printf("\n");

    auto* values = new std::string[size];
    if(data_type == INT8) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
        mvdb_->get_obj(size, data, values);
    } else if(data_type == FLOAT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        mvdb_->get_obj(size, data, values);
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return nullptr;
    }

    npy_intp return_arr_dims[1] = {static_cast<npy_intp>(size)};
    PyObject *values_npArray = PyArray_SimpleNew(1, return_arr_dims, NPY_OBJECT);

    for (npy_intp i = 0; i < size; ++i) {
        PyObject* value_bytes = PyBytes_FromStringAndSize(values[i].c_str(), values[i].size());
        PyArray_SETITEM((PyArrayObject *)values_npArray, (char *) PyArray_GETPTR1((PyArrayObject *)values_npArray, i), value_bytes);
        Py_DECREF(value_bytes);
    }

    PyArray_ENABLEFLAGS((PyArrayObject *) values_npArray, NPY_ARRAY_OWNDATA);
    return values_npArray;
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
        case FLOAT32: {
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

#ifdef FAISS

static void FaissFlatIndexNamedArgs_delete(PyObject* capsule) {
    delete static_cast<mvdb::index::FaissFlatIndexNamedArgs*>(PyCapsule_GetPointer(capsule, FAISS_FLAT_INDEX_NAMED_ARGS));
}

static PyObject* FaissFlatIndexNamedArgs_init(PyObject* self, PyObject* args) {
    auto * na_ = new mvdb::index::FaissFlatIndexNamedArgs();
    return PyCapsule_New(na_, FAISS_FLAT_INDEX_NAMED_ARGS, FaissFlatIndexNamedArgs_delete);
}

#endif

static void AnnoyIndexNamedArgs_delete(PyObject* capsule) {
    delete static_cast<mvdb::index::AnnoyIndexNamedArgs*>(PyCapsule_GetPointer(capsule, ANNOY_INDEX_NAMED_ARGS));
}

static PyObject* AnnoyIndexNamedArgs_init(PyObject* self, PyObject* args) {
    int n_trees, n_threads, search_k;
    if (!PyArg_ParseTuple(args, "iii", &n_trees, &n_threads, &search_k)) return nullptr;
    auto * na_ = new mvdb::index::AnnoyIndexNamedArgs();
    na_->n_trees = n_trees;
    na_->n_threads = n_threads;
    na_->search_k = search_k;
    return PyCapsule_New(na_, ANNOY_INDEX_NAMED_ARGS, AnnoyIndexNamedArgs_delete);
}

static void SPANNIndexNamedArgs_delete(PyObject* capsule) {
    delete static_cast<mvdb::index::SPANNIndexNamedArgs*>(PyCapsule_GetPointer(capsule, SPANN_INDEX_NAMED_ARGS));
}

static PyObject* SPANNIndexNamedArgs_init(PyObject* self, PyObject* args) {
    const char *build_config_path, *quantizer_path, *truth_path;
    PyObject *meta_mapping, *normalized;
    int thread_num, batch_size, BKTKmeansK, Samples, TPTNumber, RefineIterations, NeighborhoodSize, CEF, MaxCheckForRefineGraph, NumberOfInitialDynamicPivots, GraphNeighborhoodScale, NumberOfOtherDynamicPivots;
    if (!PyArg_ParseTuple(args, "sssOOiiiiiiiiiiii", &build_config_path, &truth_path, &quantizer_path, &meta_mapping, &normalized, &thread_num, &batch_size,
                          &BKTKmeansK, &Samples, &TPTNumber, &RefineIterations, &NeighborhoodSize, &CEF, &MaxCheckForRefineGraph, &NumberOfInitialDynamicPivots, &GraphNeighborhoodScale,
                          &NumberOfOtherDynamicPivots)) return nullptr;
    auto * na_ = new mvdb::index::SPANNIndexNamedArgs();
    std::string build_config_path_str = std::string(build_config_path);
    if(!build_config_path_str.empty()) na_->build_config_path = build_config_path_str;
    na_->truth_path = std::string(truth_path);
    na_->quantizer_path = std::string(quantizer_path);
    na_->meta_mapping = PyObject_IsTrue(meta_mapping);
    na_->normalized = PyObject_IsTrue(normalized);
    if(thread_num > 0) na_->thread_num = thread_num;
    if(batch_size > 0) na_->batch_size = batch_size;
    // hyperparameters
    na_->BKTKmeansK = BKTKmeansK;
    na_->Samples = Samples;
    na_->TPTNumber = TPTNumber;
    na_->RefineIterations = RefineIterations;
    na_->NeighborhoodSize = NeighborhoodSize;
    na_->CEF = CEF;
    na_->MaxCheckForRefineGraph = MaxCheckForRefineGraph;
    na_->NumberOfInitialDynamicPivots = NumberOfInitialDynamicPivots;
    na_->GraphNeighborhoodScale = GraphNeighborhoodScale;
    na_->NumberOfOtherDynamicPivots = NumberOfOtherDynamicPivots;
    return PyCapsule_New(na_, SPANN_INDEX_NAMED_ARGS, SPANNIndexNamedArgs_delete);
}


static PyMethodDef ExtensionMethods[] = {
//        { Python method name, C function to be called, arguments for this function, Docstring for this function },
        { "hello", hello, METH_VARARGS, "Say hello with a number" },
        { "MVDB_init", MVDB_init, METH_VARARGS, "Initialise an MVDB object" },
        { "MVDB_create", MVDB_create, METH_VARARGS, "Create an MVDB database" },
        { "MVDB_open", MVDB_open, METH_VARARGS, "Open an MVDB database" },
        { "MVDB_topk", MVDB_topk, METH_VARARGS, "Find topk results" },
        { "MVDB_get", MVDB_get, METH_VARARGS, "Find topk results" },
        { "MVDB_add", MVDB_add, METH_VARARGS, "Add vectors to the vector database" },
        { "MVDB_get_index_type", MVDB_get_index_type, METH_VARARGS, "Returns number of dimensions in db index" },
        { "MVDB_get_dims", MVDB_get_dims, METH_VARARGS, "Returns number of dimensions in db index" },
        { "MVDB_get_built", MVDB_get_built, METH_VARARGS, "Returns built flag for db index" },
        { "MVDB_get_num_items", MVDB_get_num_items, METH_VARARGS, "Returns number of items in db index" },
        #ifdef FAISS
        { "FaissFlatIndexNamedArgs_init", FaissFlatIndexNamedArgs_init, METH_VARARGS, "Return new named args obj for faiss flat index" },
        #endif
        { "AnnoyIndexNamedArgs_init", AnnoyIndexNamedArgs_init, METH_VARARGS, "Return new named args obj for ANNOY index" },
        { "SPANNIndexNamedArgs_init", SPANNIndexNamedArgs_init, METH_VARARGS, "Return new named args obj for SPANN index" },
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
