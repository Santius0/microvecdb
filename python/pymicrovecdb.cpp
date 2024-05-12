#include <Python.h>
#include <mvdb.h>
#include <faiss_flat_index.h>
#include <annoy_index.h>
#include <spann_index.h>
#include <iostream>


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

//#define DB_NAME "mvdb::DB"

#define NAMED_ARGS "mvdb::NamedArgs"
#define FAISS_FLAT_INDEX_NAMED_ARGS "mvdb::index::FaissFLatIndexNamedArgs"
#define ANNOY_INDEX_NAMED_ARGS "mvdb::index::AnnoyIndexNamedArgs"
#define SPANN_INDEX_NAMED_ARGS "mvdb::index::SPANNIndexNamedArgs"

#define MVDB_NAME_int8_t "mvdb::MVDB<int8_t>"
#define MVDB_NAME_int16_t "mvdb::MVDB<int16_t>"
#define MVDB_NAME_int32_t "mvdb::MVDB<int32_t>"
#define MVDB_NAME_int64_t "mvdb::MVDB<int64_t>"
#define MVDB_NAME_uint8_t "mvdb::MVDB<uint8_t>"
#define MVDB_NAME_uint16_t "mvdb::MVDB<uint16_t>"
#define MVDB_NAME_uint32_t "mvdb::MVDB<uint32_t>"
#define MVDB_NAME_uint64_t "mvdb::MVDB<uint64_t>"
#define MVDB_NAME_float "mvdb::MVDB<float>"
#define MVDB_NAME_double "mvdb::MVDB<double>"


enum DATA_TYPES : unsigned char {
    INT8 = 0,
    INT16 = 1,
    INT32 = 2,
    INT64 = 3,
    UINT8 = 4,
    UINT16 = 5,
    UINT32 = 6,
    UINT64 = 7,
    FLOAT = 8,
    DOUBLE = 9,
};

static void MVDB_delete_int8_t(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_int8_t));
}

static void MVDB_delete_int16_t(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_int16_t));
}

static void MVDB_delete_int32_t(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<int32_t>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_int32_t));
}

static void MVDB_delete_uint8_t(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_uint8_t));
}

static void MVDB_delete_uint16_t(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<uint16_t>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_uint16_t));
}

static void MVDB_delete_uint32_t(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<uint32_t>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_uint32_t));
}

static void MVDB_delete_float(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_float));
}

static void MVDB_delete_double(PyObject* capsule) {
    delete static_cast<mvdb::MVDB<double>*>(PyCapsule_GetPointer(capsule, MVDB_NAME_double));
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
    else if(data_type == INT32) {
        auto* mvdb_ = new mvdb::MVDB<int32_t>();
        return PyCapsule_New(mvdb_, MVDB_NAME_int32_t, MVDB_delete_int32_t);
    }
    else if(data_type == UINT8) {
        auto* mvdb_ = new mvdb::MVDB<uint8_t>();
        return PyCapsule_New(mvdb_, MVDB_NAME_uint8_t, MVDB_delete_uint8_t);
    }
    else if(data_type == UINT16) {
        auto* mvdb_ = new mvdb::MVDB<uint16_t>();
        return PyCapsule_New(mvdb_, MVDB_NAME_uint16_t, MVDB_delete_uint16_t);
    }
    else if(data_type == UINT32) {
        auto* mvdb_ = new mvdb::MVDB<uint32_t>();
        return PyCapsule_New(mvdb_, MVDB_NAME_uint32_t, MVDB_delete_uint32_t);
    }
    else if(data_type == FLOAT) {
        auto* mvdb_ = new mvdb::MVDB<float>();
        return PyCapsule_New(mvdb_, MVDB_NAME_float, MVDB_delete_float);
    }
    else if(data_type == DOUBLE) {
        auto* mvdb_ = new mvdb::MVDB<double>();
        return PyCapsule_New(mvdb_, MVDB_NAME_double, MVDB_delete_double);
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
        auto *mvdb_ = static_cast<mvdb::MVDB<int16_t >*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int16_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == INT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int32_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == INT64) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int64_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == UINT8) {
        auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == UINT16) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint16_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == UINT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint32_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == UINT64) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint64_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == FLOAT) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    } else if (data_type == DOUBLE) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_double));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->dims());
    }
}

int8_t* extract_int8_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if(PyArray_TYPE(pyarray) != NPY_INT8){
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.int8");
        return nullptr;
    }
    return (int8_t*)PyArray_DATA(pyarray);
}

int16_t* extract_int16_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_INT16) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.int16");
        return nullptr;
    }
    return (int16_t*)PyArray_DATA(pyarray);
}

int32_t* extract_int32_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.int32");
        return nullptr;
    }
    return (int32_t*)PyArray_DATA(pyarray);
}

int64_t* extract_int64_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_INT64) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.int64");
        return nullptr;
    }
    return (int64_t*)PyArray_DATA(pyarray);
}

uint8_t* extract_uint8_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.uint8");
        return nullptr;
    }
    return (uint8_t*)PyArray_DATA(pyarray);
}

uint16_t* extract_uint16_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_UINT16) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.uint16");
        return nullptr;
    }
    return (uint16_t*)PyArray_DATA(pyarray);
}

uint32_t* extract_uint32_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_UINT32) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.uint32");
        return nullptr;
    }
    return (uint32_t*)PyArray_DATA(pyarray);
}

uint64_t* extract_uint64_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_UINT64) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.uint64");
        return nullptr;
    }
    return (uint64_t*)PyArray_DATA(pyarray);
}

float* extract_float_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.float");
        return nullptr;
    }
    return (float*)PyArray_DATA(pyarray);
}

double* extract_double_arr(PyObject* arr) {
    auto* pyarray = (PyArrayObject*)arr;
    if (PyArray_TYPE(pyarray) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "input data must be of type np.double");
        return nullptr;
    }
    return (double*)PyArray_DATA(pyarray);
}

mvdb::NamedArgs* extract_named_args(unsigned char index_type, PyObject* args_capsule) {
    if(index_type == mvdb::index::IndexType::FAISS_FLAT)
        return static_cast<mvdb::index::FaissFlatIndexNamedArgs*>(PyCapsule_GetPointer(args_capsule, FAISS_FLAT_INDEX_NAMED_ARGS));
    else if(index_type == mvdb::index::IndexType::ANNOY)
        return static_cast<mvdb::index::AnnoyIndexNamedArgs*>(PyCapsule_GetPointer(args_capsule, ANNOY_INDEX_NAMED_ARGS));
    else if(index_type == mvdb::index::IndexType::SPANN)
        return static_cast<mvdb::index::SPANNIndexNamedArgs*>(PyCapsule_GetPointer(args_capsule, SPANN_INDEX_NAMED_ARGS));
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
        case INT32: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int32_t));
            if (initial_data_size > 0) extracted_data = extract_int32_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (int32_t*)extracted_data, initial_data_size, c_args);
            Py_RETURN_TRUE;
        }
        case INT64: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int64_t));
            if (initial_data_size > 0) extracted_data = extract_int64_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (int64_t*)extracted_data, initial_data_size, c_args);
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
        case UINT16: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint16_t));
            if (initial_data_size > 0) extracted_data = extract_uint16_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (uint16_t*)extracted_data, initial_data_size, c_args);
            Py_RETURN_TRUE;
        }
        case UINT32: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint32_t));
            if (initial_data_size > 0) extracted_data = extract_uint32_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (uint32_t*)extracted_data, initial_data_size, c_args);
            Py_RETURN_TRUE;
        }
        case UINT64: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint64_t));
            if (initial_data_size > 0) extracted_data = extract_uint64_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (uint64_t*)extracted_data, initial_data_size, c_args);
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
        case DOUBLE: {
            auto *mvdb_ = static_cast<mvdb::MVDB<double>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_double));
            if (initial_data_size > 0) extracted_data = extract_double_arr(initial_data);
            if (initial_data_size > 0 && !extracted_data) {
                PyErr_SetString(PyExc_TypeError, "Failed to extract initial data");
                return nullptr;
            }
            mvdb_->create(static_cast<mvdb::index::IndexType>(index_type), dims, std::string(path), std::string(initial_data_path), (double*)extracted_data, initial_data_size, c_args);
            Py_RETURN_TRUE;
        }
        default:
            PyErr_SetString(PyExc_ValueError, "Failed to create MVDB db instance: unsupported data type");
            return nullptr;
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
        case INT32: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int32_t> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int32_t));
            mvdb_->open(path);
            break;
        }
        case INT64: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int64_t> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int64_t));
            mvdb_->open(path);
            break;
        }
        case UINT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
            mvdb_->open(path);
            break;
        }
        case UINT16: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint16_t> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint16_t));
            mvdb_->open(path);
            break;
        }
        case UINT32: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint32_t> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint32_t));
            mvdb_->open(path);
            break;
        }
        case UINT64: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint64_t> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint64_t));
            mvdb_->open(path);
            break;
        }
        case FLOAT: {
            auto *mvdb_ = static_cast<mvdb::MVDB<float> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
            mvdb_->open(path);
            break;
        }
        case DOUBLE: {
            auto *mvdb_ = static_cast<mvdb::MVDB<double> *>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_double));
            mvdb_->open(path);
            break;
        }
        default:
            PyErr_SetString(PyExc_ValueError, "Failed to open MVDB db instance: unsupported data type");
            return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* MVDB_get_built(PyObject* self, PyObject* args) {
    unsigned char data_type;
    PyObject * mvdb_capsule;
    if (!PyArg_ParseTuple(args, "BO", &data_type, &mvdb_capsule)) return nullptr;
    if(data_type == INT8){
        auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
    } else if (data_type == INT16) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int16_t >*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int16_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
    } else if (data_type == INT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int32_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
    } else if (data_type == INT64) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int64_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
    } else if (data_type == UINT8) {
        auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
    } else if (data_type == UINT16) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint16_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
    } else if (data_type == UINT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint32_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
    } else if (data_type == UINT64) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint64_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
    } else if (data_type == FLOAT) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
    } else if (data_type == DOUBLE) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_double));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->built());
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
    } else if (data_type == INT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int32_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == INT64) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int64_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == UINT8) {
        auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == UINT16) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint16_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == UINT32) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint32_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == UINT64) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint64_t));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == FLOAT) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    } else if (data_type == DOUBLE) {
        auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_double));
        return PyLong_FromLong((long)mvdb_->get_db_()->index()->ntotal());
    }
}

static PyObject* MVDB_topk(PyObject* self, PyObject* args) {
    uint8_t data_type;
    PyObject *mvdb_capsule, *query_input;
    uint64_t nq, k;
    double c;
    mvdb::index::DISTANCE_METRIC metric;

    if (!PyArg_ParseTuple(args, "BOlO!lBd", &data_type, &mvdb_capsule, &nq, &PyArray_Type, &query_input, &k, &metric, &c)) return nullptr;

    npy_intp return_arr_dims[1] = {static_cast<npy_intp>(nq*k)};
    auto *ids = (mvdb::idx_t*)malloc(nq * k * sizeof(mvdb::idx_t));
    void *distances;
    auto* query_pyarray = (PyArrayObject*)query_input;
    PyObject *ids_npArray, *distances_npArray;

    switch (data_type) {
        case INT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int8_t));
            if (PyArray_TYPE(query_pyarray) != NPY_INT8) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type int8_t");
                return nullptr;
            }
            auto *query = (int8_t*)PyArray_DATA(query_pyarray);
            distances = (int8_t*)malloc(k * sizeof(int8_t));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (int8_t*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_INT8, (void*)distances);
            break;
        }
        case INT16: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int16_t));
            if (PyArray_TYPE(query_pyarray) != NPY_INT16) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type int16_t");
                return nullptr;
            }
            auto *query = (int16_t*)PyArray_DATA(query_pyarray);
            distances = (int16_t*)malloc(k * sizeof(int16_t));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (int16_t*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_INT16, (void*)distances);
            break;
        }
        case INT32: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int32_t));
            if (PyArray_TYPE(query_pyarray) != NPY_INT32) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type int32_t");
                return nullptr;
            }
            auto *query = (int32_t*)PyArray_DATA(query_pyarray);
            distances = (int32_t*)malloc(k * sizeof(int32_t));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (int32_t*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_INT32, (void*)distances);
            break;
        }
        case INT64: {
            auto *mvdb_ = static_cast<mvdb::MVDB<int64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_int64_t));
            if (PyArray_TYPE(query_pyarray) != NPY_INT64) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type int64_t");
                return nullptr;
            }
            auto *query = (int64_t*)PyArray_DATA(query_pyarray);
            distances = (int64_t*)malloc(k * sizeof(int64_t));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (int64_t*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_INT64, (void*)distances);
            break;
        }
        case UINT8: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint8_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint8_t));
            if (PyArray_TYPE(query_pyarray) != NPY_UINT8) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type uint8_t");
                return nullptr;
            }
            auto *query = (uint8_t*)PyArray_DATA(query_pyarray);
            distances = (uint8_t*)malloc(k * sizeof(uint8_t));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (uint8_t*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_UINT8, (void*)distances);
            break;
        }
        case UINT16: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint16_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint16_t));
            if (PyArray_TYPE(query_pyarray) != NPY_UINT16) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type uint16_t");
                return nullptr;
            }
            auto *query = (uint16_t*)PyArray_DATA(query_pyarray);
            distances = (uint16_t*)malloc(k * sizeof(uint16_t));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (uint16_t*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_UINT16, (void*)distances);
            break;
        }
        case UINT32: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint32_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint32_t));
            if (PyArray_TYPE(query_pyarray) != NPY_UINT32) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type uint32_t");
                return nullptr;
            }
            auto *query = (uint32_t*)PyArray_DATA(query_pyarray);
            distances = (uint32_t*)malloc(k * sizeof(uint32_t));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (uint32_t*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_UINT32, (void*)distances);
            break;
        }
        case UINT64: {
            auto *mvdb_ = static_cast<mvdb::MVDB<uint64_t>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_uint64_t));
            if (PyArray_TYPE(query_pyarray) != NPY_UINT64) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type uint64_t");
                return nullptr;
            }
            auto *query = (uint64_t*)PyArray_DATA(query_pyarray);
            distances = (uint64_t*)malloc(k * sizeof(uint64_t));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (uint64_t*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_UINT64, (void*)distances);
            break;
        }
        case FLOAT: {
            auto *mvdb_ = static_cast<mvdb::MVDB<float>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_float));
            if (PyArray_TYPE(query_pyarray) != NPY_FLOAT) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type float");
                return nullptr;
            }
            auto *query = (float*)PyArray_DATA(query_pyarray);
            distances = (float*)malloc(nq * k * sizeof(float));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (float*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_FLOAT, (void*)distances);
            break;
        }
        case DOUBLE: {
            auto *mvdb_ = static_cast<mvdb::MVDB<double>*>(PyCapsule_GetPointer(mvdb_capsule, MVDB_NAME_double));
            if (PyArray_TYPE(query_pyarray) != NPY_DOUBLE) {
                PyErr_SetString(PyExc_TypeError, "Array should be of type double");
                return nullptr;
            }
            auto *query = (double*)PyArray_DATA(query_pyarray);
            distances = (double*)malloc(k * sizeof(double));
            if (!ids || !distances){
                PyErr_SetString(PyExc_TypeError, "either ids = nullptr or distances = nullptr => failed to generate allocate arrays for ids or distances");
                return nullptr;
            }
            mvdb_->topk(nq, query, ids, (double*)distances, k, metric, (float)c);
            distances_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_DOUBLE, (void*)distances);
            break;
        }
        default:
            PyErr_SetString(PyExc_ValueError, "Failed to find topk results: unsupported data type");
            return nullptr;
    }
    ids_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_INT64, (void*)ids);

    PyArray_ENABLEFLAGS((PyArrayObject*)ids_npArray, NPY_ARRAY_OWNDATA); // ensure numpy owns and manages this data. this will make sure numpy frees the data when it's done
    PyArray_ENABLEFLAGS((PyArrayObject*)distances_npArray, NPY_ARRAY_OWNDATA); // ensure numpy owns and manages this data. this will make sure numpy frees the data when it's done

    if (!ids_npArray || !distances_npArray){
        PyErr_SetString(PyExc_TypeError, "either ids_npArray = nullptr or distances_npArray = nullptr => failed to generate allocate arrays for ids or distances");
        return nullptr;
    }

    PyObject* return_tuple = PyTuple_New(2);
    if (!return_tuple) {
        PyErr_SetString(PyExc_AssertionError, "return_tuple == nullptr => failed to create python return tuple");
        return nullptr;
    }

    // PyTuple_SetItem steals a reference to the item
    if (0 != PyTuple_SetItem(return_tuple, 0, ids_npArray)) {
        // Handle error (and avoid memory leak)
        Py_DECREF(return_tuple);
        return nullptr;
    }

    if (0 != PyTuple_SetItem(return_tuple, 1, distances_npArray)) {
        // Handle error (and avoid memory leak)
        Py_DECREF(return_tuple);
        return nullptr;
    }
    return return_tuple;
}

//static PyObject* MVDB_add(PyObject *self, PyObject *args) {
//    unsigned char data_type;
//    PyObject *mvdb_capsule, *input_array;
//    int nv;
//
//    if(!PyArg_ParseTuple(args, "BOO!i", &data_type, &mvdb_capsule, &PyArray_Type, &input_array, &nv)) return nullptr;
//
//    switch (data_type) {
//        case INT8: {
//            auto * mvdb_
//            break;
//        }
//        case INT16: {
//            break;
//        }
//        case INT32: {
//            break;
//        }
//        case INT64: {
//            break;
//        }
//        case UINT8: {
//            break;
//        }
//        case UINT16: {
//            break;
//        }
//        case UINT32: {
//            break;
//        }
//        case UINT64: {
//            break;
//        }
//        case FLOAT: {
//            break;
//        }
//        case DOUBLE: {
//            break;
//        }
//    }
//
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
//}

static void FaissFlatIndexNamedArgs_delete(PyObject* capsule) {
    delete static_cast<mvdb::index::FaissFlatIndexNamedArgs*>(PyCapsule_GetPointer(capsule, FAISS_FLAT_INDEX_NAMED_ARGS));
}

static PyObject* FaissFlatIndexNamedArgs_init(PyObject* self, PyObject* args) {
    auto * na_ = new mvdb::index::FaissFlatIndexNamedArgs();
    return PyCapsule_New(na_, FAISS_FLAT_INDEX_NAMED_ARGS, FaissFlatIndexNamedArgs_delete);
}

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

static void SPANNIndexNamedArgs_delete(PyObject* capsule) {
    delete static_cast<mvdb::index::SPANNIndexNamedArgs*>(PyCapsule_GetPointer(capsule, SPANN_INDEX_NAMED_ARGS));
}

static PyObject* SPANNIndexNamedArgs_init(PyObject* self, PyObject* args) {
    //    if (!PyArg_ParseTuple(args, "BOlO!lBd", &data_type, &mvdb_capsule, &nq, &PyArray_Type, &query_input, &k, &metric, &c)) return nullptr;
    auto * na_ = new mvdb::index::SPANNIndexNamedArgs();
    return PyCapsule_New(na_, SPANN_INDEX_NAMED_ARGS, SPANNIndexNamedArgs_delete);
}

static PyMethodDef ExtensionMethods[] = {
//        { Python method name, C function to be called, arguments for this function, Docstring for this function },
        { "MVDB_init", MVDB_init, METH_VARARGS, "Initialise an MVDB object" },
        { "MVDB_create", MVDB_create, METH_VARARGS, "Create an MVDB database" },
        { "MVDB_open", MVDB_open, METH_VARARGS, "Open an MVDB database" },
        { "MVDB_topk", MVDB_topk, METH_VARARGS, "Find topk results" },
        { "MVDB_get_dims", MVDB_get_dims, METH_VARARGS, "Returns number of dimensions in db index" },
        { "MVDB_get_built", MVDB_get_built, METH_VARARGS, "Returns built flag for db index" },
        { "MVDB_num_items", MVDB_get_num_items, METH_VARARGS, "Returns number of items in db index" },
        { "FaissFlatIndexNamedArgs_init", FaissFlatIndexNamedArgs_init, METH_VARARGS, "Return new named args obj for faiss flat index" },
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
