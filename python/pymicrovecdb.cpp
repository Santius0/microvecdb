#include <Python.h>
#include <iostream>
#include <faiss_flat_index.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define FAISS_FLAT_INDEX_NAME "mvdb::FaissFlatIndex"

// Function to be exposed - hello_world
static PyObject* hello_world(PyObject *self, PyObject *args) {
    const char *str;
    if(!PyArg_ParseTuple(args, "s", &str)) return nullptr;
    std::cout << "Hello " << str << std::endl;
    Py_RETURN_NONE;
}

static void FaissFlatIndex_delete(PyObject* capsule) {
    delete static_cast<mvdb::FaissFlatIndex*>(PyCapsule_GetPointer(capsule, FAISS_FLAT_INDEX_NAME));
}

static PyObject* FaissFlatIndex_create(PyObject *self, PyObject *args) {
    const char* index_path;
    uint64_t dims;
    if(!PyArg_ParseTuple(args, "sl", &index_path, &dims)) return nullptr;
    auto *flat_index = new mvdb::FaissFlatIndex(std::string(index_path), dims);
    return PyCapsule_New(flat_index, FAISS_FLAT_INDEX_NAME, FaissFlatIndex_delete);
}

static PyObject* FaissFlatIndex_open(PyObject *self, PyObject *args) {
    PyObject* capsule;
    if(!PyArg_ParseTuple(args, "O", &capsule)) return nullptr;
    auto* idx_obj = static_cast<mvdb::FaissFlatIndex*>(PyCapsule_GetPointer(capsule, FAISS_FLAT_INDEX_NAME));
    idx_obj->open();
//    size_t n = 4;
//    uint64_t keys[4] = {0, 1, 2, 3};
//    float* vecs = idx_obj->get(n, nullptr);
//    for(int i = 0; i < n * idx_obj->dims(); i++){
//        std::cout  << vecs[i] << (((i+1) % idx_obj->dims()) == 0 ? "\n" : " ");
//    }
    Py_RETURN_NONE;
}

static PyObject* FaissFlatIndex_add(PyObject* self, PyObject* args) {
    PyObject *capsule, *input_vector;
    int n; // number of input vectors passed in;
    if(!PyArg_ParseTuple(args, "OO!i", &capsule, &PyArray_Type, &input_vector, &n)) return nullptr;
    auto* idx_obj = static_cast<mvdb::FaissFlatIndex*>(PyCapsule_GetPointer(capsule, FAISS_FLAT_INDEX_NAME));
    auto* input_pyarray = (PyArrayObject*)input_vector;
    if (PyArray_TYPE(input_pyarray) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "Array should be of type float");
        return nullptr;
    }
    auto* input = (float*)PyArray_DATA(input_pyarray);
    npy_intp dims[1] = {static_cast<long>(idx_obj->dims())};
    PyObject* keys_pyArray = PyArray_SimpleNew(1, dims, NPY_FLOAT);
    if (!keys_pyArray) return nullptr;
    auto* keys = (uint64_t*)PyArray_DATA((PyArrayObject*)keys_pyArray);
    bool res = idx_obj->add(n, input, keys);
    return PyTuple_Pack(2, PyBool_FromLong(res), keys_pyArray);
    Py_RETURN_NONE;
}

static PyObject* FaissFlatIndex_search(PyObject* self, PyObject* args) {
    PyObject *capsule, *query_input;
    int nq, k; // nq = total size of query vector, should be a multiple of index dimensionality. k = number of results to be returned
    if (!PyArg_ParseTuple(args, "OiO!i", &capsule, &nq, &PyArray_Type, &query_input, &k)) return nullptr;
    auto* idx_obj = static_cast<mvdb::FaissFlatIndex*>(PyCapsule_GetPointer(capsule, FAISS_FLAT_INDEX_NAME));
    auto* query_pyarray = (PyArrayObject*)query_input;

    // Check if the array is of the correct type (uint64)
    if (PyArray_TYPE(query_pyarray) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "Array should be of type float");
        return nullptr;
    }

    auto* query = (float*)PyArray_DATA(query_pyarray);

    int array_size = k;
    npy_intp dims[1] = {array_size};

    // Create a new NumPy array of uint64
    PyObject* ids_pyArray = PyArray_SimpleNew(1, dims, NPY_INT64);
    PyObject* distances_pyArray = PyArray_SimpleNew(1, dims, NPY_FLOAT);
    if (!ids_pyArray || !distances_pyArray) return nullptr;

    auto* ids = (int64_t*)PyArray_DATA((PyArrayObject*)ids_pyArray);
    auto* distances = (float*)PyArray_DATA((PyArrayObject*)distances_pyArray);

    idx_obj->search(nq, query, ids, distances, k);

    return PyTuple_Pack(2, ids_pyArray, distances_pyArray);
}



// Method definition object for this extension, describes the hello_world function
static PyMethodDef MyExtensionMethods[] = {
    { "hello_world",  // Python method name
      hello_world,    // C function to be called
      METH_VARARGS,    // No arguments for this function
      "Print 'Hello, World!'" }, // Docstring for this function

    { "FaissFlatIndex_create", FaissFlatIndex_create, METH_VARARGS, "Creates index using faiss' flat index under the hood" },
    { "FaissFlatIndex_open", FaissFlatIndex_open, METH_VARARGS, "" },
    { "FaissFlatIndex_add", FaissFlatIndex_add, METH_VARARGS, "" },
    { "FaissFlatIndex_search", FaissFlatIndex_search, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL }  // Sentinel value ending the array
};

// Module definition
static struct PyModuleDef myextensionmodule = {
    PyModuleDef_HEAD_INIT,
    "microvecdb",  // Name of the module
    NULL,            // Module documentation, NULL for none
    -1,              // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    MyExtensionMethods
};

// Initialization function for this module
PyMODINIT_FUNC PyInit_microvecdb(void) {
    import_array(); // Initialize NumPy API
    return PyModule_Create(&myextensionmodule);
}
