#include <Python.h>
#include <iostream>
#include <constants.h>
#include <db.h>
#include <index.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define DB_NAME "mvdb::DB"
#define SEARCH_RESULT_NAME "mvdb::SearchResult"

// Function to be exposed - hello_world
static PyObject* hello_world(PyObject *self, PyObject *args) {
    const char *str;
    if(!PyArg_ParseTuple(args, "s", &str)) return nullptr;
    std::cout << "Hello " << str << std::endl;
    Py_RETURN_NONE;
}

static void SearchResult_delete(PyObject* capsule) {
    delete static_cast<mvdb::SearchResult*>(PyCapsule_GetPointer(capsule, SEARCH_RESULT_NAME));
}

static void DB_delete(PyObject* capsule) {
    delete static_cast<mvdb::DB*>(PyCapsule_GetPointer(capsule, DB_NAME));
}

static PyObject* DB_create(PyObject *self, PyObject *args) {
    const char *dbpath, *dbname;
    uint64_t dims;
    if(!PyArg_ParseTuple(args, "ssl", &dbname, &dbpath, &dims)) return nullptr;
    auto* db = new mvdb::DB(std::string(dbname), std::string(dbpath), dims);
    return PyCapsule_New(db, DB_NAME, DB_delete);
}

static PyObject* DB_add_vector(PyObject *self, PyObject *args) {
    PyObject *capsule, *input_array;
    int nv; // number of input vectors passed in;
    if(!PyArg_ParseTuple(args, "OO!i", &capsule, &PyArray_Type, &input_array, &nv)) return nullptr;
    auto* db = static_cast<mvdb::DB*>(PyCapsule_GetPointer(capsule, DB_NAME));
    auto* input_pyarray = (PyArrayObject*)input_array;
    npy_intp return_arr_dims[1] = {nv};
    if(PyArray_TYPE(input_pyarray) != NPY_FLOAT){
        PyErr_SetString(PyExc_TypeError, "input data must be of type float32");
        return nullptr;
    }
    auto* v = (float*)PyArray_DATA(input_pyarray);
    uint64_t* keys = db->add_vector(nv, v);
    if (!keys){
        PyErr_SetString(PyExc_TypeError, "keys = nullptr => vector add failed");
        return nullptr;
    }
    PyObject* keys_npArray = PyArray_SimpleNewFromData(1, return_arr_dims, NPY_UINT64, (void*)keys);
    if (!keys_npArray){
        PyErr_SetString(PyExc_TypeError, "keys_npArray = nullptr => failed to generate keys nparray");
        return nullptr;
    }
    // If your data should not be freed by NumPy when the array is deleted,
    // you should set the WRITEABLE flag to ensure Python code doesn't change the data.
    // PyArray_CLEARFLAGS((PyArrayObject*)np_array, NPY_ARRAY_WRITEABLE);
    return keys_npArray;
}

static PyObject* DB_add_data(PyObject *self, PyObject *args) {
    std::cout << "DB_add_data not implemented";
    Py_RETURN_NONE;
}

static PyObject* DB_add_data_with_vector(PyObject *self, PyObject *args) {
    std::cout << "DB_add_data_with_vector not implemented";
    Py_RETURN_NONE;
}

static PyObject* DB_search_with_vector(PyObject* self, PyObject* args) {
    PyObject *capsule, *query_input;
    int nq, k, ret_data; // nq = total size of query vector, should be a multiple of index dimensionality. k = number of results to be returned
    if (!PyArg_ParseTuple(args, "OiO!ip", &capsule, &nq, &PyArray_Type, &query_input, &k, &ret_data)) return nullptr;
    auto* db = static_cast<mvdb::DB*>(PyCapsule_GetPointer(capsule, DB_NAME));
    auto* query_pyarray = (PyArrayObject*)query_input;
    if (PyArray_TYPE(query_pyarray) != NPY_FLOAT) {
        PyErr_SetString(PyExc_TypeError, "Array should be of type float");
        return nullptr;
    }
    auto* query = (float*)PyArray_DATA(query_pyarray);

//    npy_intp dims[1] = {k};
//    PyObject* ids_pyArray = PyArray_SimpleNew(1, dims, NPY_INT64);
//    PyObject* distances_pyArray = PyArray_SimpleNew(1, dims, NPY_FLOAT);
//    if (!ids_pyArray || !distances_pyArray){
//        PyErr_SetString(PyExc_TypeError, "either ids_pyArray = nullptr or distances_pyArray = nullptr => failed to generate allocate arrays for ids or distances");
//        return nullptr;
//    }
//    auto* ids = (int64_t*)PyArray_DATA((PyArrayObject*)ids_pyArray);
//    auto* distances = (float*)PyArray_DATA((PyArrayObject*)distances_pyArray);

    mvdb::SearchResult* search_result = db->search_with_vector(nq, query, k, ret_data);
    for(int i = 0; i < search_result->size_; i++){
        std::cout << search_result->ids_[i] << " = " << search_result->distances_[i] << std::endl;
    }

    return PyCapsule_New(search_result, SEARCH_RESULT_NAME, SearchResult_delete);
}

static PyObject* DB_search(PyObject* self, PyObject* args){
    std::cout << "DB_search not implemented";
    Py_RETURN_NONE;
}

static PyObject* DB_start(PyObject* self, PyObject* args){
    std::cout << "DB_start not implemented";
    Py_RETURN_NONE;
}

static PyObject* DB_connect(PyObject* self, PyObject* args){
    std::cout << "DB_connect not implemented";
    Py_RETURN_NONE;
}

// Method definition object for this extension, describes the hello_world function
static PyMethodDef MyExtensionMethods[] = {
    { "hello_world",  // Python method name
      hello_world,    // C function to be called
      METH_VARARGS,    // No arguments for this function
      "Print 'Hello, World!'" }, // Docstring for this function
    { "DB_create", DB_create, METH_VARARGS, "Initialise a DB<> object" },
    { "DB_add_vector", DB_add_vector, METH_VARARGS, "Add vector data using a DB object" },
    { "DB_add_data", DB_add_data, METH_VARARGS, "Add data using a DB object" },
    { "DB_add_data_with_vector", DB_add_data_with_vector, METH_VARARGS, "Add raw data and vector data using a DB object" },
    { "DB_search_with_vector", DB_search_with_vector, METH_VARARGS, "Perform similarity using only vector data via a DB<> object" },
    { "DB_search", DB_search, METH_VARARGS, "Perform similarity using raw data via a DB object" },
    { "DB_start", DB_start, METH_VARARGS, "Start DB server" },
    { "DB_connect", DB_connect, METH_VARARGS, "Connect to distributed DB server" },
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
