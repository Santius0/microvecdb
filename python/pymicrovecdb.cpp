#include <Python.h>
#include <faiss_flat_index.h>
//#include <microvecdb.hpp>

// Function to be exposed - hello_world
static PyObject* hello_world(PyObject *self, PyObject *args) {
    printf("Hello, World!\n");
    printf("Hello, World FUCK!\n");
//    auto* micro_vec_db = new mvdb::VectorDB("./test_mvdb", "test_mvdb");
//    std::cout << micro_vec_db << std::endl;
//    delete micro_vec_db;
    Py_RETURN_NONE;
}

// Method definition object for this extension, describes the hello_world function
static PyMethodDef MyExtensionMethods[] = {
    { "hello_world",  // Python method name
      hello_world,    // C function to be called
      METH_NOARGS,    // No arguments for this function
      "Print 'Hello, World!'" }, // Docstring for this function
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
    return PyModule_Create(&myextensionmodule);
}
