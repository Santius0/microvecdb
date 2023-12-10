// run: python setup.py bdist_wheel sdist
//      from inside the 'python' folder (a.k.a this folder)
//      then do a: pip install .
//      from in the 'python' folder

#include <Python.h>
#include <microvecdb_c.h>

static PyObject* hello_world(PyObject *self, PyObject *args) {
    printf("Hello, world!\n");
    Py_RETURN_NONE;
}

static PyObject* KvStoreMetadata_new_wrapper(PyObject* self, PyObject* args) {
    const char* dataDirectoryPath;
    int create_if_missing;

    if (!PyArg_ParseTuple(args, "si", &dataDirectoryPath, &create_if_missing)) {
        return NULL;
    }
    mvdb::KvStoreMetadata* obj = KvStoreMetadata_new(dataDirectoryPath, create_if_missing);
    return PyCapsule_New(obj, "KvStoreMetadata", NULL);
}

static PyObject* KvStoreMetadata_delete_wrapper(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return NULL;
    }
    mvdb::KvStoreMetadata* obj = (mvdb::KvStoreMetadata*)PyCapsule_GetPointer(capsule, "KvStoreMetadata");
    KvStoreMetadata_delete(obj);
    Py_RETURN_NONE;
}

static PyMethodDef MicroVecDBMethods[] = {
    {"new", KvStoreMetadata_new_wrapper, METH_VARARGS, "Create a new KvStoreMetadata object."},
    {"delete", KvStoreMetadata_delete_wrapper, METH_VARARGS, "Delete a KvStoreMetadata object."},
    {"hello_world", hello_world, METH_NOARGS, "say hello to the world"}
};

static struct PyModuleDef microvecdbmodule = {
    PyModuleDef_HEAD_INIT,
    "microvecdb",
    NULL,
    -1,
    MicroVecDBMethods
};

PyMODINIT_FUNC PyInit_microvecdb(void) {
    return PyModule_Create(&microvecdbmodule);
}
