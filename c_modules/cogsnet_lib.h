#include <Python.h>

static PyObject *method__cogsnet(PyObject *self, PyObject *args);
static PyObject *CogsnetException;

static PyMethodDef CogsnetLibMethods[] = {
    {"_cogsnet", method__cogsnet, METH_VARARGS,
     "Process a file and return a list of lists."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

static struct PyModuleDef cogsnetlibmodule = {
    PyModuleDef_HEAD_INIT,
    "cogsnet_lib",     // Module name
    NULL,              // Module docstring
    -1,                // Module state
    CogsnetLibMethods  // Module methods
};

PyMODINIT_FUNC PyInit_cogsnet_lib(void) {
  PyObject *module = PyModule_Create(&cogsnetlibmodule);

  // Create and define your custom exception class
  CogsnetException =
      PyErr_NewException("cogsnetmodule.CogsnetException", NULL, NULL);
  Py_INCREF(CogsnetException);
  PyModule_AddObject(module, "CogsnetException", CogsnetException);

  return module;
}
