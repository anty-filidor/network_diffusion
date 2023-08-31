#include "cogsnet_lib.h"

#include <stdio.h>

#include "cogsnet_compute.h"

static PyObject* method_cogsnet(PyObject* self, PyObject* args) {
  const char* forgettingType;
  int snapshotInterval;
  float mu;
  float theta;
  float lambda_;
  int units;
  const char* pathEvents;
  const char* delimiter;

  // Parse arguments
  if (!PyArg_ParseTuple(args, "sifffiss", &forgettingType, &snapshotInterval,
                        &mu, &theta, &lambda_, &units, &pathEvents,
                        &delimiter)) {
    return NULL;
  }

  // Run cogsnet
  struct Cogsnet network = cogsnet(forgettingType, snapshotInterval, mu, theta,
                                   lambda_, units, pathEvents, delimiter);

  // Snaphots of the network
  float*** snapshots = network.snapshots;

  // Create the Python list of lists
  PyObject* result_list = PyList_New(0);
  if (network.exitStatus == 0) {
    for (int i = 0; i < network.numberOfSnapshots; i++) {
      PyObject* inner_list = PyList_New(0);
      for (int j = 0; j < network.numberOfNodes * network.numberOfNodes; j++) {
        PyObject* uid1 = PyFloat_FromDouble(snapshots[i][j][0]);
        PyObject* uid2 = PyFloat_FromDouble(snapshots[i][j][1]);
        PyObject* weight = PyFloat_FromDouble(snapshots[i][j][2]);

        PyObject* row = PyList_New(0);

        PyList_Append(row, uid1);
        Py_DECREF(uid1);

        PyList_Append(row, uid2);
        Py_DECREF(uid2);

        PyList_Append(row, weight);
        Py_DECREF(weight);

        PyList_Append(inner_list, row);
        Py_DECREF(row);
      }

      PyList_Append(result_list, inner_list);
      Py_DECREF(inner_list);
    }

    for (int i = 0; i < network.numberOfSnapshots; i++) {
      for (int j = 0; j < network.numberOfNodes * network.numberOfNodes; j++) {
        free(network.snapshots[i][j]);
      }
      free(network.snapshots[i]);
    }
    free(network.snapshots);
  } else {
    if (network.snapshots != NULL) {
      for (int i = 0; i < network.numberOfSnapshots; i++) {
        for (int j = 0; j < network.numberOfNodes * network.numberOfNodes;
             j++) {
          free(network.snapshots[i][j]);
        }
        free(network.snapshots[i]);
      }
      free(network.snapshots);
    }

    PyObject* exception =
        PyErr_Format(CogsnetException, "%s", network.errorMsg);
    return NULL;
  }

  return result_list;
}
