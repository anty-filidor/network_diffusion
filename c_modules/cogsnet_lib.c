// Copyright (c) 2025 by Mateusz Nurek, Radosław Michalski, Michał Czuba.
//
// This file is a part of Network Diffusion.
//
// Network Diffusion is licensed under the MIT License. You may obtain a copy
// of the License at https://opensource.org/licenses/MIT

#include "cogsnet_lib.h"

#include <stdio.h>

#include "cogsnet_compute.h"

static PyObject* method__cogsnet(PyObject* self, PyObject* args) {
  const char* forgetting_type;
  int snapshot_interval;
  int edge_lifetime;
  float mu;
  float theta;
  int units;
  const char* path_events;
  const char* delimiter;

  // Parse arguments
  if (!PyArg_ParseTuple(args, "siiffiss", &forgetting_type, &snapshot_interval,
                        &edge_lifetime, &mu, &theta, &units, &path_events,
                        &delimiter)) {
    return NULL;
  }

  // Run cogsnet
  struct Cogsnet network =
      cogsnet(forgetting_type, snapshot_interval, edge_lifetime, mu, theta,
              units, path_events, delimiter);

  // Snaphots of the network
  float*** snapshots = network.snapshots;

  // Create the Python list of lists
  PyObject* result_list = PyList_New(0);
  if (network.exit_status == 0) {
    for (int i = 0; i < network.number_of_snapshots; i++) {
      PyObject* inner_list = PyList_New(0);
      for (int j = 0; j < network.number_of_nodes * network.number_of_nodes;
           j++) {
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

    for (int i = 0; i < network.number_of_snapshots; i++) {
      for (int j = 0; j < network.number_of_nodes * network.number_of_nodes;
           j++) {
        free(network.snapshots[i][j]);
      }
      free(network.snapshots[i]);
    }
    free(network.snapshots);
  } else {
    if (network.snapshots != NULL) {
      for (int i = 0; i < network.number_of_snapshots; i++) {
        for (int j = 0; j < network.number_of_nodes * network.number_of_nodes;
             j++) {
          free(network.snapshots[i][j]);
        }
        free(network.snapshots[i]);
      }
      free(network.snapshots);
    }

    PyErr_Format(CogsnetException, "%s", network.error_msg);
    return NULL;
  }

  return result_list;
}
