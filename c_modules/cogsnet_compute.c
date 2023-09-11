// Notes:
// #1 when compiling using GCC, "-lm" compilation option is required
// #2 all the events have to be in chronological order, otherwise CogSNet
// calculations will be wrong #3 the pathEvents must contain the header and an
// empty line at the end #4 output file with CogSNet is not sorted by weights or
// anything else

#define _XOPEN_SOURCE
#define LEN(arr) ((int)(sizeof(arr) / sizeof(arr)[0]))

#include "cogsnet_compute.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
// Windows-specific code
#include <windows.h>
#else
// Unix-like system code
#include <unistd.h>
#endif

// linear forgetting
float compute_weight_linear(int new_event, float weight_last_event,
                            float time_difference, float lambda, float mu) {
  if (new_event == 1) {
    return (mu + (weight_last_event - time_difference * lambda) * (1 - mu));
  } else {
    return (weight_last_event - time_difference * lambda);
  }
}

// power forgetting
float compute_weight_power(int new_event, float weight_last_event,
                           float time_difference, float lambda, float mu) {
  // We need to check whether the time_difference is greater or equal one,
  // since the power to sth smaller than one will result with heigher weight

  if (time_difference >= 1) {
    if (new_event == 1) {
      return (mu + (weight_last_event * pow(time_difference, -1 * lambda) *
                    (1 - mu)));
    } else {
      return (weight_last_event * pow(time_difference, -1 * lambda));
    }
  } else {
    return (weight_last_event);
  }
}

// exponential forgetting
float compute_weight_exponential(int new_event, float weight_last_event,
                                 float time_difference, float lambda,
                                 float mu) {
  if (new_event == 1) {
    return (mu + (weight_last_event * exp(-1 * lambda * time_difference) *
                  (1 - mu)));
  } else {
    return (weight_last_event * exp(-1 * lambda * time_difference));
  }
}

// computing the weight, invoked for every new event and at the end (surveys'
// dates)
float compute_weight(int time_to_compute, int time_last_event,
                     const char *forgetting_type, float weight_last_event,
                     int new_event, float mu, float lambda, float theta,
                     int units) {
  // Compute the time difference between events
  float time_difference =
      ((float)(time_to_compute - time_last_event) / (float)units);

  if (time_difference >= 0) {
    // First, we need to know how much time passed since last event
    // The time difference has to be zero or a positive value

    // Our new weigth of the edge
    float weight_new;

    if (strncmp(forgetting_type, "linear", 6) == 0) {
      weight_new = compute_weight_linear(new_event, weight_last_event,
                                         time_difference, lambda, mu);
    } else if (strncmp(forgetting_type, "power", 5) == 0) {
      weight_new = compute_weight_power(new_event, weight_last_event,
                                        time_difference, lambda, mu);
    } else if (strncmp(forgetting_type, "exponential", 11) == 0) {
      weight_new = compute_weight_exponential(new_event, weight_last_event,
                                              time_difference, lambda, mu);
    } else {
      weight_new = -1;
    }

    if (weight_new <= theta)
      // Is the weight lower or equal the threshold?
      // If so, it will be zeroed
      return (0);
    else {
      // This is the typical case, return the new weight
      return (weight_new);
    }
  } else {
    // Time difference was less than zero
    return (-1);
  }
}

void create_snapshot(int number_of_nodes, int snapshot_counter,
                     int snapshot_time, const char *forgetting_type, float mu,
                     float theta, float lambda, int units, int **recent_events,
                     float **current_weights, float ***snapshots,
                     int *real_node_ids) {
  for (int i = 0; i < number_of_nodes; i++) {
    for (int j = 0; j < number_of_nodes; j++) {
      float edge_weight =
          compute_weight(snapshot_time, recent_events[i][j], forgetting_type,
                         current_weights[i][j], 0, mu, lambda, theta, units);
      snapshots[snapshot_counter][i * number_of_nodes + j][0] =
          real_node_ids[i];
      snapshots[snapshot_counter][i * number_of_nodes + j][1] =
          real_node_ids[j];
      snapshots[snapshot_counter][i * number_of_nodes + j][2] = edge_weight;
    }
  }
}

// the main function responsible for computing CogSNet
struct Cogsnet compute_cogsnet(int number_of_nodes, int *real_node_ids,
                               int number_of_events, int **events,
                               int snapshot_interval, float mu, float theta,
                               float lambda, const char *forgetting_type,
                               int units) {
  struct Cogsnet network;

  // we declare an array for storing last events between nodes
  int **recent_events = (int **)malloc(number_of_nodes * sizeof(int *));
  for (int i = 0; i < number_of_nodes; i++) {
    recent_events[i] = (int *)malloc(number_of_nodes * sizeof(int));
  }

  // we declare an array for storing weights between nodes
  float **current_weights = (float **)malloc(number_of_nodes * sizeof(float *));
  for (int i = 0; i < number_of_nodes; i++) {
    current_weights[i] = (float *)malloc(number_of_nodes * sizeof(float));
  }

  int number_of_snapshots = 0;
  if (snapshot_interval != 0) {
    number_of_snapshots =
        (events[number_of_events - 1][2] - events[0][2]) / snapshot_interval +
        1;
  } else {
    number_of_snapshots = number_of_events + 1;
  }

  // Time of the next snapshot of the network.
  // The first snapshot will be taken relative to the time of the first event in
  // the dataset.
  int snapshot_time = events[0][2] + snapshot_interval;

  // Snapshot counter
  int snapshot_counter = 0;

  // The array for storing a complete snapshot of the network.
  // It is saved to a file using the save_cogsnet function.
  // Due to the potentially large size of the array, we allocate memory using
  // malloc on the heap instead of the stack.
  float ***snapshots =
      (float ***)malloc(number_of_snapshots * sizeof(float **));
  for (int i = 0; i < number_of_snapshots; i++) {
    snapshots[i] =
        (float **)malloc(number_of_nodes * number_of_nodes * sizeof(float *));
    for (int j = 0; j < number_of_nodes * number_of_nodes; j++) {
      snapshots[i][j] = (float *)malloc(3 * sizeof(float));
    }
  }

  // zeroing arrays
  for (int i = 0; i < number_of_snapshots; i++) {
    for (int j = 0; j < number_of_nodes * number_of_nodes; j++) {
      for (int k = 0; k < 3; k++) {
        snapshots[i][j][k] = 0;
      }
    }
  }

  for (int i = 0; i < number_of_nodes; i++) {
    for (int j = 0; j < number_of_nodes; j++) {
      recent_events[i][j] = 0;
      current_weights[i][j] = 0;
    }
  }

  if (snapshot_interval == 0 ||
      ((events[number_of_events - 1][2] - events[0][2]) / snapshot_interval) <
          number_of_events) {
    // events have to be chronorogically ordered
    for (int i = 0; i < number_of_events; i++) {
      // new weight will be stored here
      double new_weight = 0;

      int uid1 = events[i][0];
      int uid2 = events[i][1];

      // was there any event with these uid1 and uid2 before?
      // we check it by looking at weights array, since
      // meanwhile the weight could have dropped below theta

      if (current_weights[uid1][uid2] == 0) {
        // no events before, we set the weight to mu
        new_weight = mu;
      } else {
        // there was an event before
        new_weight = compute_weight(
            events[i][2], recent_events[uid1][uid2], forgetting_type,
            current_weights[uid1][uid2], 1, mu, lambda, theta, units);
      }

      // set the new last event time
      // the edges are undirected, so we perform updates for both directions.
      recent_events[uid1][uid2] = events[i][2];
      recent_events[uid2][uid1] = events[i][2];

      // set the new weight
      current_weights[uid1][uid2] = new_weight;
      current_weights[uid2][uid1] = new_weight;

      if (snapshot_interval != 0) {
        // Take a snapshot after a specified interval has elapsed.
        // - ((i+1) < number_of_events) - check if there is a next event
        // - (snapshot_time < events[i+1][2]) - take a snapshot if the time of
        // the next event is greater than the time of the next snapshot
        // - if the time of the snapshot and the next event is the same, the
        // snapshot will be taken after processing that event in the next
        // iteration of the for loop
        // - if the time between events is very distant, the while loop may take
        // multiple snapshots.
        while (((i + 1) < number_of_events) &&
               (snapshot_time < events[i + 1][2])) {
          create_snapshot(number_of_nodes, snapshot_counter, snapshot_time,
                          forgetting_type, mu, theta, lambda, units,
                          recent_events, current_weights, snapshots,
                          real_node_ids);
          snapshot_counter++;
          snapshot_time += snapshot_interval;
        }
      } else {
        // Take a snapshot after each event
        while (((i + 1) < number_of_events) &&
               (snapshot_time < events[i + 1][2])) {
          create_snapshot(number_of_nodes, snapshot_counter, snapshot_time,
                          forgetting_type, mu, theta, lambda, units,
                          recent_events, current_weights, snapshots,
                          real_node_ids);
          snapshot_counter++;
          snapshot_time = events[i + 1][2];
        }
      }
    }

    //  As all events are processed, we take the final snapshot of the network.
    create_snapshot(number_of_nodes, snapshot_counter, snapshot_time,
                    forgetting_type, mu, theta, lambda, units, recent_events,
                    current_weights, snapshots, real_node_ids);
    snapshot_counter++;

    network.snapshots = snapshots;
    network.number_of_snapshots = snapshot_counter;
    network.number_of_nodes = number_of_nodes;
    network.exit_status = 0;
  } else {
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] Number of snapshots cannot be bigger than number of "
             "events! Increase snapshot interval.\n");
    network.exit_status = 1;
  }

  // Free the allocated memory
  for (int i = 0; i < number_of_nodes; i++) {
    free(recent_events[i]);
    free(current_weights[i]);
  }
  free(recent_events);
  free(current_weights);

  return network;
}

// checks whether a given element exists in an array
int existing_id(int x, int *array, int size) {
  int is_found = -1;

  int i = 0;

  while (is_found < 0 && i < size) {
    if (array[i] == x) {
      is_found = i;
      break;
    }

    i++;
  }
  return is_found;
}

// this function returns the element in the CSV organized as three-column one
// (x;y;timestamp) it is used to extract elements both from pathSurveyDates as
// from pathSurveyDates

int return_element_from_csv(char event_line[65536], int element_number,
                            const char *delimiter) {
  char *ptr;

  ptr = strtok(event_line, delimiter);

  int this_line_element_number = 0;

  while (ptr != NULL) {
    if (this_line_element_number == element_number) {
      return atoi(ptr);
    }

    this_line_element_number++;

    ptr = strtok(NULL, delimiter);
  }
}

struct Cogsnet cogsnet(const char *forgetting_type, int snapshot_interval,
                       int edge_lifetime, float mu, float theta, int units,
                       const char *path_events, const char *delimiter) {
  char buffer[65536];
  char *line;
  char line_copy[65536];

  struct Cogsnet network;
  network.snapshots = NULL;

  // validate parameters
  if (strcmp(forgetting_type, "exponential") != 0 &&
      strcmp(forgetting_type, "power") != 0 &&
      strcmp(forgetting_type, "linear") != 0) {
    network.exit_status = 1;
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] Invalid forgetting_type: %s. Allowed values are "
             "'exponential', 'power', or 'linear'.\n",
             forgetting_type);
    return network;
  }

  if (snapshot_interval < 0) {
    network.exit_status = 1;
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] snapshot_interval (%d) cannot be less than 0.\n",
             snapshot_interval);
    return network;
  }

  if (edge_lifetime <= 0) {
    network.exit_status = 1;
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] edge_lifetime (%d) has to be greater than 0.\n",
             edge_lifetime);
    return network;
  }

  if (mu <= 0 || mu > 1) {
    network.exit_status = 1;
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] mu (%f) has to be greater than 0 and less than or equal "
             "to 1.\n",
             mu);
    return network;
  }

  if (theta < 0 || theta >= mu) {
    network.exit_status = 1;
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] theta (%f) has to be between 0 and mu (%f).\n", theta,
             mu);
    return network;
  }

  if (units != 1 && units != 60 && units != 3600) {
    network.exit_status = 1;
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] Invalid units: %d. Allowed values are 1 (seconds), 60 "
             "(minutes), or 3600 (hours).\n",
             units);
    return network;
  }

#ifdef _WIN32
  // Windows-specific code
  if (GetFileAttributes(path_events) == INVALID_FILE_ATTRIBUTES) {
    network.exit_status = 1;
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] File does not exist: %s.\n", path_events);
    return network;
  }
#else
  // Unix-like system code
  if (access(path_events, F_OK) != 0) {
    network.exit_status = 1;
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] File does not exist: %s.\n", path_events);
    return network;
  }
#endif

  if (!(strcmp(delimiter, ",") == 0 || strcmp(delimiter, ";") == 0 ||
        strcmp(delimiter, "\t") == 0)) {
    network.exit_status = 1;
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] Invalid delimiter: %s. Allowed delimiters are ',', ';', "
             "or '\\t'.\n",
             delimiter);
    return network;
  }

  // The variable snapshot_interval and edge_lifetime are usually expressed in
  // hours or minutes. The variable units scales the snapshot_interval and
  // edge_lifetime to seconds.
  snapshot_interval = snapshot_interval * units;
  edge_lifetime = edge_lifetime * units;

  // compute lambda
  float lambda;
  if (strncmp(forgetting_type, "exponential", 11) == 0) {
    lambda = (1.0 / edge_lifetime) * log(mu / theta);
  } else if (strncmp(forgetting_type, "power", 5) == 0) {
    lambda = log(mu / theta) * log(edge_lifetime);
    ;
  } else {
    // linear
    lambda = (1.0 / edge_lifetime) * (mu - theta);
  }

  // start reading events

  // define the stream for events
  FILE *file_pointer;
  file_pointer = fopen(path_events, "r");

  int number_of_lines = 0;

  // check if there is no other problem with the file stream
  if (file_pointer != NULL) {
    // firstly, we check how many lines we do have to read from the file

    // read the header
    line = fgets(buffer, sizeof(buffer), file_pointer);

    // we read one line (header)
    number_of_lines++;

    // now read the rest of the file until the condition won't be met
    while ((line = fgets(buffer, sizeof(buffer), file_pointer)) != NULL) {
      number_of_lines++;
    }

    fclose(file_pointer);

    // if the path_events has more then one line (header always should be
    // there)
    if (number_of_lines > 1) {
      // define an array for events
      // sender, receiver, timestamp
      int number_of_events = number_of_lines - 1;
      int **events = (int **)malloc(number_of_events * sizeof(int *));
      for (int i = 0; i < number_of_events; i++) {
        events[i] = (int *)malloc(3 * sizeof(int));
      }

      // now, reopen the path_events with events as the stream
      file_pointer = fopen(path_events, "r");

      // skip first line, as it is a header
      line = fgets(buffer, sizeof(buffer), file_pointer);

      // read all further lines and put them into events' matrix
      int events_node_id_sender = 0;
      int events_node_id_receiver = 0;
      int events_timestamp = 0;

      for (int event_number = 0; event_number < number_of_events;
           event_number++) {
        // read the line
        line = fgets(buffer, sizeof(buffer), file_pointer);

        // extract the first element (sender's nodeID)
        strcpy(line_copy, line);
        events_node_id_sender =
            return_element_from_csv(line_copy, 0, delimiter);

        // extract the second element (receiver's nodeID)
        strcpy(line_copy, line);
        events_node_id_receiver =
            return_element_from_csv(line_copy, 1, delimiter);

        // extract the second element (event timestamp)
        strcpy(line_copy, line);
        events_timestamp = return_element_from_csv(line_copy, 2, delimiter);

        // set the proper values of array
        events[event_number][0] = events_node_id_sender;
        events[event_number][1] = events_node_id_receiver;
        events[event_number][2] = events_timestamp;
      }

      fclose(file_pointer);

      // the array holding real nodeIDs
      int current_size = 1;
      int *real_node_ids = (int *)malloc(current_size * sizeof(int));

      // actual number of nodes in the events files
      int number_of_nodes = 0;

      // ----- convert node ids -----
      for (int i = 0; i < number_of_events; i++) {
        for (int j = 0; j < 2; j++) {
          int real_node_id = events[i][j];
          int converted_node_id =
              existing_id(real_node_id, real_node_ids, number_of_nodes);

          // do we already have this nodeID in real_node_ids?
          if (converted_node_id < 0) {
            if (current_size < (number_of_nodes + 1)) {
              current_size *= 2;  // double the size
              real_node_ids =
                  (int *)realloc(real_node_ids, current_size * sizeof(int));
            }
            // no, we need to add it
            real_node_ids[number_of_nodes] = real_node_id;
            events[i][j] = number_of_nodes;

            number_of_nodes++;
          } else {
            events[i][j] = converted_node_id;
          }
        }
      }
      network = compute_cogsnet(number_of_nodes, real_node_ids,
                                number_of_events, events, snapshot_interval, mu,
                                theta, lambda, forgetting_type, units);

      // free the allocated memory
      for (int i = 0; i < number_of_events; i++) {
        free(events[i]);
      }
      free(events);
      free(real_node_ids);

      return network;
    } else {
      snprintf(network.error_msg, sizeof(network.error_msg),
               "[ERROR] Reading events from %s: no events to read\n",
               path_events);
    }
  } else {
    snprintf(network.error_msg, sizeof(network.error_msg),
             "[ERROR] Reading events from %s: error reading from filestream\n",
             path_events);
  }

  // network.snapshots = snapshots;
  network.exit_status = 1;
  return network;
}
