// Copyright (c) 2023 by Mateusz Nurek, Radosław Michalski, Michał Czuba.
//
// This file is a part of Network Diffusion.
//
// Network Diffusion is licensed under the MIT License. You may obtain a copy
// of the License at https://opensource.org/licenses/MIT

struct Cogsnet {
  float ***snapshots;
  int number_of_snapshots;
  int number_of_nodes;
  int exit_status;
  char error_msg[1024];
};

struct Cogsnet cogsnet(const char *forgetting_type, int snapshot_interval,
                       int edge_lifetime, float mu, float theta, int units,
                       const char *path_events, const char *delimiter);
