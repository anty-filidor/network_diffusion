struct Cogsnet {
  float ***snapshots;
  int number_of_snapshots;
  int number_of_nodes;
  int exit_status;
  char error_msg[1024];
};

struct Cogsnet cogsnet(const char *forgetting_type, int snapshot_interval,
                       float mu, float theta, float lambda, int units,
                       const char *path_events, const char *delimiter);
