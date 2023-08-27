struct Cogsnet {
    float ***snapshots;
    int numberOfSnapshots;
    int numberOfNodes;
    int exitStatus;
    char errorMsg[1024];
};

struct Cogsnet cogsnet(const char *forgettingType, int snapshotInterval, float mu, float theta, float lambda, int units, const char *pathEvents, const char *delimiter);