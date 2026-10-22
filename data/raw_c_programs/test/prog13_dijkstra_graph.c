/* Program 13: Dijkstra Shortest Path on Graph Matrix (Test Program) */
#define INF 999999
#define VERTICES 6

int min_distance(const int dist[], const int visited[]) {
    int min_val = INF;
    int min_index = -1;
    for (int v = 0; v < VERTICES; v++) {
        if (!visited[v] && dist[v] <= min_val) {
            min_val = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

void dijkstra(int graph[VERTICES][VERTICES], int src, int dist[VERTICES]) {
    int visited[VERTICES];
    for (int i = 0; i < VERTICES; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[src] = 0;
    for (int count = 0; count < VERTICES - 1; count++) {
        int u = min_distance(dist, visited);
        if (u == -1) break;
        visited[u] = 1;
        for (int v = 0; v < VERTICES; v++) {
            if (!visited[v] && graph[u][v] && dist[u] != INF &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
}
