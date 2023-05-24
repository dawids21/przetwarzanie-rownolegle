import sys
import random

def generate_graph(n, p):
    graph = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if random.random() <= p:
                graph[i][j] = 1
    return graph

def main():
    n = int(sys.argv[1])
    p = float(sys.argv[2])
    graph = generate_graph(n, p)
    with open('graph.txt', 'w') as f:
        for row in graph:
            f.write(' '.join(map(str, row)) + '\n')

if __name__ == '__main__':
    main()