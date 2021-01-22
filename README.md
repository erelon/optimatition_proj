# Project description


This project deals with segmentation of object in images by converting the images to a flow graph, and running the following algorithms to solve the max flow-min cut problem. 
In the project we will compare the performance of the algorithms under different conditions.

## background
1. **Flow network** is a [directed graph](https://en.wikipedia.org/wiki/Directed_graph "Directed graph") where each edge has a **capacity** and each edge receives a flow. The amount of flow on an edge cannot exceed the capacity of the edge. A flow must satisfy the restriction that the amount of flow into a node equals the amount of flow out of it, unless it is a **source**, which has only outgoing flow, or **sink**, which has only incoming flow.
2. **max-flow min-cut theorem** states that in a [flow network](https://en.wikipedia.org/wiki/Flow_network "Flow network"), the maximum amount of flow passing from the [_source_](https://en.wikipedia.org/wiki/Glossary_of_graph_theory#Direction "Glossary of graph theory") to the [_sink_](https://en.wikipedia.org/wiki/Glossary_of_graph_theory#Direction "Glossary of graph theory") is equal to the total weight of the edges in a [minimum cut](https://en.wikipedia.org/wiki/Minimum_cut "Minimum cut"), i.e. the smallest total weight of the edges which if removed would disconnect the source from the sink.
## From image to Graph
1. Every pixel of the image converted to node in the graph.
2. Add 2 more nodes: the source node and the sink node.
3. Node values are their grayscale values
4.  From every node ,except the source and sink, we add edges to his neighbors (top, down, left, right nodes).
5. Each of those edges recieves a capacity  that describes the change in the values ​​of the vertices that it connects.
We used the following function: 
𝐵(𝑣, 𝑢) = 100 ⋅ e𝑥𝑝 ( −(𝑣 − 𝑢) 2 2𝜎 2 )
6. Add edges between all "chosen object nodes" to the source node with the maximum capacity.
7. Add edges between all "chosen non-object nodes" to the sink node with the maximom capacity.

##Our min cut-max flow algorithms

1. Augmenting path [wiki](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm)
2. Dinic's algorithm [wiki](https://en.wikipedia.org/wiki/Dinic%27s_algorithm)
3. Edmonds–Karp algorithm [wiki](https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm)
4. Karger's algorithm[wiki](https://en.wikipedia.org/wiki/Karger%27s_algorithm)
5. Push–relabel algorithm [wiki](https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm)
6. Boykov kolmogorov [article](https://discovery.ucl.ac.uk/id/eprint/13383/1/13383.pdf)
7. Sim cut [article](https://patentimages.storage.googleapis.com/2b/1e/e9/5834a9cc3312a0/US9214029.pdf)

### Get started 
1. install all dependencies 
