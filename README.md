# Project description


This project deals with segmentation of an object in images by converting the images into a flow graph and running the following algorithms to solve the max flow-min cut problem. 
In the project, we will compare the performance of the algorithms under different conditions.

## background
1. **Flow network** is a [directed graph](https://en.wikipedia.org/wiki/Directed_graph "Directed graph") where each edge has a **capacity** and each edge receives a flow. The amount of flow on an edge cannot exceed the capacity of the edge. A flow must satisfy the restriction that the amount of flow into a node equals the amount of flow out of it unless it is a **source**, which has only outgoing flow, or **sink**, which has only incoming flow.
2. **max-flow min-cut theorem** states that in a [flow network](https://en.wikipedia.org/wiki/Flow_network "Flow network"), the maximum amount of flow passing from the [_source_](https://en.wikipedia.org/wiki/Glossary_of_graph_theory#Direction "Glossary of graph theory") to the [_sink_](https://en.wikipedia.org/wiki/Glossary_of_graph_theory#Direction "Glossary of graph theory") is equal to the total weight of the edges in a [minimum cut](https://en.wikipedia.org/wiki/Minimum_cut "Minimum cut"), i.e. the smallest total weight of the edges which if removed would disconnect the source from the sink.
## From image to Graph
1. Every pixel of the image converted to a node in the graph.
2. Add 2 more nodes: the source node and the sink node.
3. Node values are their grayscale values
4.  From every node, except the source and sink, we add edges to his neighbours (top, down, left, right nodes).
5. Each of those edges receives a capacity that describes the change in the values ​​of the vertices that it connects.
We used the following function:  <img src="https://latex.codecogs.com/gif.latex?\inline&space;B(v,u)=100*exp(\frac{-(v-u)^2}{2\sigma^2})" title="B(v,u)=100*exp(\frac{-(v-u)^2}{2\sigma^2})" />
6. Add edges between all "chosen object nodes" to the source node with the maximum capacity.
7. Add edges between all "chosen non-object nodes" to the sink node with the maximum capacity.

##Our min cut-max flow algorithms

1. Augmenting path [wiki](https://en.wikipedia.org/wiki/Ford%E2%80%93Fulkerson_algorithm)
2. Dinic's algorithm [wiki](https://en.wikipedia.org/wiki/Dinic%27s_algorithm)
3. Edmonds–Karp algorithm [wiki](https://en.wikipedia.org/wiki/Edmonds%E2%80%93Karp_algorithm)
4. Karger's algorithm[wiki](https://en.wikipedia.org/wiki/Karger%27s_algorithm)
5. Push–relabel algorithm [wiki](https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm)
6. Boykov kolmogorov [article](https://discovery.ucl.ac.uk/id/eprint/13383/1/13383.pdf)
7. Sim cut [wiki](https://en.wikibooks.org/wiki/Algorithm_Implementation/Graphs/Maximum_flow/Sim_Cut)

### Get started 
1. Download the environment.yml
2. Activate the environment: ```conda env create -f environment.yml``` The name of the environment is "opt".
3. Now you run the code in the conda environment.
4. If you want to use the better implemented Sim cut: Download and install [swig](http://www.swig.org/download.html) program, and add this to the path variables: ```path\of\installed\swig\swigwin-4.0.2\```

### Usage
If you want to see an example of this work just run ```python imagesegmentation.py``` in the conda environment. It will use Boykov Kolmogorov algorithm and show you 4 segmented cats.
Otherwise, use your own images: ```python imagesegmentation.py image1.png imag2.png```. A window will open with the image, click on the background area then close it, Then the same window will open again, click on the object and close it. After the calculation will finish- your segmented image will be saved in its own directory.
The options are:
-  ```-a``` : choose the algorithm to use for the segmentation. you can choose more than one. the options are:
    - all : Run all algorithms
    - bk : Boykov kolmogorov. This is the default.
    - sc : Sim cut. The default amount of iterations is 20. If you want to change it add: ```-i (number of wanted iterations)```
    - pp : Push–relabel.
    - sap : Augmenting path. They are 2 versions of this algorithm. If you want to see them both, add this twice. ```-a sap sap```
    - d : Dinitz.
    - ek: Edmonds–Karp.
-  ```--sigma``` : Change the sigma. The default is 30.
-  ```-s``` : Size to resize to. The default is 30, so the output will be 30X30. For maximum type ```-s None```. It will cap at about 300X300.

For example for the images "dog1.png" and "dog2.png", this line:
```
python imagesegmentation.py path/to/dog1.png path/to/dog1.png -a bk sc pp -i 10 --sigma 25 -s None
```
The result will be 3 segmentations for both of the images that will be saved in their own directory. The segmentations will be created with sigma=25 in the maximum resolution, and the Sim cut algorithm will use 10 iterations.
