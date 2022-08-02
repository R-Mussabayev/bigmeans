# bigmeans
Big-Means: A Simple and Effective Algorithm for Big Data Minimum Sum-of-Squares Clustering

Original preprint paper:
Rustam Mussabayev, Nenad Mladenovic, Ravil Mussabayev, Bassem Jarboui. Big-means: Less is More for K-means Clustering. arXiv preprint arXiv:2204.07485. 14 Apr 2022. pp. 1-40
https://arxiv.org/pdf/2204.07485.pdf

K-means clustering plays a vital role in data mining. However, its performance drastically drops when applied to huge amounts of data. We propose a new heuristic that is built on the basis of regular K-means for faster and more accurate big data clustering using the "less is more" and decomposition approaches. The main advantage of the proposed algorithm is that it naturally turns the K-means local search into global one through the process of decomposition of the minimum sum-of-squares clustering (MSSC) problem. On one hand, decomposition of the MSSC problem into smaller subproblems reduces the computational complexity and allows for their parallel processing. On the other hand, the MSSC decomposition provides a new method for the natural data-driven shaking of the incumbent solution while introducing a new neighborhood structure for the solution of the MSSC problem. The proposed algorithm is scalable, fast, and accurate. The scalability of the algorithm can be easily adjusted by choosing the appropriate number of subproblems and their size. In our experiments it outperforms all recent state-of-the-art algorithms for the MSSC in both in time and the solution quality.
