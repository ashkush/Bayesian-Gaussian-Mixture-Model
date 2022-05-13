# Distributed Version of Scikit learn Implementation of Bayesian Gaussian Mixture Model


This is the distributed implementation of Scikit learn [Bayesian Gaussian Mixture Model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html), using Apache Spark which basically utilizes advantage of parallel processing power of clusters (Distributed Computing) as compared to its single node version implementation.


The base code for all the calculation of the internal calculation of the algorithm has been taken from Scikit-Learn implementation which has been distributed using Apache Spark's Python API and uses RDDs (Resilient Distributed Datasets) for performing fast and effecient parallel computation required for the algorithm. It uses MapPartition and reduce functionality of Apache Spark for distributing the work load among the various worker nodes within the cluster on which the algorithm will be used.


