# Parallel Systems
### Projects done as a part of Parallel Systems coursework- NC State (Spring 2022)

#### 1. Distributed Training with Tensorflow and Keras

Parallelised the training process of a Keras model over multiple GPU nodes based on synchronous data parallelism, using the tf.distribute API. Performed multi-worker distributed synchronous training using MultiWorkerMirroredStrategy.

-----------------------------------------------------------------------------

#### 2. Distributed Testing with PyTorch

Implemented model testing using distributed communication in order to support multiple nodes, performed message passing between GPUs using PyTorch.distributed API.

-----------------------------------------------------------------------------

#### 3. TFICF - MPI Implementation

Computed TFICF values for each word in a corpus of documents via an MPI implementation to enable sharing of data of each word to the different worker nodes using non-blocking MPI calls such as MPI_Isend, MPI_Irecv etc. The root rank in the set of processors acts as the master that divides and distributes the documents in the corpus to the worker nodes. The worker nodes communicate with each other to compute the TFICF values of each word.

-----------------------------------------------------------------------------

#### 4. TFICF - Spark Implementation

Computed the TFICF value of each word in the corpus of documents provided using Apache Spark and Java. The program is split into three primary jobs namely TF Job, ICF Job, and TF-ICF Job with each phase having its mapping and reduce phase with transformations such as mapToPair, flatMapToPair, reduceByKey, etc.

-----------------------------------------------------------------------------

#### 5. TFICF - MapReduce Implementation

Implemented the calculation of TFICF for each word in a corpus of documents using MapReduce, where the TFICF calculation has been accomplished via three jobs - first job counts the number of times each legal word appears in each input document, second job counts the total number of words in each document which is the term frequency, the third job counts the number of total documents in each corpus, the number of documents containing each word, and finally calculates the TFICF value for each word. 

-----------------------------------------------------------------------------

#### 6. CUDA Programming 

Computed the integral of a given function (cos() here) by calculating the cumulative area under the curve in a parallel scheme using CUDA programming.

-----------------------------------------------------------------------------

#### 7. MPI Programming - Computing derivatives using MPI

Implemented the computation of the derivative of a function (sinx() in this case) in a parallel manner using MPI, data decomposition and finite difference method. Boundary conditions have been communicated using blocking point-to-point communication and non-blocking point-to-point communication.

-----------------------------------------------------------------------------

#### 8. MPI Programming - Point-to-Point Message Latency between two nodes

The average round-trip time (RTT) as a function of message size is computed and plotted along with the standard deviation (std. dev) as error bars. The program calculates the point-to-point message latency for four different pairs of nodes (pair, 2 pairs, 3 pairs, 4 pairs) for the following message sizes: 32KB, 64KB, 128KB, 256KB, 512KB, 1MB, and 2MB between Process 0 and Process 1.
