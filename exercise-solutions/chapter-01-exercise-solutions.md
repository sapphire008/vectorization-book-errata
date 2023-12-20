1. Give the two definitions of "Vectorization" as introduced in this chapter.

* Vectorization is a type of parallel computing paradigm that performs arithmetic operations on an array of numbers within a single data processing unit (either CPU or GPU). 
* The term Vectorization also refers to the process of transforming an algorithm computable on one data point at a time to an operation that calculates a collection of data simultaneously.
* Additional, another definition of the term "vectorization" is related to the more recent advances in large language models: using a vector or an array of values to represent a word.

2. When is vectorization faster than for-loops in Python? 

* All data fits into the memory of the machine that runs the vectorization algorithm.
* The algorithm is implemented by taken advantage of vectorizing operations in either CPU or GPU, such that multiple data points are processed in parallel.

3. When is vectorization slower than for-loops in Python?

Certain operations in Python is already optimized, such as using list comprehension to process strings. More recent releases of Python starts to optimize routines like for-loop out-of-the-box.

4. What are the limitations of vectorization?

* All data needs to fit into the memory to maximize the performance. Hence vectorized operations can be memory intensive sometimes. If not all the data can fit into the memory, multi-worker distributed computing can be used in combination with vectorized processing to gain further speed.
* If an operation is not implemented in the high-level library, then the operation may not be vectorizable without thinking about work-arounds, and sometimes, creating the operation from scratch or making contributions to the library as a feature.


5. Do some research, and find out how CPU and GPU differ in their approach to parallelizing computation. If GPU is always "faster" than CPU, e.g. for deep learning, why do we still need CPU?

To put it simply, GPU is good at performing simple operations in parallel, while CPU is good at performing complex operations in a series. Many computer processes are complex in nature (e.g. managing state of multiple applications, how much resources to allocate, how to move data around between disks and memory) and cannot be exectued in parallel. This is where CPU is more suitable. Other computer processes such as rendering graphics (i.e. calculating the value of pixels to be displayed on the screen) are parallel in nature. This is where GPU is more suitable in managing the compute of these tasks.

6. Define mltithreading.

Multithreading can only process one stream of data at a time, instead of processing data in parallel. It can switch between different threads of the same program (or sometimes different threads of different programs), with each thread representing a different part of the process.


7. Define multiprocessing.

Multiprocessing processes data by operating on multiple processing units, e.g. multi-core CPUs or GPUs. Each core of the processing unit can access data through a shared memory (the RAM). 

8. Define multi-worker distributed computing.

Multi-worker distributed computing process data by connecting multiple machines's processing units, memory, and/or disks together via networking. A driver (or main) node issues a series of commands to create workers and assigns each worker a task to process a portion of the data. Once each worker completes their own task, the worker will send the results back to the driver for aggregation.  The driver then waits for all the workers to finish their tasks before returning the results to the user.

9. How do we decide when to use vectorization vs. multithreading vs. multiprocessing vs. multi-worker computing? Consider scale of the data, and the complexity of the operation.

In general, the size of the data as well as the available resource will be the key factors to decide which parallel computing process to use. Different variants of the processing technique would also require mastery of the logic as well as the libraries implementing these processes.

10. Do some research and list several frameworks for distributed computing in Python that are similar to Apache Beam.

Apache Spark, Apache Flink, Dask, Ray