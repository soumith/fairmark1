There are three set of experiments that are run:
- alexnetowtbn
- vgg
- googlenet

They are run from 1-GPU to 8-GPU configurations.

The output is in this format:

[BENCHMARK SUMMARY] Time(s): 11.14      BatchSize: 128   netType: alexnetowtbn   backend: cudnn  nGPUs: 1        epochSize: 20

The "Time(s)" is the time to run the benchmark in seconds. The batchSize has been configured such that it is easy to read the benchmark.
With the best possible system, you will see the SAME time regardless of nGPUs, i.e. nGPUs = 1 should have the same time as nGPUs = 8. This will be the ideal condition and is linear-scaling.
Practically, you will see sub-linear scaling because of communcation drops.

Example:
[BENCHMARK SUMMARY] Time(s): 11.14      BatchSize: 128   netType: alexnetowtbn   backend: cudnn  nGPUs: 1        epochSize: 20
[BENCHMARK SUMMARY] Time(s): 12.12      BatchSize: 256   netType: alexnetowtbn   backend: cudnn  nGPUs: 2        epochSize: 20
[BENCHMARK SUMMARY] Time(s): 14.41      BatchSize: 512   netType: alexnetowtbn   backend: cudnn  nGPUs: 4        epochSize: 20

As you see, 4-GPU has a slight overhead of 3 extra seconds lost in various communication.