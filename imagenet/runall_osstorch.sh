luajit main.lua -netType alexnet -backend cudnn -nGPU 1
luajit main.lua -netType alexnet -backend cudnn -nGPU 2
#luajit main.lua -netType alexnet -backend fbcunn -nGPU 1
#luajit main.lua -netType alexnet -backend fbcunn -nGPU 2
#luajit main.lua -netType alexnet -backend cunn -nGPU 1
#luajit main.lua -netType alexnet -backend cunn -nGPU 2

luajit main.lua -netType alexnetowt -backend cudnn -nGPU 1
luajit main.lua -netType alexnetowt -backend cudnn -nGPU 2
luajit main.lua -netType alexnetowt -backend cudnn -nGPU 4
luajit main.lua -netType alexnetowt -backend cudnn -nGPU 8

luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 1
luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 2
luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 4
luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 8
luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 4 -batchSize 256
luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 8 -batchSize 256
luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 4 -batchSize 512
luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 8 -batchSize 512
luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 4 -batchSize 1024
luajit main.lua -netType alexnetowtbn -backend cudnn -nGPU 8 -batchSize 1024

luajit main.lua -netType overfeat -backend cudnn -nGPU 1
luajit main.lua -netType overfeat -backend cudnn -nGPU 2
luajit main.lua -netType overfeat -backend cudnn -nGPU 4
luajit main.lua -netType overfeat -backend cudnn -nGPU 8
luajit main.lua -netType overfeat -backend cudnn -nGPU 4 -batchSize 256
luajit main.lua -netType overfeat -backend cudnn -nGPU 8 -batchSize 256
luajit main.lua -netType overfeat -backend cudnn -nGPU 4 -batchSize 512
luajit main.lua -netType overfeat -backend cudnn -nGPU 8 -batchSize 512


luajit main.lua -netType vgg -backend cudnn -nGPU 1
luajit main.lua -netType vgg -backend cudnn -nGPU 2
luajit main.lua -netType vgg -backend cudnn -nGPU 4
luajit main.lua -netType vgg -backend cudnn -nGPU 8
luajit main.lua -netType vgg -backend cudnn -nGPU 4 -batchSize 256
luajit main.lua -netType vgg -backend cudnn -nGPU 8 -batchSize 256
luajit main.lua -netType vgg -backend cudnn -nGPU 4 -batchSize 512
luajit main.lua -netType vgg -backend cudnn -nGPU 8 -batchSize 512

luajit main.lua -netType googlenet -backend cudnn -nGPU 1
luajit main.lua -netType googlenet -backend cudnn -nGPU 2
luajit main.lua -netType googlenet -backend cudnn -nGPU 4
luajit main.lua -netType googlenet -backend cudnn -nGPU 8
luajit main.lua -netType googlenet -backend cudnn -nGPU 4 -batchSize 256
luajit main.lua -netType googlenet -backend cudnn -nGPU 8 -batchSize 256
luajit main.lua -netType googlenet -backend cudnn -nGPU 4 -batchSize 512
luajit main.lua -netType googlenet -backend cudnn -nGPU 8 -batchSize 512
