th main.lua -netType alexnet -backend cudnn -nGPU 1
th main.lua -netType alexnet -backend cudnn -nGPU 2
#th main.lua -netType alexnet -backend fbcunn -nGPU 1
#th main.lua -netType alexnet -backend fbcunn -nGPU 2
th main.lua -netType alexnet -backend cunn -nGPU 1
th main.lua -netType alexnet -backend cunn -nGPU 2

th main.lua -netType alexnetowt -backend cudnn -nGPU 1
th main.lua -netType alexnetowt -backend cudnn -nGPU 2
th main.lua -netType alexnetowt -backend cudnn -nGPU 4
#th main.lua -netType alexnetowt -backend cudnn -nGPU 8

th main.lua -netType alexnetowtbn -backend cudnn -nGPU 1
th main.lua -netType alexnetowtbn -backend cudnn -nGPU 2
th main.lua -netType alexnetowtbn -backend cudnn -nGPU 4
#th main.lua -netType alexnetowtbn -backend cudnn -nGPU 8

th main.lua -netType overfeat -backend cudnn -nGPU 1
th main.lua -netType overfeat -backend cudnn -nGPU 2
th main.lua -netType overfeat -backend cudnn -nGPU 4
#th main.lua -netType overfeat -backend cudnn -nGPU 8

th main.lua -netType vgg -backend cudnn -nGPU 1
th main.lua -netType vgg -backend cudnn -nGPU 2
th main.lua -netType vgg -backend cudnn -nGPU 4
#th main.lua -netType vgg -backend cudnn -nGPU 8

th main.lua -netType googlenet -backend cudnn -nGPU 1
th main.lua -netType googlenet -backend cudnn -nGPU 2
th main.lua -netType googlenet -backend cudnn -nGPU 4
#th main.lua -netType googlenet -backend cudnn -nGPU 8
