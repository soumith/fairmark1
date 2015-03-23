CUTH="/mnt/vol/contbuild/contbuild/fbcode/deeplearning_cuth/_bin/deeplearning/torch/cuth.lar"
$CUTH main.lua -netType alexnet -backend cudnn -nGPU 1
$CUTH main.lua -netType alexnet -backend cudnn -nGPU 2
#$CUTH main.lua -netType alexnet -backend fbcunn -nGPU 1
#$CUTH main.lua -netType alexnet -backend fbcunn -nGPU 2
$CUTH main.lua -netType alexnet -backend cunn -nGPU 1
$CUTH main.lua -netType alexnet -backend cunn -nGPU 2

$CUTH main.lua -netType alexnetowt -backend cudnn -nGPU 1
$CUTH main.lua -netType alexnetowt -backend cudnn -nGPU 2
$CUTH main.lua -netType alexnetowt -backend cudnn -nGPU 4
#$CUTH main.lua -netType alexnetowt -backend cudnn -nGPU 8

$CUTH main.lua -netType alexnetowtbn -backend cudnn -nGPU 1
$CUTH main.lua -netType alexnetowtbn -backend cudnn -nGPU 2
$CUTH main.lua -netType alexnetowtbn -backend cudnn -nGPU 4
#$CUTH main.lua -netType alexnetowtbn -backend cudnn -nGPU 8

$CUTH main.lua -netType overfeat -backend cudnn -nGPU 1
$CUTH main.lua -netType overfeat -backend cudnn -nGPU 2
$CUTH main.lua -netType overfeat -backend cudnn -nGPU 4
#$CUTH main.lua -netType overfeat -backend cudnn -nGPU 8

$CUTH main.lua -netType vgg -backend cudnn -nGPU 1
$CUTH main.lua -netType vgg -backend cudnn -nGPU 2
$CUTH main.lua -netType vgg -backend cudnn -nGPU 4
#$CUTH main.lua -netType vgg -backend cudnn -nGPU 8

$CUTH main.lua -netType googlenet -backend cudnn -nGPU 1
$CUTH main.lua -netType googlenet -backend cudnn -nGPU 2
$CUTH main.lua -netType googlenet -backend cudnn -nGPU 4
#$CUTH main.lua -netType googlenet -backend cudnn -nGPU 8
