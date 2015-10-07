LUA=$(which luajit lua | head -n 1)

if [ ! -x "$LUA" ]
    then
        echo "Neither luajit nor lua found in path"
            exit 1
            fi

echo "Using Lua at:"
echo "$LUA"

$LUA main.lua -netType alexnetowtbn -backend cudnn -nGPU 1 -batchSize 128
$LUA main.lua -netType alexnetowtbn -backend cudnn -nGPU 2 -batchSize 256
$LUA main.lua -netType alexnetowtbn -backend cudnn -nGPU 4 -batchSize 512
$LUA main.lua -netType alexnetowtbn -backend cudnn -nGPU 8 -batchSize 1024

$LUA main.lua -netType vgg -backend cudnn -nGPU 1 -batchSize 64 -epochSize 10
$LUA main.lua -netType vgg -backend cudnn -nGPU 2 -batchSize 128 -epochSize 10
$LUA main.lua -netType vgg -backend cudnn -nGPU 4 -batchSize 256 -epochSize 10
$LUA main.lua -netType vgg -backend cudnn -nGPU 8 -batchSize 512 -epochSize 10

$LUA main.lua -netType googlenet -backend cudnn -nGPU 1 -batchSize 64 -epochSize 10
$LUA main.lua -netType googlenet -backend cudnn -nGPU 2 -batchSize 128 -epochSize 10
$LUA main.lua -netType googlenet -backend cudnn -nGPU 4 -batchSize 256 -epochSize 10
$LUA main.lua -netType googlenet -backend cudnn -nGPU 8 -batchSize 512 -epochSize 10
