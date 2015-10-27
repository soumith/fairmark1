require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'nn'
require 'cunn'
require 'optim'
require 'cudnn'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 Imagenet Benchmark script')
cmd:text()
cmd:text('Options:')
cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
cmd:option('-backend',     'cudnn', 'Options: cudnn | fbcunn | cunn')
------------- Training options --------------------
cmd:option('-epochSize',       20, 'Number of batches per epoch')
cmd:option('-batchSize',       128,   'mini-batch size (1 = pure stochastic)')
---------- Optimization options ----------------------
cmd:option('-netType',     'alexnet', 'Options: alexnet | overfeat')
cmd:option('-useCollectives', false, 'whether to use ring collective operations')
cmd:text()

opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')

local config = opt.netType .. '_' .. opt.backend
paths.dofile('models/' .. config .. '.lua')
nClasses = 1000
model = createModel(opt.nGPU) -- for the model creation code, check the models/ folder
criterion = nn.ClassNLLCriterion()
model:training()
criterion:cuda()
collectgarbage()

local optimState = {
    learningRate = 0.001,
    learningRateDecay = 0.0,
    momentum = 0.9,
    dampening = 0.0,
    weightDecay = 0
}

local inputsCPU = cutorch.createCudaHostTensor(opt.batchSize, 3, 224, 224):normal()
local labelsCPU = torch.FloatTensor(opt.batchSize):fill(1)

local inputs = torch.CudaTensor(opt.batchSize, 3, 224, 224):normal()
local labels = torch.CudaTensor(opt.batchSize):fill(1)

cudnn.benchmark = true

local parameters, gradParameters = model:getParameters()
--warmup
model:forward(inputs)
local tm = torch.Timer()
for i=1,opt.epochSize do
   cutorch.synchronize()
   inputs:copy(inputsCPU)
   labels:copy(labelsCPU)
   local function feval(x)
       local outputs = model:forward(inputs)
       local err = criterion:forward(outputs, labels)
       local df_do = criterion:backward(outputs, labels)
       local gradInput = model:backward(inputs, df_do)
       return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)
   model:apply(function(m) if m.syncParameters then m:syncParameters() end end)
   cutorch.synchronize()
end
cutorch.synchronize()

print(string.format('[BENCHMARK SUMMARY] Time(s): %.2f\tBatchSize: %d \t '
                       .. 'netType: %s\t backend: %s\t nGPUs: %d \t epochSize: %d',
                    tm:time().real, opt.batchSize, opt.netType, opt.backend, opt.nGPU, opt.epochSize))
collectgarbage()
