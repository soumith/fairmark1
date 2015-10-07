require 'cudnn'
require 'nnx'

local Convolution = cudnn.SpatialConvolution
local Max = cudnn.SpatialMaxPooling
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU

local function inception(input_size, config, stride)
   stride = stride or 1
   local concat = nn.Concat(2)
   if config[1][1] ~= 0 then -- 1x1 correlation learner, is not present in grid-reduction
      local conv1 = nn.Sequential()
      conv1:add(Convolution(input_size, config[1][1], 1, 1, 1, 1))
      conv1:add(ReLU(true))
      concat:add(conv1)
   end

   local conv3 = nn.Sequential()  -- 3x3 correlation learner. Stride 2 in grid reduction
   conv3:add(Convolution(input_size, config[2][1], 1, 1, 1, 1))
   conv3:add(ReLU(true))
   conv3:add(Convolution(config[2][1], config[2][2], 3, 3, stride, stride, 1, 1))
   conv3:add(ReLU(true))
   concat:add(conv3)

   local conv3xx = nn.Sequential() -- (approx) 5x5 correlation learner. Stride 2 in grid reduction
   conv3xx:add(Convolution(  input_size, config[3][1], 1, 1, 1, 1))
   conv3xx:add(ReLU(true))
   conv3xx:add(Convolution(config[3][1], config[3][2], 3, 3, 1, 1, 1, 1))
   conv3xx:add(ReLU(true))
   conv3xx:add(Convolution(config[3][2], config[3][2], 3, 3, stride, stride, 1, 1))
   conv3xx:add(ReLU(true))
   concat:add(conv3xx)

   local pool = nn.Sequential() -- pooler, stride 2 in grid reduction
   if config[4][1] == 'max' then
      pool:add(Max(3, 3, stride, stride, 1, 1))
   elseif config[4][1] == 'avg' then
      pool:add(Avg(3, 3, stride, stride, 1, 1))
   else
      error('Unknown pooling')
   end
   if config[4][2] ~= 0 then
      pool:add(Convolution(input_size, config[4][2], 1, 1, 1, 1))
      pool:add(ReLU(true))
   end
   concat:add(pool)

   return concat
end

function createModel(nGPU)
   local features = nn.Sequential()
   local mainpipe = nn.Sequential()

   features:add(Convolution(3, 64, 7, 7, 2, 2, 3, 3))
   features:add(ReLU(true))
   features:add(Max(3, 3, 2, 2))
   features:add(Convolution(64, 64, 1, 1))
   features:add(ReLU(true))
   features:add(Convolution(64, 192, 3, 3, 1, 1, 1, 1))
   features:add(ReLU(true))
   features:add(Max(3, 3, 2, 2))
   features:add(inception( 192, {{ 64}, { 64,  64}, { 64,  96}, {'avg',  32}}))    -- 3(a)
   features:add(inception( 256, {{ 64}, { 64,  96}, { 64,  96}, {'avg',  64}}))    -- 3(b)
   features:add(inception( 320, {{  0}, {128, 160}, { 64,  96}, {'max',   0}}, 2)) -- 3(c), grid reduction
   features:add(inception( 576, {{224}, { 64,  96}, { 96, 128}, {'avg', 128}}))    -- 4(a)
   features:add(inception( 576, {{192}, { 96, 128}, { 96, 128}, {'avg', 128}}))    -- 4(b)
   features:add(inception( 576, {{160}, {128, 160}, {128, 160}, {'avg',  96}}))    -- 4(c)
   features:add(inception( 576, {{ 96}, {128, 192}, {160, 192}, {'avg',  96}}))    -- 4(d)
   mainpipe:add(inception( 576, {{  0}, {128, 192}, {192, 256}, {'max',   0}}, 2)) -- 4(e), grid reduction
   mainpipe:add(inception(1024, {{352}, {192, 320}, {160, 224}, {'avg', 128}}))    -- 5(a)
   mainpipe:add(inception(1024, {{352}, {192, 320}, {192, 224}, {'max', 128}}))    -- 5(b)
   mainpipe:add(Avg(7, 7, 1, 1))
   mainpipe:add(nn.View(1024):setNumInputDims(3))
   mainpipe:add(nn.Linear(1024, 1000))

   local splitter = nn.Concat(2)
   splitter:add(mainpipe)

   if opt.auxillary then
       -- add auxillary classifier here (thanks to Christian Szegedy for the details)
       local aux_classifier = nn.Sequential()
       aux_classifier:add(Avg(5, 5, 3, 3))
       aux_classifier:add(Convolution(576, 128, 1, 1, 1, 1))
       aux_classifier:add(nn.View(128 * 4 * 4):setNumInputDims(3))
       aux_classifier:add(nn.Linear(128 * 4 * 4, 768))
       aux_classifier:add(nn.ReLU(true))
       aux_classifier:add(nn.Linear(768, 1000))

       splitter:add(aux_classifier)
   end

   local model = nn.Sequential():add(features):add(splitter):cuda()
   if opt.auxillary then
       model:add(nn.View(2, 1000):setNumInputDims(1):cuda())
   end

   if nGPU > 1 then
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1,nGPU do
          cutorch.setDevice(i)
          model:add(model_single:clone(), i)
      end
      cutorch.setDevice(1)
   end

   return model
end
