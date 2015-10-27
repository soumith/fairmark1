function createModel(nGPU)
   require 'cudnn'

   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local features = nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   features:add(nn.SpatialBatchNormalization(64,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(cudnn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   features:add(nn.SpatialBatchNormalization(192,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(cudnn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(384,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
   features:cuda()
   if nGPU > 1 then
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local features_single = features
      features = nn.DataParallelTable(1)
      for i=1,nGPU do
         cutorch.withDevice(i, function()
                               features:add(features_single:clone(), i)
         end)
      end
      features.gradInput = nil
      features.flattenParams = true
      features.useCollectives = opt.useCollectives
   end

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))

   local branch1
   branch1 = nn.Concat(2)
   local s = nn.Sequential()
   s:add(nn.Linear(256*6*6, 4096))
   s:add(nn.BatchNormalization(4096,1e-3))
   s:add(nn.ReLU())
   branch1:add(s)
   classifier:add(branch1)
   local branch2
   branch2 = nn.Concat(2)

   local s = nn.Sequential()
   s:add(nn.Linear(4096, 4096))
   s:add(nn.BatchNormalization(4096))
   s:add(nn.ReLU())
   branch2:add(s)

   classifier:add(branch2)
   classifier:add(nn.Linear(4096, 1000))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential():cuda():add(features):add(classifier:cuda())

   return model
end
