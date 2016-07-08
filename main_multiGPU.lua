require 'nn'
require 'cunn'
require 'xlua'
require 'image'
require 'optim'
require 'cutorch'
require './fb.lua'
require './vgg.lua'
require './resnet.lua'
require './augment.lua'
require './xLogger.lua'
require './provider.lua'
require './checkpoints.lua'

cudnn = require 'cudnn'
optnet = require 'optnet'
c = require 'trepl.colorize'

opt = lapp[[
    -n,--nGPU              (default 1)                   num of GPUs to use
    -b,--batchSize         (default 28)                 batch size
    -c,--checkpointPath    (default './checkpoints/')    checkpoint saving path
    -r,--resume                                          resume from checkpoint
]]


function setupResNet()
    print(c.blue '==> ' .. 'setting up ResNet..')

    local resnet = getResNet()
	-- this replaces 'nn' modules with 'cudnn' counterparts in-place
    cudnn.convert(resnet, cudnn):cuda()

    -- use 'optnet' to reduce memory useage
	local sample_input = torch.randn(8,3,32,32):cuda()
    optnet.optimizeMemory(resnet, sample_input, {inplace = false, mode = 'training'})

	-- with this cudnn will optimize itself for efficiency
    cudnn.benchmark = true

	-- init whole net
    --local net = nn.Sequential()
    --            :add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
    --            :cuda()
    local net = nn.Sequential()
                :add(nn.BatchFlip():float())
                :add(nn.RandomCrop(4, 'zero'):float())
                :add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))

    if opt.nGPU == 1 then
        -- use single GPU, use the first on in default
		net:add(resnet)
		cutorch.setDevice(1)  -- change the GPU ID as you like
    else
		-- multi-GPU, use GPU #1,#2,...,#n
        local gpus = torch.range(1, opt.nGPU):totable()
        net:add(nn.DataParallelTable(1, true, true)
        :add(resnet, gpus)
        :threads(function()
           local cudnn = require 'cudnn'
           cudnn.benchmark = true
        end))
    end

    print(c.blue '==> ' .. 'set criterion..')
    local criterion = nn.CrossEntropyCriterion():cuda()

    return net, criterion
end


function setupModel(opt)
    -- Either load from checkpoint or build a new model.
    if opt.resume == true then
        -- resume from checkpoint
        print(c.blue '==> ' .. 'loading from checkpoint..')
        latest = checkpoint.load(opt)
        epoch = latest.epoch
        model = torch.load(latest.modelFile)
        optimState = torch.load(latest.optimFile)
        bestTestAcc = latest.bestTestAcc
    else
        -- build a new model
        model, criterion = setupResNet()
    end
    print(model)

    return model, criterion
end


print(c.blue '==> '..'loading data..')
--provider = Provider()
provider = torch.load('provider.t7')
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)

testLogger = xLogger(paths.concat('log', 'test.log'))
if not opt.resume then
    testLogger:setNames{'Train accuracy (%)', 'Test accuracy (%)'}
end

print(c.blue '==> ' .. 'setting up model..')
net, criterion = setupModel(opt)

parameters, gradParameters = net:getParameters()
criterion = criterion or nn.CrossEntropyCriterion():cuda()

print(c.blue '==> ' .. 'configure optimizer..\n')
optimState = optimState or {
    learningRate = 0.1,
    learningRateDecay = 1e-7,
    weightDecay = 0.0005,
    momentum = 0.9,
    nesterov = true,
    dampening = 0.0
    }

bestTestAcc = bestTestAcc or 0


function train()
    net:training()
    epoch = epoch and epoch+1 or 1

    if epoch % 80 == 0 then -- after some epochs, decrease lr
        optimState.learningRate = optimState.learningRate/10
    end

    print((c.Red '==> '..'epoch: %d (lr = %.3f)')
            :format(epoch, optimState.learningRate))
    print(c.Green '==> '..'training')

    targets = torch.CudaTensor(opt.batchSize)

    indices = torch.randperm(provider.trainData:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil

    local loss = 0
    for k, v in pairs(indices) do
        xlua.progress(k, #indices)

        inputs = provider.trainData.data:index(1,v)    -- [N, C, H, W]
        print(torch.type(inputs))
		targets:copy(provider.trainData.labels:index(1,v))

        feval = function(x)
            if x~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()

            local outputs = net:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            net:backward(inputs, df_do)

            loss = loss + f
            confusion:batchAdd(outputs, targets)

            return f, gradParameters
        end
        optim.sgd(feval, parameters, optimState)
    end

    confusion:updateValids()

    trainAcc = confusion.totalValid * 100
    print((c.Green '==> '..('Train acc: '.. c.Cyan('%.2f%%')..'\tloss: '..c.Cyan('%.5f')):format(trainAcc, loss/#indices)))

    confusion:zero()
end


function test()
    net:evaluate()
    print(c.Blue '==> '..'testing')

    local bs = 125
    for i = 1, provider.testData.data:size(1), bs do
        xlua.progress(math.ceil(1+i/bs), provider.testData.data:size(1)/bs)
        local outputs = net:forward(provider.testData.data:narrow(1,i,bs))
        confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
    end

    confusion:updateValids()

    local testAcc = confusion.totalValid * 100
    local isBestModel = false
    if testAcc > bestTestAcc then
        bestTestAcc = testAcc
        isBestModel = true
    end

    print(c.Blue '==> '..('Test acc: '.. c.Cyan('%.2f%%')..'\tBest test acc: '..c.Cyan('%.2f%%')):format(testAcc, bestTestAcc))

    if testLogger then
        testLogger:add{trainAcc, testAcc}
    end

    confusion:zero()

    if epoch % 2 == 0 then
        print('\n')
        print(c.Yellow '==> '.. 'saving checkpoint..')
        checkpoint.save(epoch, net, optimState, opt, isBestModel, testAcc)
    end
    print('\n')
end


-- do for 500 epochs
for i = 1,500 do
    train()
    test()
end
