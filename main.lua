require 'xlua'
require 'optim'
require 'nn'
--require './vgg.lua'
require './resnet.lua'
require './provider.lua'
require './checkpoints.lua'
require './fb.lua'

c = require 'trepl.colorize'

opt = lapp[[
    -g,--gpu               (default 3)                   GPU ID
    -c,--checkpointPath    (default './checkpoints/')    checkpoint saving path
    -b,--batchSize         (default 256)                  batch size
    -r,--resume                                          resume from checkpoint
]]


do
    BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

    function BatchFlip:__init()
        parent.__init(self)
        self.train = true
    end

    function BatchFlip:updateOutput(input)
        if self.train then
            local batchSize = input:size(1)
            local flipMask = torch.randperm(batchSize):le(batchSize/2)

            for i = 1, batchSize do
                if flipMask[i] == 1 then
                    image.hflip(input[i], input[i])
                end
            end
        end
        self.output:set(input)
        return self.output
    end
end


function setupResNet()
    print(c.blue '==> ' .. 'setting up ResNet..')
    local net = nn.Sequential()
    --net:add(nn.BatchFlip():float())

    --vgg = getVGG()
    --net:add(vgg:float())
    resnet = cifarResNet()
    --resnet = createModel()
    net:add(resnet:float())

    print(c.blue '==> ' .. 'set criterion..')
    local criterion = nn.CrossEntropyCriterion():float()

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
    else
        -- build a new model
        model, criterion = setupResNet()
    end

    return model, criterion
end


print(c.blue '==> ' .. 'loading data')
--provider = Provider()
provider = torch.load('provider.t7')
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)
paths.mkdir('log')
testLogger = optim.Logger(paths.concat('log','test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}


print(c.blue '==> ' .. 'setting up model..')
net, criterion = setupModel(opt)
parameters, gradParameters = net:getParameters()
criterion = criterion or nn.CrossEntropyCriterion():float()

print(c.blue '==> ' .. 'configure optimizer')
optimState = optimState or {
    learningRate = 1e-3,
    learningRateDecay = 1e-7,
    weightDecay = 0.0005,
    momentum = 0.9,
    nesterov = true,
    dampening = 0.0
    }

bestTestAcc = 0

function train()
    net:training()
    epoch = epoch or 1

    if epoch % 25 == 0 then -- every 25 epochs, decrease lr
        optimState.learningRate = optimState.learningRate/2
    end

    print('online epoch #' .. epoch .. ' batch size = ' .. opt.batchSize)

    targets = torch.FloatTensor(opt.batchSize)

    indices = torch.randperm(provider.trainData:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil

    local tic = torch.tic()

    for k, v in pairs(indices) do
        xlua.progress(k, #indices)

        inputs = provider.trainData.data:index(1,v)    -- [N, C, H, W]
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

            confusion:batchAdd(outputs, targets)

            return f, gradParameters
        end
        optim.sgd(feval, parameters, optimState)
    end

    confusion:updateValids()

    trainAcc = confusion.totalValid * 100
    print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
      trainAcc, torch.toc(tic)))

    confusion:zero()

end


function test()
    net:evaluate()
    print(c.blue '==> ' .. 'testing')

    local bs = 125
    for i = 1, provider.testData.data:size(1), bs do
        xlua.progress(i, provider.testData.data:size(1))

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
    print('test accuracy: ', testAcc)

    if testLogger then
        testLogger:add{trainAcc, testAcc}
    end

    confusion:zero()

    torch.save('a.t7', net)

    checkpoint.save(epoch, net, optimState, opt, isBestModel)
    epoch = epoch + 1
end

-- do for 300 epochs
for i = 1,300 do
    train()
    test()
end
