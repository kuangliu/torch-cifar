require 'xlua';
require 'optim';
require 'nn';
require './model.lua'
require './provider.lua'

c = require 'trepl.colorize'

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

print(c.blue '==>' .. ' configuring model')

net = nn.Sequential()
net:add(nn.BatchFlip():float())

vgg = Models:getVGG()
net:add(vgg:float())

print(net)

print(c.blue '==>' .. 'loading data')
provider = Provider()
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)
paths.mkdir('log')
testLogger = optim.Logger(paths.concat('log','test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}

parameters, gradParameters = net:getParameters()

print(c.blue '==>' .. 'set criterion')

criterion = nn.CrossEntropyCriterion():float()

print(c.blue '==>' .. 'configure optimizer')

optimState = {
    learningRate = 1e-3,
    learningRateDecay = 1e-7,
    weightDecay = 0.0005,
    momentum = 0.9,
    }

opt = {
    batchSize = 128
    }

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
    print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
      confusion.totalValid * 100, torch.toc(tic)))

    train_acc = confusion.totalValid * 100
    confusion:zero()
    epoch = epoch + 1
end


function test()
    net:evaluate()
    print(c.blue '==>' .. 'testing')

    local bs = 125
    for i = 1, provider.testData.data:size(1), bs do
        xlua.progress(i, provider.testData.data:size(1))

        local outputs = net:forward(provider.testData.data:narrow(1,i,bs))
        confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
    end

    confusion:updateValids()
    print('test accuracy: ', confusion.totalValid * 100)

    if testLogger then
        testLogger:add{train_acc, confusion.totalValid*100}
        testLogger:style{'-','-'}
    end

    confusion:zero()
end

-- do for 300 epochs
for i = 1,300 do
    train()
    test()
end
