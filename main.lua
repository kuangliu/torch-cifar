require 'xlua'
require 'image'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require './model.lua'
require './resnet.lua'
require './provider.lua'
require './fb.lua'

c = require 'trepl.colorize'

args = lapp[[
    -g,--gpu    (default 3)    GPU_ID
]]

cutorch.setDevice(args.gpu)

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


local function cast(t)
    return t:cuda()
end


print(c.blue '==> '..' configuring model')

net = nn.Sequential()
net:add(nn.BatchFlip():float())
net:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))

--vgg = Models:getVGG()
--net:add(cast(vgg))
resnet = cifarResNet()
--resnet = createModel()
net:add(cast(resnet))
net:get(2).updateGradInput = function(input) return end

print(net)

-- use cuDNN
cudnn.convert(net:get(3), cudnn)

print(c.blue '==> '..'loading data')
provider = Provider()
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)
paths.mkdir('log')
testLogger = optim.Logger(paths.concat('log','test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}

parameters, gradParameters = net:getParameters()

print(c.blue '==> '..'set criterion')

criterion = cast(nn.CrossEntropyCriterion())

print(c.blue '==> '..'configure optimizer')

optimState = {
    learningRate = 0.1,
    learningRateDecay = 1e-7,
    weightDecay = 0.0005,
    momentum = 0.9,
    }

opt = {
    batchSize = 256
    }


function train()
    net:training()
    epoch = epoch or 1

    if epoch % 80 == 0 then -- after some epochs, decrease lr
        optimState.learningRate = optimState.learningRate/10
    end

    print((c.Red '==> '..'epoch: %d (lr = %.3f)'):format(epoch, optimState.learningRate))

    print(c.Green '==> '..'training')

    targets = cast(torch.FloatTensor(opt.batchSize))

    indices = torch.randperm(provider.trainData:size(1)):long():split(opt.batchSize)
    indices[#indices] = nil

    local lastLoss = 0

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

            lastLoss = f
	
            confusion:batchAdd(outputs, targets)

            return f, gradParameters
        end
        optim.sgd(feval, parameters, optimState)
    end

    confusion:updateValids()
    
    train_acc = confusion.totalValid * 100
    print((c.Green '==> '..('Train acc: %.2f%%\tloss: %.5f '):format(train_acc, lastLoss)))
    
    confusion:zero()
    epoch = epoch + 1
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
    print(c.Blue '==> '..('Test acc: %.2f%% '):format(confusion.totalValid * 100))
    print('\n')

    if testLogger then
        testLogger:add{train_acc, confusion.totalValid*100}
--        testLogger:style{'-','-'}
    end

    confusion:zero()
end


-- do for 500 epochs
for i = 1,500 do
    train()
    test()
end
