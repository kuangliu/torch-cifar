require 'nn'

ReLU = nn.ReLU
Conv = nn.SpatialConvolution
MaxPool = nn.SpatialMaxPooling
AvgPool = nn.SpatialAveragePooling
BN = nn.SpatialBatchNormalization

local shortCutType = 'CONV' or 'ZERO_PAD'
local blockType = 'BOTTLENECK' or 'BASIC'


function shortCut(nInputPlane, nOutputPlane, stride)
    -----------------------------------------------------------------------
    -- The shortcut layer is either:
    --      - Identity: when input shape == output shape
    --      - Zero padding: when input shape ~= output shape
    --      - 1x1 CONV: when input shape ~= output shape
    -----------------------------------------------------------------------
    if nInputPlane == nOutputPlane then
        return nn.Identity()
    elseif shortCutType == 'CONV' then
        return nn.Sequential()
            :add(Conv(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(BN(nOutputPlane))
    elseif shortCutType == 'ZERO_PAD'then
        return nn.Sequential()
            :add(AvgPool(1, 1, stride, stride))
            :add(nn.Concat(2)
                :add(nn.Identity())
                :add(nn.MulConstant(0)))
    else
        error('Unknown shortCutType!')
    end
end


function basicblock(nConvPlane, stride)
    ---------------------------------------------------------------------------
    -- The basic block of ResNet: a 2 CONV layer CNN + an Identity shortcut.
    -- In torch we use `ConCatTable` to implement.
    -- Inputs:
    --      - nConvPlane: CONV channels
    --      - stride: CONV stride
    -- Note the 2 CONV layers share the same CONV channels.
    ---------------------------------------------------------------------------
    local nInputPlane = nPlane
    nPlane = nConvPlane

    local s = nn.Sequential()
    s:add(Conv(nInputPlane,nConvPlane,3,3,stride,stride,1,1))
    s:add(BN(nConvPlane))
    s:add(ReLU(true))
    s:add(Conv(nConvPlane,nConvPlane,3,3,1,1,1,1))
    s:add(BN(nConvPlane))

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(s)
            :add(shortCut(nInputPlane, nConvPlane, stride)))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
end


function bottleneck(nFirstConvPlane, stride)
    -----------------------------------------------------------------------
    -- Bottlenect block.
    -- which follows the structure:
    --      1. 1x1 CONV with nFirstConvPlane channels
    --      2. 3x3 CONV with nFirstConvPlane channels
    --      3. 1x1 CONV with 4*nFirstConvPlane channles
    -- Inputs:
    --      - nFirstConvPlane: CONV channels of the first CONV layer
    --      - stride: CONV stride
    -----------------------------------------------------------------------
    local nInputPlane = nPlane              -- intput data channels
    local nOutputPlane = 4*nFirstConvPlane  -- first conv layer channels
    nPlane = nOutputPlane                   -- nPlane flow between blocks

    local s = nn.Sequential()
    s:add(Conv(nInputPlane,nFirstConvPlane,1,1,1,1,0,0))
    s:add(BN(nFirstConvPlane))
    s:add(ReLU(true))
    s:add(Conv(nFirstConvPlane,nFirstConvPlane,3,3,stride,stride,1,1))
    s:add(BN(nFirstConvPlane))
    s:add(ReLU(true))
    s:add(Conv(nFirstConvPlane,nOutputPlane,1,1,1,1,0,0))
    s:add(BN(nOutputPlane))

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(s)
            :add(shortCut(nInputPlane, nOutputPlane, stride)))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
end


function nBlock(nFirstConvPlane, n, stride)
    -------------------------------------------------------------------------
    -- Stack n basic/bottleneck blocks together
    -- Inputs:
    --      - nFirstConvPlane: CONV channels of the first CONV layer
    --          - For basic block: all CONV layers in the nBlock share the same CONV channels (=nFirstConvPlane)
    --          - For bottlenect block:
    --              - nSecondConvPlane = nFirstConvPlane
    --              - nThirdConvPlane = 4*nFirstConvPlane
    --      - n: # of blocks
    --      - stride: CONV stride of first block, could be 1 or 2
    --          - stride=1: maintain input H&W
    --          - stride=2: decrease input H&W by half
    -------------------------------------------------------------------------
    local block
    if blockType == 'BASIC' then
        block = basicblock
    elseif blockType == 'BOTTLENECK' then
        block = bottleneck
    else
        error('Unknown blockType..'..blockType)
    end

    local s = nn.Sequential()
    s:add(block(nFirstConvPlane, stride))

    for i = 2,n do
        s:add(block(nFirstConvPlane, 1))
    end

    return s
end


function getResNet()
    --------------------------------------------------
    -- Define the CIFAR-10 ResNet
    --------------------------------------------------
    local net = nn.Sequential()
    -- net:add(Conv(3,16,3,3,1,1,1,1))
    -- net:add(BN(16))
    net:add(Conv(3,64,3,3,1,1,1,1))
    net:add(BN(64))
    net:add(ReLU(true))

    nPlane = 64

    net:add(nBlock(64,3,1))
    net:add(nBlock(128,3,2))
    net:add(nBlock(256,3,2))

    net:add(AvgPool(8,8,1,1))
    net:add(nn.View(1024):setNumInputDims(3))
    net:add(nn.Linear(1024,10))

    --Xavier/2 initialization
    for _, layer in pairs(net:findModules('nn.SpatialConvolution')) do
        local n = layer.kH*layer.kW*layer.nOutputPlane
        layer.weight:normal(0, math.sqrt(2/n))
        layer.bias:zero()
    end

    return net
end
