require 'nn'

ReLU = nn.ReLU
Conv = nn.SpatialConvolution
MaxPool = nn.SpatialMaxPooling
AvgPool = nn.SpatialAveragePooling
BN = nn.SpatialBatchNormalization

local shortCutType = 'CONV' or 'ZERO_PAD'
local blobType = 'NIN' or 'BASIC'

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
        assert(1==2, 'Unknown shortCutType!')     -- never reached here.
    end
end


function basicBlob(nInputPlane, nOutputPlane, stride)
    -----------------------------------------------------------------------
    -- The basic blob of ResNet: a normal small CNN + an Identity shortcut.
    -- In torch we use `ConCatTable` to implement.
    -- Inputs:
    --      - nInputPlane: # of input channels
    --      - nOutputPlane: # of kernels
    --      - stride: CONV stride
    -----------------------------------------------------------------------
    local s = nn.Sequential()
    s:add(Conv(nInputPlane,nOutputPlane,3,3,stride,stride,1,1))
    s:add(BN(nOutputPlane))
    s:add(ReLU(true))
    s:add(Conv(nOutputPlane,nOutputPlane,3,3,1,1,1,1))
    s:add(BN(nOutputPlane))

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(s)
            :add(shortCut(nInputPlane, nOutputPlane, stride)))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
end


function nBasicBlob(nInputPlane, nOutputPlane, n, stride)
    -------------------------------------------------------------------------
    -- Stack n basic blobs together, each of these blobs share the same filter num.
    -- Inputs:
    --      - nInputPlane: # of input channels
    --      - nOutputPlane: # of CONV kernels
    --      - n: # of blobs
    --      - stride: CONV stride of first blob, could be 1 or 2
    --          - stride=1: maintain input H&W
    --          - stride=2: decrease input H&W by half
    -------------------------------------------------------------------------
    local s = nn.Sequential()

    -- The first blob of nBlob
    s:add(basicBlob(nInputPlane, nOutputPlane, stride))

    -- The rest blobs of nBlob: # of CONV kernels unchanged, and use `stride=1`
    for i = 2,n do
        -- For the first blob of nBlob, use stride=2 to decrease the size
        s:add(basicBlob(nOutputPlane, nOutputPlane, 1))
    end

    return s
end


function ninBlob(nOutputPlane, stride)
    local nInputPlane = nPlane              -- intput data channels
    local nFirstConvPlane = nOutputPlane/4  -- first conv layer channels
    nPlane = nOutputPlane                   -- nPlane flow between blobs

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


function nNINBlob(nOutputPlane, n, stride)
    local s = nn.Sequential()

    s:add(ninBlob(nOutputPlane, stride))

    for i = 2,n do
        s:add(ninBlob(nOutputPlane, 1))
    end

    return s
end


function cifarResNet()
    --------------------------------------------------
    -- Define the CIFAR-10 ResNet
    --------------------------------------------------
    local net = nn.Sequential()
    -- net:add(Conv(3,16,3,3,1,1,1,1))
    -- net:add(BN(16))
    net:add(Conv(3,64,3,3,1,1,1,1))
    net:add(BN(64))
    net:add(ReLU(true))

    -- net:add(nBasicBlob(16,16,3,1))
    -- net:add(nBasicBlob(16,32,3,2))
    -- net:add(nBasicBlob(32,64,3,2))
    --net:add(nBasicBlob(64,64,3,2))

    nPlane = 64
    net:add(nNINBlob(256,3,1))
    net:add(nNINBlob(512,3,2))
    net:add(nNINBlob(1024,3,2))
    net:add(AvgPool(8,8,1,1))
    net:add(nn.View(1024):setNumInputDims(3))
    net:add(nn.Linear(1024,10))

    print(net)


    --Xavier/2 initialization
    for _, layer in pairs(net:findModules('nn.SpatialConvolution')) do
        local n = layer.kH*layer.kW*layer.nOutputPlane
        layer.weight:normal(0, math.sqrt(2/n))
        layer.bias:zero()
    end

    return net
end
