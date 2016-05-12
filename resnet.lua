
require 'nn';

ReLU = nn.ReLU
Conv = nn.SpatialConvolution
MaxPool = nn.SpatialMaxPooling
AvgPool = nn.SpatialAveragePooling
BN = nn.SpatialBatchNormalization

function shortCut(nInputPlane, nOutputPlane, stride)
    -----------------------------------------------------------------------
    -- The shortcut layer is either:
    --      - Identity: within the same nBlob
    --      - 1x1 CONV: surrounding the first blob or between different nBlobs
    -----------------------------------------------------------------------
    if stride == 2 then
        -- the first blob, the short cut is CONV
        return nn.Sequential()
            :add(Conv(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(BN(nOutputPlane))
    else
        return nn.Identity()
    end
end


function blob(nInputPlane, nOutputPlane, stride)
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


function nBlob(nInputPlane, nOutputPlane)
    -------------------------------------------------------------------------
    -- Stack n blobs together, each of these blobs are of the same filter num.
    -- In this particular case, we use `n=3`.
    -- Inputs:
    --      - nInputPlane: # of input channels
    --      - nOutputPlane: # of CONV kernels
    -------------------------------------------------------------------------
    local s = nn.Sequential()

    -- The first blob of nBlob: double the # of CONV kernels; decrease the size
    -- by using `stride=2`
    s:add(blob(nInputPlane, nOutputPlane, 2))

    -- The rest blobs of nBlob: # of CONV kernels unchanged, and use `stride=1`
    local n = 3
    for i = 2,n do
        -- For the first blob of nBlob, use stride=2 to decrease the size
        s:add(blob(nOutputPlane, nOutputPlane, 1))
    end

    return s
end

function cifarResNet()
    --------------------------------------------------
    -- Define the CIFAR-10 ResNet
    --------------------------------------------------
    local net = nn.Sequential()
    net:add(Conv(3,16,3,3,1,1,1,1))
    net:add(BN(16))
    net:add(ReLU(true))

    net:add(nBlob(16,16))
    net:add(nBlob(16,32))
    net:add(nBlob(32,64))

    net:add(AvgPool(4,4,1,1))
    net:add(nn.View(64):setNumInputDims(3))
    net:add(nn.Linear(64,10))

    -- Xavier/2 initialization
    for _, layer in pairs(net:findModules('nn.SpatialConvolution')) do
        layer.weight:normal(0, math.sqrt(2/layer.kH*layer.kW*layer.nInputPlane))
        layer.bias:zero()
    end

    return net
end
