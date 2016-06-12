require 'nn';

Conv = nn.SpatialConvolution
ReLU = nn.ReLU
BN = nn.SpatialBatchNormalization
MaxPool = nn.SpatialMaxPooling
AvgPool = nn.SpatialAveragePooling

function fireModule(nInputPlane, nOutputPlane, nSqueezePlane)
    -------------------------------------------------
    -- Squeeze: nInputPlane -> nSqueezePlane
    -- Expand1: nSqueezePlane -> nOutputPlane/2
    -- Expand2: nSqueezePlane -> nOutputPlane/2
    -- Output = Concat(Expand1, Expand2)
    -------------------------------------------------
    assert(nOutputPlane%2==0, 'nOutputPlane%2!=0')
    local halfN = nOutputPlane/2

    local squeeze = nn.Sequential()
                    :add(Conv(nInputPlane, nSqueezePlane, 1,1,1,1,0,0))
                    :add(ReLU(true))

    local expand1 = nn.Sequential()
                    :add(Conv(nSqueezePlane, halfN, 1,1,1,1,0,0))
                    :add(ReLU(true))

    local expand2 = nn.Sequential()
                    :add(Conv(nSqueezePlane, halfN, 3,3,1,1,1,1))
                    :add(ReLU(true))

    return nn.Sequential()
            :add(squeeze)
            :add(nn.Concat(2)
                :add(expand1)
                :add(expand2))
end


function getSqueezeNet()
    local net = nn.Sequential()
    net:add(Conv(3,96,7,7,2,2,3,3))
    net:add(MaxPool(3,3,2,2))

    net:add(fireModule(96,128,16))
    net:add(fireModule(128,128,16))
    net:add(fireModule(128,256,32))
    net:add(MaxPool(3,3,2,2))

    net:add(fireModule(256,256,32))
    net:add(fireModule(256,384,48))
    net:add(fireModule(384,384,48))
    net:add(fireModule(384,512,64))
    net:add(MaxPool(3,3,2,2))

    net:add(fireModule(512,512,64))
    net:add(Conv(512,1000,1,1,1,1,0,0))
    net:add(AvgPool(13,13,1,1))

    net:add(nn.View(1000))
    net:add(nn.Linear(1000,10))
    return net
end
