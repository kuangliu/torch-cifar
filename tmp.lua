require 'nn';

ReLU = nn.ReLU
Conv = nn.SpatialConvolution
MaxPool = nn.SpatialMaxPooling
AvgPool = nn.SpatialAveragePooling
BN = nn.SpatialBatchNormalization

net = nn.Sequential()
-- net:add(Conv(3,16,3,3,1,1,1,1))
-- net:add(BN(16))
net:add(Conv(3,64,3,3,1,1,1,1))
net:add(BN(64))
net:add(ReLU(true))

net:add(Conv(64,64,3,3,1,1,1,1))
net:add(BN(64))
net:add(ReLU(true))

net:add(Conv(64,64,3,3,2,2,1,1))
net:add(BN(64))
net:add(ReLU(true))


net:add(AvgPool(8,8,1,1))

y = net:forward(torch.Tensor(1,3,32,32))
#y
