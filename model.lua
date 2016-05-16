require 'nn';

local Models = torch.class 'Models'

function Models:getVGG()
    local vgg = nn.Sequential()

    local function ConvBNReLU(nInputPlane, nOutputPlane)
        vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
        vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
        vgg:add(nn.ReLU(true))
        return vgg
    end

    local MaxPooling = nn.SpatialMaxPooling

    ConvBNReLU(3,64):add(nn.Dropout(0.3));
    ConvBNReLU(64,64);
    vgg:add(MaxPooling(2,2,2,2):ceil());

    ConvBNReLU(64,128):add(nn.Dropout(0.4));
    ConvBNReLU(128,128);
    vgg:add(MaxPooling(2,2,2,2):ceil());

    ConvBNReLU(128,256):add(nn.Dropout(0.4));
    ConvBNReLU(256,256):add(nn.Dropout(0.4));
    ConvBNReLU(256,256);
    vgg:add(MaxPooling(2,2,2,2):ceil());

    ConvBNReLU(256,512):add(nn.Dropout(0.4));
    ConvBNReLU(512,512):add(nn.Dropout(0.4));
    ConvBNReLU(512,512);
    vgg:add(MaxPooling(2,2,2,2):ceil());

    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    vgg:add(nn.View(512))

    classifier = nn.Sequential()
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(512,512))
    classifier:add(nn.BatchNormalization(512))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(512,10))
    vgg:add(classifier)

    -- Xavier/2 initialization
    for _, layer in pairs(vgg:findModules('nn.SpatialConvolution')) do
        layer.weight:normal(0,math.sqrt(2/(layer.kH*layer.kW*layer.nInputPlane)))
        layer.bias:zero()
    end

    return vgg
end
