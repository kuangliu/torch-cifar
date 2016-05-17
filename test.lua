require 'nn'

opt = lapp[[
    -g,--gpu               (default 3)                   GPU ID
    -c,--checkpointPath    (default './checkpoints/')    checkpoint saving path
    -b,--batchSize         (default 32)                  batch size
    -r,--resume                                          resume from checkpoint
]]


a = torch.load('a.t7')


for i, m in ipairs(a:findModules('nn.ConcatTable')) do
    print(i,m)
end
