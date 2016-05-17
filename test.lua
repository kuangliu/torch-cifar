require 'nn'

opt = lapp[[
    -g,--gpu               (default 3)                   GPU ID
    -c,--checkpointPath    (default './checkpoints/')    checkpoint saving path
    -b,--batchSize         (default 32)                  batch size
    -r,--resume                                          resume from checkpoint
]]



epoch=nil
for i = 1,10 do
    epoch = epoch and epoch+1 or 1
    print(epoch)
end
