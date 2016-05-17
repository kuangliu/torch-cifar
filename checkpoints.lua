checkpoint = {}

function checkpoint.load(opt)
    if opt.checkpointPath == nil then
        print('Checkpoint saving path not specified!')
        return nil      -- use `return nil` instead of `return` is a good habit
    end

    local latestPath = paths.concat(opt.checkpointPath, 'latest.t7')
    if not paths.filep(latestPath) then
        print('Latest checkpoint not exist!')
        return nil
    end

    local latest = torch.load(latestPath)

    return latest
end


function checkpoint.save(epoch, model, optimState, opt, isBestModel)
    if opt.checkpointPath == nil then
        print('Checkpoint saving path not specified!')
        return nil
    end

    local modelFile = paths.concat(opt.checkpointPath, 'model_'..epoch..'.t7')
    local optimFile = paths.concat(opt.checkpointPath, 'optimState_'..epoch..'.t7')
    local latest = paths.concat(opt.checkpointPath, 'latest.t7')
    local bestModel = paths.concat(opt.checkpointPath, 'model_best.t7')

    torch.save(modelFile, model)
    torch.save(optimFile, optimState)
    torch.save(latest, {
        epoch = epoch,
        modelFile = modelFile,
        optimFile = optimFile
    })

    if isBestModel then
        -- TODO save best TestAcc for resuming
        torch.save(bestModel, model)

    end
end

return checkpoint
