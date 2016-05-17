checkpoint = {}

function checkpoint.latest(opt)
    if opt.checkpointPath == nil then
        print('Checkpoint saving path not specified!')
        return nil      -- use `return nil` instead of `return` is a good habit
    end

    local latestPath = paths.concat(opt.resume, 'latest.t7')
    if not paths.filep(latestPath) then
        print('Latest checkpoint not exist!')
        return nil
    end

    local latest = torch.load(latestPath)
    local optimState = torch.load(path.concat(opt.resume, latest.optimFile))

    return latest, optimState
end


function checkpoint.save(epoch, model, optimState, isBestModel, opt)
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
        modelFile = modelFilem
        optimFile = optimFile
    })

    if isBestModel then
        torch.save(bestModel, model)
    end
end

return checkpoint
