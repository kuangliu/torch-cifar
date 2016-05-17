------------------------------------------------------------------
-- My own version of Logger.
-- Use file 'append' mode, support checkpoint.
------------------------------------------------------------------

local xLogger = torch.class('xLogger')


local function table2String(tbl)
    local s = ''
    for _,v in pairs(tbl) do
        s = s .. ('%.2f'):format(v) .. '\t'
    end
    return s
end


function xLogger:__init(logName)
    self.names = {}
    self.logName = logName
    self.file = io.open(logName, 'a+')
end


function xLogger:setNames(names)
    self.names = names
    local s = table2String(names)
    self.file:write(s..'\n')
end


function xLogger:add(entry)
    -- concat table entry to a string
    local s = table2String(entry)
    self.file:write(s..'\n')
end
