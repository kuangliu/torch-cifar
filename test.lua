require './xLogger.lua'

l = xLogger('./log/test.log')
l:setNames{'haha','hehe'}

l:add({1,2})
l:add({3,4})


a = false

if not a then
    print('a')
end


a = {1.2313141,2.331314}

b = ('%.2f'):format(a[1])
b

function table2String(tbl)
    local s = ''
    for _,v in pairs(tbl) do
        s = s .. ('%.2f'):format(v) .. '\t'
    end
    return s
end

table2String(a)
