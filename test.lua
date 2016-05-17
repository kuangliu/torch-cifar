require './xLogger.lua'

l = xLogger('./log/test.log')
l:setNames{'haha','hehe'}

l:add({1,2})
l:add({3,4})
