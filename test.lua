require 'xlua'

for i = 1,1000 do
    io.write(i)
    xlua.progress(i,1000)
end
