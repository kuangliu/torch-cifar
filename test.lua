require 'image'

a = torch.range(1,12):reshape(3,2,2)

print(a)
image.flip(a, b)

print(b)
