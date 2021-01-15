import torch
import torch.nn.functional as F
import time 
x=torch.ones(1,3,224,224)
weights=torch.ones(32,3,3,3)
bias=torch.zeros(32)
input=x.reshape(-1)
for i in range(2*224*224):
    input[i]=i%10.
input = input.reshape(1,3,224,224)
out=F.conv2d(x,weights,bias,stride=1,padding=0)

time_start=time.time()
while(True):
    out=F.conv2d(x,weights,bias,stride=1,padding=0)
time_end=time.time()
print('time cost',time_end-time_start,'s')
