# conv_test 
不同卷积实现的效率对比 
```
Convolution test one case:
input (224,224,3), output channels 32; kernel:3*3, no pad
origin Convolution spend -----214.942ms
img2col Convolution spend -----66.136ms
origin Convolution(use Mat)-----202.171ms
winograd Convolution(use Mat):-----75.512ms
winograd Convolution(use OpenMP 2 threads):-----41.253ms


Convolution test one case:
input (112,112,3), output channels 3; kernel:3*3, no pad
img2col Convolution spend -----6.798ms
winograd Convolution(use Mat):-----3.727ms
```
