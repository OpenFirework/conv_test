# conv_test 
不同卷积实现的效率对比 
```
Convolution test one case:
input (224,224,3), output channels 32; kernel:3*3, no pad
origin Convolution spend -----180.075ms
img2col Convolution spend -----66.543ms
origin Convolution(use Mat)-----204.082ms
winograd Convolution(use Mat):-----74.286ms
winograd Convolution(use OpenMP 2 threads):-----41.477ms
Convolution test one case:
input (112,112,3), output channels 3; kernel:3*3, no pad
img2col Convolution spend -----68.323ms
winograd Convolution(use Mat):-----80.821ms
```
