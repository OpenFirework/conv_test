# conv_test 
不同卷积实现的效率对比 
```
Convolution test one case:
input (224,224,3), output channels 32; kernel:3*3, no pad
origin Convolution spend -----185.894ms
img2col Convolution spend -----115.071ms
origin Convolution(use Mat)-----208.848ms
winograd Convolution(use Mat):-----77.249ms
winograd Convolution(use omp 2 threads):-----44.353ms

Convolution test two case:
input (32,32,16), output channels 16; kernel:3*3, no pad
img2col Convolution spend -----1.917ms
winograd Convolution(use Mat):-----3.335ms


compare result convolution， make sure results are same!
first channel result:
105,105,105
102,102,102
99,99,99
126,126,126
123,123,123
120,120,120
117,117,117
144,144,144
141,141,141
138,138,138
105,105,105
102,102,102
99,99,99
126,126,126
123,123,123
120,120,120
117,117,117
144,144,144
141,141,141
138,138,138
105,105,105
102,102,102
99,99,99
126,126,126
last channel result:
120,120,120
123,123,123
126,126,126
99,99,99
102,102,102
105,105,105
138,138,138
141,141,141
144,144,144
117,117,117
120,120,120
123,123,123
126,126,126
99,99,99
102,102,102
105,105,105
138,138,138
141,141,141
144,144,144
117,117,117
120,120,120
123,123,123
126,126,126
99,99,99
```
