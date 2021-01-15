#include <iostream>
#include <fstream> 
#include <sys/time.h> 
using namespace std;

int32_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int32_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

void conv3x3_x86_origin(float *inputdata, int inshape[4], int outc, float *weights,  float *outdata);
void conv3x3_x86_img2col(float *inputdata, int inshape[4], int outc, float *weights,  float *outdata);

int main() {
    float *input = new float[3*224*224];
    //假设input是h,w,c的排列方式，首先还是需要将通道分离出来，变成C，H，W的方式比较方便并行计算
    for(int i=0;i<3*224*224;i++) {
        input[i] = i%10;
       // input[i] = 1.0;
    }
    float *weights = new float[32*3*3*3];
    for(int i=0;i<32*3*3*3;i++) {
        weights[i] = 1.0;
    }
    std::cout<<input[3*224*224-1]<<"\n";
    // ofstream file("weights.bin",ios::out|ios::binary);
    // file.write((char*)weights,32*3*3*3*sizeof(float));
    // file.close();
    int inshape[4] = {1,224,224,3};
    int outc = 32;
    float *outdata;
    int outw = inshape[2]-2;
    int outh = inshape[1]-2;
    int outdatalen = outc*outw*outh;
    outdata = new float[outdatalen];
    int start = NowMicros();
    conv3x3_x86_origin(input, inshape, outc, weights, outdata);
    int end = NowMicros();
    std::cout<<(end-start)/1000.0<<"ms\n";


    start = NowMicros();
    //while(1)
    {
        conv3x3_x86_img2col(input, inshape, outc, weights, outdata);
    }
    end = NowMicros();
    std::cout<<(end-start)/1000.0<<"ms\n";

    // start = NowMicros();
    // conv3x3_x86(input, inshape, outc, weights, outdata);
    // end = NowMicros();
    // std::cout<<(end-start)/1000.0<<"ms\n";
    //  for(int i=0;i<outh;i++) {
    //       for(int j=0;j<outw;j++)
    //         std::cout<<i<<","<<j<<",:\t"<<outdata[i*outw+j]<<"\n";
    //  }
           
    // for(int c=0;c<outc;c++) {
    //     int index = c*outw*outh;
    //     std::cout<<outdata[index]<<",";
    //     if(c==30) {
    //         for(int i=0;i<outw*outh;i++) {
    //             std::cout<<i<<","<<outdata[index+i]<<"\n";
    //         }
    //     }
    // }

    return 0;
}