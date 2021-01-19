#include <iostream>
#include <fstream> 
#include <sys/time.h> 
#include "mat.h"
#include<bitset>
using namespace std;

int32_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int32_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

void conv3x3_x86_origin(float *inputdata, int inshape[4], int outc, float *weights,  float *outdata);
void conv3x3_x86_img2col(float *inputdata, int inshape[4], int outc, float *weights,  float *outdata);
void conv3x3_x86_mat(const Mat& bottom_blob, Mat& top_blob, Mat &weight_data);

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

   
   
    Mat bottom(224,224,3,(void*)input, (size_t)(sizeof(float)), 1);
    Mat weightm(32*3*3*3,(void*)weights, (size_t)(sizeof(float)), 1);
    Mat top;

    start = NowMicros();
    conv3x3_x86_mat(bottom,top, weightm);
     end = NowMicros();
    std::cout<<(end-start)/1000.0<<"ms\n";
    // for(int q=0;q<1;q++) {
    //     float *ptr = top.channel(q);
    //     for(int i=0;i<222*222;i++) {
    //         std::cout<<ptr[i]<<"\n";
    //     }
    // }


    
  /* 
    Mat mat(2,3,4,(size_t)(sizeof(float)),1);
    int value=0;
    for(int c=0;c<4;c++) {
        float *ptr = mat.channel(c);
        int index=0;
        for(int h=0;h<3;h++) {
            for(int w=0;w<2;w++) {
                ptr[index]=value;
                index++;
                value++;
            }
        }
    }

    for(int c=0;c<4;c++) {
        float *ptr = mat.channel(c);
        int index=0;
        for(int h=0;h<3;h++) {
            for(int w=0;w<2;w++) {
                std::cout<<ptr[index]<<",";
                index++;
            }
            std::cout<<"\n";
        }
        std::cout<<"\n\n";
    }

    Mat dst;
    convert_packing(mat,dst,2);
    
  
    float *p = (float*)dst.data;
    for(int i=0;i<24;i++) {
        std::cout<<p[i]<<",";
    }

    std::cout<<"\n";
    p = (float*)mat.data;
    for(int i=0;i<24;i++) {
        std::cout<<p[i]<<",";
    }
   
    std::cout<<"\n";
    int a=123;
    bitset<32> bs(a);
    cout<<bs<<endl;
    bitset<32> b1(a+15);
    cout<<b1<<endl;
    a=-16;
    bitset<32> b2(a);
    cout<<b2<<endl;
    cout<<(b1&b2)<<endl;*/
    return 0;
}