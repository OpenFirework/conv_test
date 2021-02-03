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

void conv3x3_winograd23_transform_kernel_my(const Mat& kernel, Mat& kernel_tm, int inch, int outch);
void conv3x3s1_winograd23_my(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm);
void conv3x3s1_winograd23_omp_my(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm);


int main() {
    float *input = new float[3*224*224];
    //假设input是h,w,c的排列方式，首先还是需要将通道分离出来，变成C，H，W的方式比较方便并行计算
    for(int c=0;c<3;c++) {
        for(int i=0;i<224*224;i++) {
            input[c*224*224+i] = i%10;
        }
    }

    float *weights = new float[32*3*3*3];
    for(int i=0;i<32*3*3*3;i++) {
        weights[i] = 1.0;
    }
    std::cout<<"Convolution test one case:\n";
    std::cout<<"input (224,224,3), output channels 32; kernel:3*3, no pad\n";
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

    conv3x3_x86_origin(input, inshape, outc, weights, outdata);
    int start = NowMicros();
    conv3x3_x86_origin(input, inshape, outc, weights, outdata);
    int end = NowMicros();
    std::cout<<"origin Convolution spend -----"<<(end-start)/1000.0<<"ms\n";
    
    conv3x3_x86_img2col(input, inshape, outc, weights, outdata);  
    start = NowMicros();   
    conv3x3_x86_img2col(input, inshape, outc, weights, outdata);  
    end = NowMicros();
    std::cout<<"img2col Convolution spend -----"<<(end-start)/1000.0<<"ms\n";

    Mat bottom(224,224,3, (size_t)(sizeof(float)), 1);
    for(int c=0;c<3;c++) {
        float *ptr = bottom.channel(c);
        for(int i=0;i<224*224;i++) {
             ptr[i] = i%10;
        }
       
    }
    Mat weightm(32*3*3*3, (size_t)(sizeof(float)), 1);
    for(int i=0;i<32*3*3*3;i++){
        weightm[i] = 1.0;
    }
    Mat top;

    conv3x3_x86_mat(bottom,top, weightm);
    start = NowMicros();
    conv3x3_x86_mat(bottom,top, weightm);
     end = NowMicros();
    std::cout<<"origin Convolution(use Mat)-----"<<(end-start)/1000.0<<"ms\n";

    // for(int i=0;i<24;i++) {
    //     std::cout<<top[i]<<","<<outdata[i]<<std::endl;
    // }
    Mat top2;
    top2.create(top.w, top.h, top.c, 4u, 1);
    Mat weightwino;
    conv3x3_winograd23_transform_kernel_my(weightm, weightwino, 3, 32);
    conv3x3s1_winograd23_my(bottom, top2, weightwino);
    start = NowMicros();
    conv3x3s1_winograd23_my(bottom, top2, weightwino);
    end = NowMicros();
    std::cout<<"winograd Convolution(use Mat):-----"<<(end-start)/1000.0<<"ms\n";

    start = NowMicros();
    conv3x3s1_winograd23_omp_my(bottom, top2, weightwino);
    end = NowMicros();
    std::cout<<"winograd Convolution(use omp 2 threads):-----"<<(end-start)/1000.0<<"ms\n";

    
    std::cout<<"compare result convolution， make sure results are same!\n";
    std::cout<<"first channel result:\n";
    for(int i=0;i<24;i++) {
        std::cout<<top2[i]<<","<<outdata[i]<<","<<top[i]<<std::endl;
    }
    std::cout<<"last channel result:\n";
    float* ptr0 = top2.channel(31);
    float* ptr1 = outdata+31*outw*outh;
    float* ptr2 = top.channel(31);
    int cleng = outw*outh-1;
    for(int i=0;i<24;i++) {
        std::cout<<ptr0[cleng-i]<<","<<ptr1[cleng-i]<<","<<ptr2[cleng-i]<<std::endl;
    }



    //case 2
    int inw=32;
    int inh=32;
    int inc=16;
    outc=16;

    std::cout<<"Convolution test two case:\n";
    std::cout<<"input ("<<inw<<","<<inh<<","<<inc<<"), output channels "<<outc<<"; kernel:3*3, no pad\n";
    
    float *input3 = new float[inc*inw*inh];
    //假设input是h,w,c的排列方式，首先还是需要将通道分离出来，变成C，H，W的方式比较方便并行计算
    for(int i=0;i<inc*inh*inw;i++) {
        input3[i] = i%10;
    }
    float *weights3 = new float[inc*outc*3*3];
    for(int i=0;i<inc*outc*3*3;i++) {
        weights3[i] = 1.0;
    }
    int inshape3[4] = {1,inw,inh,inc};
   
    float *outdata3;
    outw = inshape3[2]-2;
    outh = inshape3[1]-2;
    outdatalen = outc*outw*outh;
    outdata3 = new float[outdatalen];
    conv3x3_x86_img2col(input3, inshape3, outc, weights3, outdata3);  


    start = NowMicros();   
    conv3x3_x86_img2col(input3, inshape3, outc, weights3, outdata3);  
    end = NowMicros();
    std::cout<<"img2col Convolution spend -----"<<(end-start)/1000.0<<"ms\n";

    Mat bottom3(inw,inh,inc, (size_t)(sizeof(float)), 1);
    for(int c=0;c<inc;c++) {
        float *ptr = bottom3.channel(c);
        for(int i=0;i<inw*inh;i++) {
             ptr[i] = i%10;
        }
       
    }
    Mat weightm3(inc*outc*3*3, (size_t)(sizeof(float)), 1);
    for(int i=0;i<inc*outc*3*3;i++){
        weightm3[i] = 1.0;
    }
    Mat weightwino3;
    conv3x3_winograd23_transform_kernel_my(weightm3, weightwino3, inc, outc);
    Mat top3;
    top3.create(inw-2, inh-2, outc, 4u, 1);
    conv3x3s1_winograd23_my(bottom3, top3, weightwino3);

    start = NowMicros();
    conv3x3s1_winograd23_my(bottom3, top3, weightwino3);
    end = NowMicros();
    std::cout<<"winograd Convolution(use Mat):-----"<<(end-start)/1000.0<<"ms\n\n\n";

   


/*
    Mat weightwino;
    conv3x3s1_winograd23_transform_kernel_sse(weightm, weightwino, 3, 32);
    conv3x3s1_winograd23_sse(bottom, top, weightwino);
    start = NowMicros();
    conv3x3s1_winograd23_sse(bottom, top, weightwino);
    end = NowMicros();
    std::cout<<"winograd Convolution(use Mat):-----"<<(end-start)/1000.0<<"ms\n";

    conv3x3s1_winograd23_sse_omp(bottom, top, weightwino);
    start = NowMicros();
    conv3x3s1_winograd23_sse_omp(bottom, top, weightwino);
    end = NowMicros();
    std::cout<<"winograd Convolution(use OpenMP 2 threads):-----"<<(end-start)/1000.0<<"ms\n\n\n";

    std::cout<<"Convolution test one case:\n";
    std::cout<<"input (112,112,3), output channels 3; kernel:3*3, no pad\n";

   //case two
    int inshape1[4] = {1,112,112,3};
    int outc1 = 32;
    start = NowMicros();   
    conv3x3_x86_img2col(input, inshape1, outc1, weights, outdata);  
    end = NowMicros();
    std::cout<<"img2col Convolution spend -----"<<(end-start)/1000.0<<"ms\n";

    Mat bottom2(112,112,3, (size_t)(sizeof(float)), 1);
    index=0;
    for(int c=0;c<3;c++) {
        float *ptr = bottom2.channel(c);
        for(int i=0;i<112*112;i++) {
             ptr[i] = index%10;
             index++;
        }
       
    }
    Mat weightm2(16*3*3*3, (size_t)(sizeof(float)), 1);
    for(int i=0;i<16*3*3*3;i++){
        weightm2[i] = 1.0;
    }
    Mat weightwino2;
    conv3x3s1_winograd23_transform_kernel_sse(weightm2, weightwino2, 3, 16);
    Mat top2;
    top2.create(110, 110, 32, 4u, 1);
    start = NowMicros();
    conv3x3s1_winograd23_sse(bottom2, top2, weightwino2);
    end = NowMicros();
    std::cout<<"winograd Convolution(use Mat):-----"<<(end-start)/1000.0<<"ms\n\n\n";
*/

    //case 3
    /*
    float *input3 = new float[16*10*10];
    //假设input是h,w,c的排列方式，首先还是需要将通道分离出来，变成C，H，W的方式比较方便并行计算
    for(int i=0;i<16*10*10;i++) {
        input3[i] = i%10;
    }
    float *weights3 = new float[16*16*3*3];
    for(int i=0;i<16*16*3*3;i++) {
        weights3[i] = 1.0;
    }
    int inshape3[4] = {1,10,10,16};
    int outc3 = 16;
    float *outdata3;
    outw = inshape3[2]-2;
    outh = inshape3[1]-2;
    outdatalen = outc3*outw*outh;
    outdata3 = new float[outdatalen];
    conv3x3_x86_img2col(input3, inshape3, outc3, weights3, outdata3);  


    start = NowMicros();   
    conv3x3_x86_img2col(input3, inshape3, outc3, weights3, outdata3);  
    end = NowMicros();
    std::cout<<"img2col Convolution spend -----"<<(end-start)/1000.0<<"ms\n";

    Mat bottom3(10,10,16, (size_t)(sizeof(float)), 1);
    for(int c=0;c<16;c++) {
        float *ptr = bottom3.channel(c);
        for(int i=0;i<10*10;i++) {
             ptr[i] = i%10;
        }
       
    }
    Mat weightm3(16*16*3*3, (size_t)(sizeof(float)), 1);
    for(int i=0;i<16*16*3*3;i++){
        weightm3[i] = 1.0;
    }
    Mat weightwino3;
    conv3x3_winograd23_transform_kernel_my(weightm3, weightwino3, 16, 16);
    Mat top3;
    top3.create(8, 8, 16, 4u, 1);
    conv3x3s1_winograd23_my(bottom3, top3, weightwino3);

    start = NowMicros();
    conv3x3s1_winograd23_my(bottom3, top3, weightwino3);
    end = NowMicros();
    std::cout<<"winograd Convolution(use Mat):-----"<<(end-start)/1000.0<<"ms\n\n\n";

   */



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