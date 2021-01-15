#include <iostream>
using namespace std;
//输入数据的排列方式c,h,w
//strid=1, padding=0,
//朴素的实现，直接相乘
void conv3x3_x86_origin(float *inputdata, int inshape[4], int outc, float *weights,  float *outdata) {
    int n = inshape[0];
    int h = inshape[1];
    int w = inshape[2];
    int c = inshape[3];

    int outw = w-2;
    int outh = h-2;
    for(int i=0; i<outc; i++) {
        float *ptr = outdata+i*outh*outw; //当前输出通道的指针
        float *weight = weights +i*3*3*3;  //当前通道使用的卷积kernal
        for(int k=0;k<outh;k++) {
            for(int l=0;l<outw;l++) {
                //ptr[k*outw+l] 当前通道的特征图的计算位置
                float data = 0;
                float sums = 0;
                for(int x=0;x<c;x++) {  //输入的通道数
                    float *inputinner = inputdata+x*h*w;
                    float *innerweight=weight+x*3*3; //3*3
                    int left = l;
                    int top = k;
                    int right = l+2;
                    int bottom = k+2;
                    
                    int start_index = top*w+left;
                    
                    float inter = 0;
                    for(int wi=0;wi<9;wi++) {
                        int index = start_index + (wi/3)*w + wi%3;
                        inter += innerweight[wi]*inputinner[index];
                    }
                    sums += inter; 
                }
                ptr[k*outw+l] = sums;

            }
        }
        
    }
}

void img2col(float *imgData, int imgWidth, int imgHeight, int imgCh, float *col, int ksize, int stride=1) {
    int out_h = (imgHeight - ksize)/stride + 1;
    int out_w = (imgWidth - ksize)/stride + 1;
    int out_size = out_h*out_w*imgCh*ksize*ksize;
    int out_index = 0;
    for(int h=0;h<imgHeight;h++) {
        if(h+ksize>imgHeight) {
            continue;
        }
        for(int w=0;w<imgWidth;w++) {
            if(w+ksize>imgWidth) {
                continue;
            }
            int index = h*imgWidth+w;  //每个特征图当前的位置，从此处将kernelsize的元素展开成一维
            for(int c=0;c<imgCh;c++) {
                float *pdata = imgData + c*imgWidth*imgHeight + index;   //当前通道，当前特征图位置的pos
                for(int s=0; s<ksize; s++) {
                    int cur_index = index + s*imgWidth;   //当前特征图的vector的新的起始位置
                    for(int s1=0; s1<ksize; s1++) {
                        col[out_index] = pdata[cur_index];
                        out_index++;
                        cur_index++;
                    }

                }
            }
        }
    }
/*
    for(int c=0;c<imgCh;c++) {
        float *inptr = imgData + c*imgHeight*imgWidth;
        for(int i=0;i<imgHeight;i++) {
            if(i+ksize>imgHeight) {
                continue;
            }
            for(int j=0;i<imgWidth;j++) {
                if(j+ksize>imgWidth) {
                    continue;
                }
                int left_index = (i*imgWidth + j)*3;
                float temp = 0.0;
                for(int s=0; s<ksize; s++) {
                    int cur_index = left_index + s*imgWidth;
                    for(int s1=0; s1<ksize; s1++) {
                        col[out_index] = inptr[cur_index];
                        out_index++;
                        cur_index++;
                    }

                }

            }
        }
    }
*/

}

void conv3x3_x86_img2col(float *inputdata, int inshape[4], int outc, float *weights,  float *outdata) {
    int n = inshape[0];
    int imgHeight = inshape[1];
    int imgWidth = inshape[2];
    int imgCh = inshape[3];
    int out_h = imgHeight - 2;
    int out_w = imgWidth - 2;
    int out_size = out_h*out_w*imgCh*3*3;
    float *col = new float[out_size];
    int out_index = 0;
    int ksize = 3;
    for(int h=0;h<imgHeight;h++) {
        if(h+ksize>imgHeight) {
            continue;
        }
        for(int w=0;w<imgWidth;w++) {
            if(w+ksize>imgWidth) {
                continue;
            }
            int index = h*imgWidth+w;  //每个特征图当前的位置，从此处将kernelsize的元素展开成一维
            for(int c=0;c<imgCh;c++) {                    
                float *pdata = inputdata + c*imgWidth*imgHeight + index;   //当前通道，当前特征图位置的pos
                for(int s=0; s<ksize; s++) {
                    int cur_index = s*imgWidth;   //当前特征图的vector的新的起始位置
                    for(int s1=0; s1<ksize; s1++) {
                        col[out_index] = pdata[cur_index];
                        out_index++;
                        cur_index++;
                    }

                }
            }
        }
    }
    //计算卷积
    int outw = out_h;
    int outh = out_w;
    int featurelen = outw*outh;
    int outindex=0;
    #pragma omp parallel for num_threads(4)
    for(int i=0;i<outc;i++) {
        float *weight = weights +i*3*3*3;  //当前通道使用的卷积kernal
        for(int h=0;h<outh;h++) {
            for(int w=0;w<outw;w++) {
                float *pdata = col + (h*outw+w)*27; //当前输入数据的位置
                float sum = 0.0f;
                for(int k=0;k<27;k++) {
                 sum +=pdata[k]*weight[k];
                }
                outdata[outindex] = sum;
                outindex++;
            }
        }
    }

}
