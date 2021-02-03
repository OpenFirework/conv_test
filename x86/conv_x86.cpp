#include <iostream>
#include "mat.h"
using namespace std;
//输入数据的排列方式c,h,w
//strid=1, padding=0,
//朴素的实现，直接相乘
int32_t NowMicros();
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

void conv3x3_x86_mat(const Mat& bottom_blob, Mat& top_blob, Mat &weight_data) {
     // param
    int num_output=32;
    int kernel_w=3;
    int kernel_h=3;
    int dilation_w=1;
    int dilation_h=1;
    int stride_w=1;
    int stride_h=1;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    const int maxk = kernel_w * kernel_h;      //3*3
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // float32
    top_blob.create(outw, outh, num_output, elemsize, 1);
    if (top_blob.empty())
        return;
    int index = 0;
    for (int p = 0; p < num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                const float* kptr = (const float*)weight_data + maxk * channels * p;

                // channels,weights的通道内部的kernels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[space_ofs[k]]; // 20.72
                        float wt = kptr[k];
                        sum += val * wt; // 41.45  
                        
                    }
                    kptr += maxk;
                }
                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}


int copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v) {
    int w = src.w;
    int h = src.h;
    int channels = src.c;
    int dims = src.dims;
    size_t elemsize = src.elemsize;

    int outw = w + left + right;
    int outh = h + top + bottom;
    int outc = channels;

    dst.create(outw, outh, outc, elemsize, 1);
    if (dst.empty())
        return -100;
    
    for (int q = 0; q < outc; q++) {
        Mat borderm = dst.channel(q);
        int q_ = q;
        const Mat m = src.channel(q_); 

        int w = dst.w;
        int h = dst.h;

        const float* ptr = src;
        float* outptr = dst;

        if (type == 0)
        {
            int y = 0;
            // fill top
            for (; y < top; y++)
            {
                int x = 0;
                for (; x < outw; x++)
                {
                    outptr[x] = v;
                }
                outptr += outw;
            }
            // fill center
            for (; y < (top + src.h); y++)
            {
                int x = 0;
                for (; x < left; x++)
                {
                    outptr[x] = v;
                }
                if (src.w < 12)
                {
                    for (; x < (left + src.w); x++)
                    {
                        outptr[x] = ptr[x - left];
                    }
                }
                else
                {
                    memcpy(outptr + left, ptr, src.w * sizeof(float));
                    x += src.w;
                }
                for (; x < outw; x++)
                {
                    outptr[x] = v;
                }
                ptr += src.w;
                outptr += outw;
            }
            // fill bottom
            for (; y < outh; y++)
            {
                int x = 0;
                for (; x < outw; x++)
                {
                    outptr[x] = v;
                }
                outptr += outw;
            }
        }

        }
}


int copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right) {
    int woffset = top;
    int hoffset = bottom;
    int coffset = 0;
    int outw = src.w - left - right;
    int outh = src.h - top - bottom;
    int outc = -233;
    int woffset2 = 0;
    int hoffset2 = 0;
    int coffset2 = 0;
    
    Mat bottom_blob = src;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int _woffset, _hoffset, _coffset;
    int _outw = -1, _outh = -1, _outc;

    if (_outw == w && _outh == h && _outc == channels)
    {
        dst = src;
        return 0;
    }

    {
        _woffset = woffset;
        _hoffset = hoffset;
        _coffset = coffset;
        _outw = w;
        _outh = h;
        _outc = channels;

        if (dims == 1)
        {
            _outw = w - woffset - woffset2;
            if (outw != -233)
                _outw = std::min(outw, _outw);
        }
        if (dims == 2)
        {
            if (hoffset == -233)
            {
                _woffset = 0;
                _hoffset = woffset;

                _outw = w;
                _outh = h - woffset - woffset2;
                if (outw != -233)
                    _outh = std::min(outw, _outh);
            }
            else
            {
                _outw = w - woffset - woffset2;
                if (outw != -233)
                    _outw = std::min(outw, _outw);

                _outh = h - hoffset - hoffset2;
                if (outh != -233)
                    _outh = std::min(outh, _outh);
            }
        }
        if (dims == 3)
        {
            if (hoffset == -233 && coffset == -233)
            {
                _woffset = 0;
                _hoffset = 0;

                _outw = w;
                _outh = h;
                _outc = channels - woffset - woffset2;
                if (outw != -233)
                    _outc = std::min(outw, _outc);
            }
            else if (coffset == -233)
            {
                _woffset = 0;

                _outw = w;
                _outh = h - woffset - woffset2;
                if (outw != -233)
                    _outh = std::min(outw, _outh);

                _outc = channels - hoffset - hoffset2;
                if (outh != -233)
                    _outc = std::min(outh, _outc);
            }
            else
            {
                _outw = w - woffset - woffset2;
                if (outw != -233)
                    _outw = std::min(outw, _outw);

                _outh = h - hoffset - hoffset2;
                if (outh != -233)
                    _outh = std::min(outh, _outh);

                _outc = channels - coffset - coffset2;
                if (outc != -233)
                    _outc = std::min(outc, _outc);
            }
        }
    }

    dst.create(_outw, _outh, _outc, elemsize, 1);
    if (dst.empty())
        return -100;

    w = dst.w;
    h = dst.h;

    const float* ptr = src.row<float>(top) + left;
    float* outptr = dst; //.data;

    for (int y = 0; y < h; y++)
    {
        if (w < 12)
        {
            for (int x = 0; x < w; x++)
            {
                outptr[x] = ptr[x];
            }
        }
        else
        {
            memcpy(outptr, ptr, w * sizeof(float));
        }
        outptr += w;
        ptr += src.w;
    }
}


void conv3x3_winograd23_transform_kernel_my(const Mat& kernel, Mat& kernel_tm, int inch, int outch) {
    kernel_tm.create(4 * 4, inch, outch, 4u, 1);
    // G
    const float ktm[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {1.0f / 2, 1.0f / 2, 1.0f / 2},
        {1.0f / 2, -1.0f / 2, 1.0f / 2},
        {0.0f, 0.0f, 1.0f}
    };
    
    for(int p=0; p<outch; p++) {
        for(int q=0; q<inch; q++) {
            float *weights = (float*)(kernel.data) + p*inch*9 + q*9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

              // transform kernel
            const float* k0 = weights;
            const float* k1 = weights + 3;
            const float* k2 = weights + 6;

             // h
            float tmp[4][3];
            for (int i = 0; i < 4; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }
            
            // U
            for (int j = 0; j < 4; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 4; i++)
                {
                    kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
}


void conv3x3s1_winograd23_my(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm) {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 2n+2, winograd F(2,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
 //   bottom_blob_bordered.create(w, h, bottom_blob.c, (size_t)(sizeof(float)), 1);
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f);


    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4 * 4, tiles, inch, 4u, 1);

        // BT
        // const float itm[4][4] = {
        //     {1.0f,  0.0f, -1.0f,  0.0f},
        //     {0.0f,  1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  0.00f, 1.0f}
        // };
        for (int q = 0; q < inch; q++)
        {
            const float* img = bottom_blob_bordered.channel(q);
            float* out_tm0 = bottom_blob_tm.channel(q);

            for (int j = 0; j < nColBlocks; j++)
            {
                const float* r0 = img + w * j * 2;
                const float* r1 = r0 + w;
                const float* r2 = r1 + w;
                const float* r3 = r2 + w;

                for (int i = 0; i < nRowBlocks; i++)
                {
                    float d0[4], d1[4], d2[4], d3[4];
                    float w0[4], w1[4], w2[4], w3[4];
                    float t0[4], t1[4], t2[4], t3[4];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                    }
                    // w = B_t * d
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = d0[n] - d2[n];
                        w1[n] = d1[n] + d2[n];
                        w2[n] = d2[n] - d1[n];
                        w3[n] = d3[n] - d1[n];
                    }
                    // transpose d to d_t
                    {
                        t0[0] = w0[0];
                        t1[0] = w0[1];
                        t2[0] = w0[2];
                        t3[0] = w0[3];
                        t0[1] = w1[0];
                        t1[1] = w1[1];
                        t2[1] = w1[2];
                        t3[1] = w1[3];
                        t0[2] = w2[0];
                        t1[2] = w2[1];
                        t2[2] = w2[2];
                        t3[2] = w2[3];
                        t0[3] = w3[0];
                        t1[3] = w3[1];
                        t2[3] = w3[2];
                        t3[3] = w3[3];
                    }
                    // d = B_t * d_t
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = t0[n] - t2[n];
                        d1[n] = t1[n] + t2[n];
                        d2[n] = t2[n] - t1[n];
                        d3[n] = t3[n] - t1[n];
                    }
                    // save to out_tm
                    for (int n = 0; n < 4; n++)
                    {
                        out_tm0[n] = d0[n];
                        out_tm0[n + 4] = d1[n];
                        out_tm0[n + 8] = d2[n];
                        out_tm0[n + 12] = d3[n];
                    }
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;

                    out_tm0 += 16;
                }
            }
        }
    }
    bottom_blob_bordered = Mat();

        // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        top_blob_tm.create(16, tiles, outch, 4u,1);

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p + 1);
            Mat out2_tm = top_blob_tm.channel(p + 2);
            Mat out3_tm = top_blob_tm.channel(p + 3);

            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p + 1);
            const Mat kernel2_tm = kernel_tm.channel(p + 2);
            const Mat kernel3_tm = kernel_tm.channel(p + 3);

            for (int i = 0; i < tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);
                float* output1_tm = out1_tm.row(i);
                float* output2_tm = out2_tm.row(i);
                float* output3_tm = out3_tm.row(i);

                float sum0[16] = { 0.0f };
                float sum1[16] = { 0.0f };
                float sum2[16] = { 0.0f };
                float sum3[16] = { 0.0f };

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q + 1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q + 2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q + 3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r1[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r2[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r3[n] * k0[n];
                        k0 -= 16 * 3;

                        sum1[n] += r0[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r1[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r2[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r3[n] * k1[n];
                        k1 -= 16 * 3;

                        sum2[n] += r0[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r1[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r2[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r3[n] * k2[n];
                        k2 -= 16 * 3;

                        sum3[n] += r0[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r1[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r2[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r3[n] * k3[n];
                        k3 -= 16 * 3;
                    }
                }

                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum1[n] += r0[n] * k1[n];
                        sum2[n] += r0[n] * k2[n];
                        sum3[n] += r0[n] * k3[n];
                    }
                }

                for (int n = 0; n < 16; n++)
                {
                    output0_tm[n] = sum0[n];
                    output1_tm[n] = sum1[n];
                    output2_tm[n] = sum2[n];
                    output3_tm[n] = sum3[n];
                }
            }
        }
        for (int p = remain_outch_start; p < outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int i = 0; i < tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);

                float sum0[16] = { 0.0f };

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q + 1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q + 2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q + 3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel0_tm.row(q + 1);
                    const float* k2 = kernel0_tm.row(q + 2);
                    const float* k3 = kernel0_tm.row(q + 3);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum0[n] += r1[n] * k1[n];
                        sum0[n] += r2[n] * k2[n];
                        sum0[n] += r3[n] * k3[n];
                    }
                }

                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* k0 = kernel0_tm.row(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                    }
                }

                for (int n = 0; n < 16; n++)
                {
                    output0_tm[n] = sum0[n];
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

     Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 4u, 1);
    }
    {
        // AT
        // const float itm[2][4] = {
        //     {1.0f,  1.0f,  1.0f,  0.0f},
        //     {0.0f,  1.0f, -1.0f,  1.0f}
        // };

        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;
        for (int p = 0; p < outch; p++)
        {
            Mat out_tm = top_blob_tm.channel(p);
            Mat out = top_blob_bordered.channel(p);

           // const float bias0 = bias ? bias[p] : 0.f;

            for (int j = 0; j < nColBlocks; j++)
            {
                float* outRow0 = out.row(j * 2);
                float* outRow1 = out.row(j * 2 + 1);

                for (int i = 0; i < nRowBlocks; i++)
                {
                    float* out_tile = out_tm.row(j * nRowBlocks + i);

                    float s0[4], s1[4], s2[4], s3[4];
                    float w0[4], w1[4];
                    float d0[2], d1[2], d2[2], d3[2];
                    float o0[2], o1[2];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n + 4];
                        s2[n] = out_tile[n + 8];
                        s3[n] = out_tile[n + 12];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n];
                        w1[n] = s1[n] - s2[n] + s3[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0];
                        d0[1] = w1[0];
                        d1[0] = w0[1];
                        d1[1] = w1[1];
                        d2[0] = w0[2];
                        d2[1] = w1[2];
                        d3[0] = w0[3];
                        d3[1] = w1[3];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 2; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n];
                        o1[n] = d1[n] - d2[n] + d3[n];
                    }
                    // save to top blob tm
                    outRow0[0] = o0[0];
                    outRow0[1] = o0[1];
                    outRow1[0] = o1[0];
                    outRow1[1] = o1[1];

                    outRow0 += 2;
                    outRow1 += 2;
                }
            }
        }
    }
    // END transform output
    //top_blob = top_blob_bordered;
    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w);
}


void conv3x3s1_winograd23_omp_my(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm) {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 2n+2, winograd F(2,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4 * 4, tiles, inch, 4u, 1);

        // BT
        // const float itm[4][4] = {
        //     {1.0f,  0.0f, -1.0f,  0.0f},
        //     {0.0f,  1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  0.00f, 1.0f}
        // };
        #pragma omp parallel for num_threads(2)
        for (int q = 0; q < inch; q++)
        {
            const float* img = bottom_blob_bordered.channel(q);
            float* out_tm0 = bottom_blob_tm.channel(q);

            for (int j = 0; j < nColBlocks; j++)
            {
                const float* r0 = img + w * j * 2;
                const float* r1 = r0 + w;
                const float* r2 = r1 + w;
                const float* r3 = r2 + w;

                for (int i = 0; i < nRowBlocks; i++)
                {
                    float d0[4], d1[4], d2[4], d3[4];
                    float w0[4], w1[4], w2[4], w3[4];
                    float t0[4], t1[4], t2[4], t3[4];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                    }
                    // w = B_t * d
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = d0[n] - d2[n];
                        w1[n] = d1[n] + d2[n];
                        w2[n] = d2[n] - d1[n];
                        w3[n] = d3[n] - d1[n];
                    }
                    // transpose d to d_t
                    {
                        t0[0] = w0[0];
                        t1[0] = w0[1];
                        t2[0] = w0[2];
                        t3[0] = w0[3];
                        t0[1] = w1[0];
                        t1[1] = w1[1];
                        t2[1] = w1[2];
                        t3[1] = w1[3];
                        t0[2] = w2[0];
                        t1[2] = w2[1];
                        t2[2] = w2[2];
                        t3[2] = w2[3];
                        t0[3] = w3[0];
                        t1[3] = w3[1];
                        t2[3] = w3[2];
                        t3[3] = w3[3];
                    }
                    // d = B_t * d_t
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = t0[n] - t2[n];
                        d1[n] = t1[n] + t2[n];
                        d2[n] = t2[n] - t1[n];
                        d3[n] = t3[n] - t1[n];
                    }
                    // save to out_tm
                    for (int n = 0; n < 4; n++)
                    {
                        out_tm0[n] = d0[n];
                        out_tm0[n + 4] = d1[n];
                        out_tm0[n + 8] = d2[n];
                        out_tm0[n + 12] = d3[n];
                    }
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;

                    out_tm0 += 16;
                }
            }
        }
    }
    bottom_blob_bordered = Mat();

        // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;

        top_blob_tm.create(16, tiles, outch, 4u,1);

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;
                
        #pragma omp parallel for num_threads(2)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p + 1);
            Mat out2_tm = top_blob_tm.channel(p + 2);
            Mat out3_tm = top_blob_tm.channel(p + 3);

            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p + 1);
            const Mat kernel2_tm = kernel_tm.channel(p + 2);
            const Mat kernel3_tm = kernel_tm.channel(p + 3);

            for (int i = 0; i < tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);
                float* output1_tm = out1_tm.row(i);
                float* output2_tm = out2_tm.row(i);
                float* output3_tm = out3_tm.row(i);

                float sum0[16] = { 0.0f };
                float sum1[16] = { 0.0f };
                float sum2[16] = { 0.0f };
                float sum3[16] = { 0.0f };

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q + 1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q + 2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q + 3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r1[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r2[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r3[n] * k0[n];
                        k0 -= 16 * 3;

                        sum1[n] += r0[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r1[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r2[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r3[n] * k1[n];
                        k1 -= 16 * 3;

                        sum2[n] += r0[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r1[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r2[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r3[n] * k2[n];
                        k2 -= 16 * 3;

                        sum3[n] += r0[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r1[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r2[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r3[n] * k3[n];
                        k3 -= 16 * 3;
                    }
                }

                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum1[n] += r0[n] * k1[n];
                        sum2[n] += r0[n] * k2[n];
                        sum3[n] += r0[n] * k3[n];
                    }
                }

                for (int n = 0; n < 16; n++)
                {
                    output0_tm[n] = sum0[n];
                    output1_tm[n] = sum1[n];
                    output2_tm[n] = sum2[n];
                    output3_tm[n] = sum3[n];
                }
            }
        }

        #pragma omp parallel for num_threads(2)
        for (int p = remain_outch_start; p < outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int i = 0; i < tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);

                float sum0[16] = { 0.0f };

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q + 1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q + 2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q + 3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel0_tm.row(q + 1);
                    const float* k2 = kernel0_tm.row(q + 2);
                    const float* k3 = kernel0_tm.row(q + 3);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum0[n] += r1[n] * k1[n];
                        sum0[n] += r2[n] * k2[n];
                        sum0[n] += r3[n] * k3[n];
                    }
                }

                for (; q < inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* k0 = kernel0_tm.row(q);

                    for (int n = 0; n < 16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                    }
                }

                for (int n = 0; n < 16; n++)
                {
                    output0_tm[n] = sum0[n];
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

     Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, 4u, 1);
    }
    {
        // AT
        // const float itm[2][4] = {
        //     {1.0f,  1.0f,  1.0f,  0.0f},
        //     {0.0f,  1.0f, -1.0f,  1.0f}
        // };

        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;
                
        #pragma omp parallel for num_threads(2)
        for (int p = 0; p < outch; p++)
        {
            Mat out_tm = top_blob_tm.channel(p);
            Mat out = top_blob_bordered.channel(p);

           // const float bias0 = bias ? bias[p] : 0.f;

            for (int j = 0; j < nColBlocks; j++)
            {
                float* outRow0 = out.row(j * 2);
                float* outRow1 = out.row(j * 2 + 1);

                for (int i = 0; i < nRowBlocks; i++)
                {
                    float* out_tile = out_tm.row(j * nRowBlocks + i);

                    float s0[4], s1[4], s2[4], s3[4];
                    float w0[4], w1[4];
                    float d0[2], d1[2], d2[2], d3[2];
                    float o0[2], o1[2];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n + 4];
                        s2[n] = out_tile[n + 8];
                        s3[n] = out_tile[n + 12];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n];
                        w1[n] = s1[n] - s2[n] + s3[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0];
                        d0[1] = w1[0];
                        d1[0] = w0[1];
                        d1[1] = w1[1];
                        d2[0] = w0[2];
                        d2[1] = w1[2];
                        d3[0] = w0[3];
                        d3[1] = w1[3];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 2; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n];
                        o1[n] = d1[n] - d2[n] + d3[n];
                    }
                    // save to top blob tm
                    outRow0[0] = o0[0];
                    outRow0[1] = o0[1];
                    outRow1[0] = o1[0];
                    outRow1[1] = o1[1];

                    outRow0 += 2;
                    outRow1 += 2;
                }
            }
        }
    }
    // END transform output
    top_blob = top_blob_bordered;
    // cut result pad
   // copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w);
}

