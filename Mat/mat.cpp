#include "mat.h"

int convert_packing(const Mat& src, Mat& dst, int out_elempack)
{
    int elempack = src.elempack;

    if (elempack == out_elempack)
    {
        dst = src;
        return 0;
    }

    int w = src.w;
    int h = src.h;
    int channels = src.c;
    int dims = src.dims;
    size_t elemsize = src.elemsize;

    if (dims == 1) {
        if (out_elempack == 1)
        {
            dst = src;
            dst.w = w * elempack;
            dst.cstep = (size_t)w * elempack;
            dst.elemsize = elemsize / elempack;
            dst.elempack = out_elempack;
            return 0;
        }

        int outw = (w * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        dst.create(outw, out_elemsize, out_elempack);
        if (dst.empty())
            return -100;

        memcpy(dst.data, src.data, w * elemsize);

        return 0;
    }

    if (dims == 2)
    {
        int outh = (h * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;
        size_t lane_size = out_elemsize / out_elempack;

        dst.create(w, outh, out_elemsize, out_elempack);
        if (dst.empty())
            return -100;

       
        for (int i = 0; i < outh; i++)
        {
            unsigned char* outptr = (unsigned char*)dst + (size_t)i * w * out_elemsize;

            for (int j = 0; j < w; j++)
            {
                unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                for (int k = 0; k < out_elempack; k++)
                {
                    int srcy = (i * out_elempack + k) / elempack;
                    if (srcy >= h)
                        break;

                    int srck = (i * out_elempack + k) % elempack;

                    const unsigned char* ptr = (const unsigned char*)src + (size_t)srcy * w * elemsize;
                    const unsigned char* elem_ptr = ptr + j * elemsize;

                    memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        int outc = (channels * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;
        size_t lane_size = out_elemsize / out_elempack;
        //std::cout<<"lane_size="<<lane_size<<"\n";
        //
        dst.create(w, h, outc, out_elemsize, out_elempack);
        if (dst.empty())
            return -100;

        for (int q = 0; q < outc; q++)
        {
            Mat out = dst.channel(q);

            for (int i = 0; i < h; i++)
            {
                //当前位置的数据指针
                unsigned char* outptr = (unsigned char*)out + (size_t)i * w * out_elemsize;

                for (int j = 0; j < w; j++)
                {
                    unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                    //如何把原channels中的元素组成新的pack
                    for (int k = 0; k < out_elempack; k++)
                    {
                        int srcq = (q * out_elempack + k) / elempack;
                     //   std::cout<<"srcq="<<srcq<<endl;
                        if (srcq >= channels)
                            break;

                        int srck = (q * out_elempack + k) % elempack;

                        const Mat m = src.channel(srcq);
                        const unsigned char* ptr = (const unsigned char*)m + (size_t)i * w * elemsize;
                        const unsigned char* elem_ptr = ptr + j * elemsize;

                        memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                    }
                }
            }
        }

        return 0;
    }

}
