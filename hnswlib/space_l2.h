#pragma once
#include "hnswlib.h"
#include <tuple>

namespace hnswlib {

    static float
    L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
    L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

static inline std::tuple<__m512i, __m512i> u8s_to_i16s(const void * input) {
	//Read 64 bytes, convert into two i16x32s (i.e. 64 int16s).
	auto inc = (const unsigned char*)input;
	__m256i in_0 = _mm256_loadu_si256((__m256i const *) input);
	__m256i in_1 = _mm256_loadu_si256((__m256i const * )(inc + 32));
	__m512i out_0 = _mm512_cvtepu8_epi16(in_0);
	__m512i out_1 = _mm512_cvtepu8_epi16(in_1);

	//uint16_t left_check[32];
	//uint16_t right_check[32];

	//_mm512_storeu_si512(&left_check, out_0);
	//_mm512_storeu_si512(&right_check, out_1);
	//for (auto i = 0; i < 32; i++) {
		//uint16_t orig = *(((unsigned char *)input) + i);
		//if (left_check[i] != orig) {
		//std::cout << "at " << i << ": left " << left_check[i] << " orig: " << orig << std::endl;
			//throw std::runtime_error("bad exraction!");
		//}
	//}
	
	//for (auto i = 0; i < 32; i++) {
		//uint16_t orig = *(((unsigned char *)input) + 32 + i);
		//if (right_check[i] != orig) {
			//throw std::runtime_error("bad exraction!");
		//}
		////std::cout << " right: " << right_check[i] << " orig: " << orig << std::endl;
	//}
	std::tuple<__m512i, __m512i> result(out_0, out_1);

	return result;
}

// Works only for bytes, in multiples of 64!
//TODO hide behind more specific flag than AVX

    	static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr);

	static int
	L2SqrSIMDVNNI_u8_x64(const void * __restrict pVect1v, const void * __restrict pVect2v, const void * __restrict qty_ptr) {
		auto *pVect1 = (const unsigned char *) pVect1v;
		auto *pVect2 = (const unsigned char *) pVect2v;
		size_t qty = *((size_t *) qty_ptr);
		const unsigned char *pEnd1 = pVect1 + qty;

		__m512i sum = _mm512_setzero_epi32();
		__m512i diff;
		std::tuple<__m512i, __m512i> v1, v2;

		//uint16_t lefti[128];
		//uint16_t righti[128];
		auto result = 0;
		//auto offset = 0;
		while (pVect1 < pEnd1) {
			//if (offset > 64) {
				//throw new std::runtime_error("hey!");
			//}
			v1 = u8s_to_i16s(pVect1);
			v2 = u8s_to_i16s(pVect2);
			pVect1 += 64;
			pVect2 += 64;
			auto l0 = std::get<0>(v1);
			auto r0 = std::get<0>(v2);
			//_mm512_storeu_si512(lefti + offset, l0);
			//_mm512_storeu_si512(righti + offset, r0);
			auto l1 = std::get<1>(v1);
			auto r1 = std::get<1>(v2);
			//_mm512_storeu_si512(lefti + (offset + 32), l1);
			//_mm512_storeu_si512(righti + (offset + 32), r1);
			//offset += 64;

		    diff = _mm512_sub_epi16(l0, r0);
		     //sum = _mm512_setzero_epi32();
		    sum = _mm512_dpwssd_epi32(sum, diff, diff);
		    //result += _mm512_reduce_add_epi32(sum);
		    diff = _mm512_sub_epi16(l1, r1);
		     //sum = _mm512_setzero_epi32();
		    sum = _mm512_dpwssd_epi32(sum, diff, diff);
		    //result += _mm512_reduce_add_epi32(sum);
		}
		result = _mm512_reduce_add_epi32(sum);
		//auto expected = L2SqrI(pVect1v, pVect2v, qty_ptr);
		//if (result != expected) {
			//pVect1 = (const unsigned char *) pVect1v;
			//pVect2 = (const unsigned char *) pVect2v;
			//for(auto i = 0; i < 128; i++) {
				//std::cout << "lc " << +*(pVect1 + i) << " li " << lefti[i] 
					//<< " rc " << +*(pVect2 + i) << " ri "<< righti[i] << std::endl;
			//}
			//std::cout << "Expected  " << expected << ", got " << result << std::endl;
			//throw std::runtime_error("wrong!");
		//}
		return result;
	}
#elif defined(USE_SSE)

    static float
    L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static float
    L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }
#endif


#ifdef USE_SSE
    static float
    L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);


        size_t qty4 = qty >> 2;

        const float *pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    static float
    L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }
#endif

    class L2Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space(size_t dim) {
            fstdistfunc_ = L2Sqr;
        #if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
        #endif
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2Space() {}
    };

    static int
    L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {

        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        qty = qty >> 2;
        for (size_t i = 0; i < qty; i++) {

            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }

    static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        unsigned char* a = (unsigned char*)pVect1;
        unsigned char* b = (unsigned char*)pVect2;

        for(size_t i = 0; i < qty; i++)
        {
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }

//TODO: SpaceInterface needs 2 parameters, one for result type and one for element type?
    class L2SpaceI : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceI(size_t dim) {
		if (dim % 64 == 0) {
#ifdef USE_AVX
			std::cout << "Using accelerated vnni distance" << std::endl;
			fstdistfunc_ = L2SqrSIMDVNNI_u8_x64;
#else

	                fstdistfunc_ = L2SqrI4x;
#endif
		} else if(dim % 4 == 0) {
                fstdistfunc_ = L2SqrI4x;
            }
            else {
                fstdistfunc_ = L2SqrI;
            }
            dim_ = dim;
            data_size_ = dim * sizeof(unsigned char);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<int> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2SpaceI() {}
    };


}
