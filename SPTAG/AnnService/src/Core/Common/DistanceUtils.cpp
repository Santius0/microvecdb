// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/DistanceUtils.h"

using namespace SPTAG;
using namespace SPTAG::COMMON;

#ifndef _MSC_VER
#define DIFF128 diff128
#define DIFF256 diff256
#else
#define DIFF128 diff128.m128_f32
#define DIFF256 diff256.m256_f32
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
inline __m128 _mm_mul_epi8(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(xlo, ylo), _mm_madd_epi16(xhi, yhi)));
}

inline __m128 _mm_sqdf_epi8(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
}

inline __m128 _mm_mul_epu8(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i xlo = _mm_unpacklo_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, zero);
    __m128i ylo = _mm_unpacklo_epi8(Y, zero);
    __m128i yhi = _mm_unpackhi_epi8(Y, zero);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(xlo, ylo), _mm_madd_epi16(xhi, yhi)));
}

inline __m128 _mm_sqdf_epu8(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i xlo = _mm_unpacklo_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, zero);
    __m128i ylo = _mm_unpacklo_epi8(Y, zero);
    __m128i yhi = _mm_unpackhi_epi8(Y, zero);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
}

inline __m128 _mm_mul_epi16(__m128i X, __m128i Y)
{
    return _mm_cvtepi32_ps(_mm_madd_epi16(X, Y));
}

inline __m128 _mm_sqdf_epi16(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi16(X, zero);
    __m128i sign_y = _mm_cmplt_epi16(Y, zero);

    __m128i xlo = _mm_unpacklo_epi16(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi16(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi16(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi16(Y, sign_y);

    __m128 dlo = _mm_cvtepi32_ps(_mm_sub_epi32(xlo, ylo));
    __m128 dhi = _mm_cvtepi32_ps(_mm_sub_epi32(xhi, yhi));

    return _mm_add_ps(_mm_mul_ps(dlo, dlo), _mm_mul_ps(dhi, dhi));
}

inline __m128 _mm_sqdf_ps(__m128 X, __m128 Y)
{
    __m128 d = _mm_sub_ps(X, Y);
    return _mm_mul_ps(d, d);
}

inline __m256 _mm256_mul_epi8(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, ylo), _mm256_madd_epi16(xhi, yhi)));
}

inline __m256 _mm256_sqdf_epi8(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    __m256i dhi = _mm256_sub_epi16(xhi, yhi);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(dlo, dlo), _mm256_madd_epi16(dhi, dhi)));
}

inline __m256 _mm256_mul_epu8(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i xlo = _mm256_unpacklo_epi8(X, zero);
    __m256i xhi = _mm256_unpackhi_epi8(X, zero);
    __m256i ylo = _mm256_unpacklo_epi8(Y, zero);
    __m256i yhi = _mm256_unpackhi_epi8(Y, zero);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, ylo), _mm256_madd_epi16(xhi, yhi)));
}

inline __m256 _mm256_sqdf_epu8(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i xlo = _mm256_unpacklo_epi8(X, zero);
    __m256i xhi = _mm256_unpackhi_epi8(X, zero);
    __m256i ylo = _mm256_unpacklo_epi8(Y, zero);
    __m256i yhi = _mm256_unpackhi_epi8(Y, zero);

    __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    __m256i dhi = _mm256_sub_epi16(xhi, yhi);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(dlo, dlo), _mm256_madd_epi16(dhi, dhi)));
}

inline __m256 _mm256_mul_epi16(__m256i X, __m256i Y)
{
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(X, Y));
}

inline __m256 _mm256_sqdf_epi16(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi16(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi16(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi16(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi16(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi16(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi16(Y, sign_y);

    __m256 dlo = _mm256_cvtepi32_ps(_mm256_sub_epi32(xlo, ylo));
    __m256 dhi = _mm256_cvtepi32_ps(_mm256_sub_epi32(xhi, yhi));

    return _mm256_add_ps(_mm256_mul_ps(dlo, dlo), _mm256_mul_ps(dhi, dhi));
}

inline __m256 _mm256_sqdf_ps(__m256 X, __m256 Y)
{
    __m256 d = _mm256_sub_ps(X, Y);
    return _mm256_mul_ps(d, d);
}

// Do not use intrinsics not supported by old MS compiler version
#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
inline __m512 _mm512_mul_epi8(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __mmask64 sign_x_mask = _mm512_cmpgt_epi8_mask (zero, X);
    __mmask64 sign_y_mask = _mm512_cmpgt_epi8_mask (zero, Y);

    __m512i sign_x = _mm512_movm_epi8(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi8(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi8(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi8(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi8(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi8(Y, sign_y);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(xlo, ylo), _mm512_madd_epi16(xhi, yhi)));
}

inline __m512 _mm512_sqdf_epi8(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __mmask64 sign_x_mask = _mm512_cmpgt_epi8_mask (zero, X);
    __mmask64 sign_y_mask = _mm512_cmpgt_epi8_mask (zero, Y);

    __m512i sign_x = _mm512_movm_epi8(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi8(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi8(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi8(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi8(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi8(Y, sign_y);

    __m512i dlo = _mm512_sub_epi16(xlo, ylo);
    __m512i dhi = _mm512_sub_epi16(xhi, yhi);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(dlo, dlo), _mm512_madd_epi16(dhi, dhi)));
}

inline __m512 _mm512_mul_epu8(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __m512i xlo = _mm512_unpacklo_epi8(X, zero);
    __m512i xhi = _mm512_unpackhi_epi8(X, zero);
    __m512i ylo = _mm512_unpacklo_epi8(Y, zero);
    __m512i yhi = _mm512_unpackhi_epi8(Y, zero);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(xlo, ylo), _mm512_madd_epi16(xhi, yhi)));
}

inline __m512 _mm512_sqdf_epu8(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __m512i xlo = _mm512_unpacklo_epi8(X, zero);
    __m512i xhi = _mm512_unpackhi_epi8(X, zero);
    __m512i ylo = _mm512_unpacklo_epi8(Y, zero);
    __m512i yhi = _mm512_unpackhi_epi8(Y, zero);

    __m512i dlo = _mm512_sub_epi16(xlo, ylo);
    __m512i dhi = _mm512_sub_epi16(xhi, yhi);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(dlo, dlo), _mm512_madd_epi16(dhi, dhi)));
}

inline __m512 _mm512_mul_epi16(__m512i X, __m512i Y)
{
    return _mm512_cvtepi32_ps(_mm512_madd_epi16(X, Y));
}

inline __m512 _mm512_sqdf_epi16(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __mmask32 sign_x_mask = _mm512_cmpgt_epi16_mask (zero, X);
    __mmask32 sign_y_mask = _mm512_cmpgt_epi16_mask (zero, Y);

    __m512i sign_x = _mm512_movm_epi16(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi16(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi16(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi16(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi16(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi16(Y, sign_y);

    __m512 dlo = _mm512_cvtepi32_ps(_mm512_sub_epi32(xlo, ylo));
    __m512 dhi = _mm512_cvtepi32_ps(_mm512_sub_epi32(xhi, yhi));

    return _mm512_add_ps(_mm512_mul_ps(dlo, dlo), _mm512_mul_ps(dhi, dhi));
}

inline __m512 _mm512_sqdf_ps(__m512 X, __m512 Y)
{
    __m512 d = _mm512_sub_ps(X, Y);
    return _mm512_mul_ps(d, d);
}
#endif


#define REPEAT(type, ctype, delta, load, exec, acc, result) \
            { \
                type c1 = load((ctype *)(pX)); \
                type c2 = load((ctype *)(pY)); \
                pX += delta; pY += delta; \
                result = acc(result, exec(c1, c2)); \
            } \

float DistanceUtils::ComputeL2Distance_SSE(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_sqdf_epi8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}


float DistanceUtils::ComputeL2Distance_SSE(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::uint8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_sqdf_epu8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}


float DistanceUtils::ComputeL2Distance_SSE(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
    }
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }

    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_sqdf_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }

    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int16_t* pEnd32 = pX + ((length >> 5) << 5);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m512i, __m512i, 32, _mm512_loadu_si512, _mm512_sqdf_epi16, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_sqdf_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }

    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}


float DistanceUtils::ComputeL2Distance_SSE(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd16 = pX + ((length >> 4) << 4);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd16)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) {
        float c1 = (*pX++) - (*pY++); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd16 = pX + ((length >> 4) << 4);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd16)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_sqdf_ps, _mm256_add_ps, diff256)
            REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_sqdf_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) {
        float c1 = (*pX++) - (*pY++); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd8 = pX + ((length >> 3) << 3);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const float* pEnd16 = pX + ((length >> 4) << 4);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m512, const float, 16, _mm512_loadu_ps, _mm512_sqdf_ps, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd8)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_sqdf_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) {
        float c1 = (*pX++) - (*pY++); diff += c1 * c1;
    }
    return diff;
}


float DistanceUtils::ComputeCosineDistance_SSE(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
    }
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 16129 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 16129 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_mul_epi8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 16129 - diff;
}


float DistanceUtils::ComputeCosineDistance_SSE(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
    }
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 65025 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 65025 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::uint8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_mul_epu8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 65025 - diff;
}


float DistanceUtils::ComputeCosineDistance_SSE(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
            REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
    }
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }

    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return  1073676289 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_mul_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }

    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return  1073676289 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int16_t* pEnd32 = pX + ((length >> 5) << 5);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m512i, __m512i, 32, _mm512_loadu_si512, _mm512_mul_epi16, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_mul_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }

    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return  1073676289 - diff;
}


float DistanceUtils::ComputeCosineDistance_SSE(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd16 = pX + ((length >> 4) << 4);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    __m128 diff128 = _mm_setzero_ps();
    while (pX < pEnd16)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
            REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
    }
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) diff += (*pX++) * (*pY++);
    return 1 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd16 = pX + ((length >> 4) << 4);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    __m256 diff256 = _mm256_setzero_ps();
    while (pX < pEnd16)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_mul_ps, _mm256_add_ps, diff256)
            REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_mul_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) diff += (*pX++) * (*pY++);
    return 1 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd8 = pX + ((length >> 3) << 3);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const float* pEnd16 = pX + ((length >> 4) << 4);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m512, const float, 16, _mm512_loadu_ps, _mm512_mul_ps, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd8)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_mul_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) diff += (*pX++) * (*pY++);
    return 1 - diff;
}


#elif defined(__arm__) || defined(__aarch64__)
float DistanceUtils::ComputeL2Distance_NEON(const std::int8_t* pX, const std::int8_t* pY, DimensionType length) {
    const std::int8_t* pEnd16 = pX + (length & ~15); // Align to process blocks of 16
    const std::int8_t* pEnd1 = pX + length;

    float32x4_t diffVec = vdupq_n_f32(0.0f); // Accumulator vector for differences

    // Process in blocks of 16
    while (pX < pEnd16) {
        int8x16_t xVec = vld1q_s8(pX); // Load 16 elements from pX
        int8x16_t yVec = vld1q_s8(pY); // Load 16 elements from pY

        // Calculate the difference
        int16x8_t diffLow = vsubl_s8(vget_low_s8(xVec), vget_low_s8(yVec));
        int16x8_t diffHigh = vsubl_s8(vget_high_s8(xVec), vget_high_s8(yVec));

        // Calculate the square of the differences and accumulate
        int32x4_t diffLowLow = vmull_s16(vget_low_s16(diffLow), vget_low_s16(diffLow));
        int32x4_t diffHighLow = vmull_s16(vget_high_s16(diffLow), vget_high_s16(diffLow));
        int32x4_t diffLowHigh = vmull_s16(vget_low_s16(diffHigh), vget_low_s16(diffHigh));
        int32x4_t diffHighHigh = vmull_s16(vget_high_s16(diffHigh), vget_high_s16(diffHigh));

        diffVec = vaddq_f32(diffVec, vcvtq_f32_s32(diffLowLow));
        diffVec = vaddq_f32(diffVec, vcvtq_f32_s32(diffHighLow));
        diffVec = vaddq_f32(diffVec, vcvtq_f32_s32(diffLowHigh));
        diffVec = vaddq_f32(diffVec, vcvtq_f32_s32(diffHighHigh));

        pX += 16;
        pY += 16;
    }

    // Final reduction to a single float
    float diff = vaddvq_f32(diffVec); // Sum all elements of vector to a single scalar

    // Process remaining elements one by one
    while (pX < pEnd1) {
        float c1 = static_cast<float>(*pX++) - static_cast<float>(*pY++);
        diff += c1 * c1;
    }

    return diff;
} // + => 03/05/24

float DistanceUtils::ComputeL2Distance_NEON(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length) {  // + => 03/05/24
    const std::uint8_t* pEnd16 = pX + (length & ~15); // Process blocks of 16
    const std::uint8_t* pEnd1 = pX + length;

    float32x4_t diffVec = vdupq_n_f32(0.0); // Accumulator vector for differences

    // Process in blocks of 16
    while (pX < pEnd16) {
        uint8x16_t xVec = vld1q_u8(pX); // Load 16 elements from pX
        uint8x16_t yVec = vld1q_u8(pY); // Load 16 elements from pY

        uint16x8_t xLow = vmovl_u8(vget_low_u8(xVec)); // Extend lower 8 elements to 16 bits
        uint16x8_t yLow = vmovl_u8(vget_low_u8(yVec));

        uint16x8_t xHigh = vmovl_u8(vget_high_u8(xVec)); // Extend higher 8 elements to 16 bits
        uint16x8_t yHigh = vmovl_u8(vget_high_u8(yVec));

        int32x4_t diffLowLow = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vabdq_u16(xLow, yLow))));
        int32x4_t diffHighLow = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vabdq_u16(xLow, yLow))));
        int32x4_t diffLowHigh = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vabdq_u16(xHigh, yHigh))));
        int32x4_t diffHighHigh = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vabdq_u16(xHigh, yHigh))));

        diffVec = vaddq_f32(diffVec, vcvtq_f32_s32(diffLowLow));
        diffVec = vaddq_f32(diffVec, vcvtq_f32_s32(diffHighLow));
        diffVec = vaddq_f32(diffVec, vcvtq_f32_s32(diffLowHigh));
        diffVec = vaddq_f32(diffVec, vcvtq_f32_s32(diffHighHigh));

        pX += 16;
        pY += 16;
    }

    // Final reduction to single float
    float diff = vaddvq_f32(diffVec); // Sum all elements of vector to single scalar

    // Process remaining elements one by one
    while (pX < pEnd1) {
        float c1 = (float)(*pX++) - (float)(*pY++);
        diff += c1 * c1;
    }

    return diff;
}  // + => 03/05/24

float DistanceUtils::ComputeL2Distance_NEON(const std::int16_t* pX, const std::int16_t* pY, DimensionType length) {
    const std::int16_t* pEnd8 = pX + (length & ~7); // Align to process blocks of 8
    const std::int16_t* pEnd1 = pX + length;

    float32x4_t diffVec = vdupq_n_f32(0.0f); // Accumulator vector for differences

    // Process blocks of 8
    while (pX < pEnd8) {
        int16x8_t xVec = vld1q_s16(pX); // Load 8 int16 elements from pX
        int16x8_t yVec = vld1q_s16(pY); // Load 8 int16 elements from pY

        int16x8_t diff = vsubq_s16(xVec, yVec); // Compute the difference of the vectors

        // Convert differences to 32-bit integers before squaring to prevent overflow
        int32x4_t diff_lo = vmovl_s16(vget_low_s16(diff));
        int32x4_t diff_hi = vmovl_s16(vget_high_s16(diff));

        // Convert squared differences to float and accumulate
        diffVec = vmlaq_f32(diffVec, vcvtq_f32_s32(diff_lo), vcvtq_f32_s32(diff_lo));
        diffVec = vmlaq_f32(diffVec, vcvtq_f32_s32(diff_hi), vcvtq_f32_s32(diff_hi));

        pX += 8;
        pY += 8;
    }

    // Reduce the vector sum into a single float
    float diff = vaddvq_f32(diffVec);

    // Handle any remaining elements
    while (pX < pEnd1) {
        float c1 = (float)(*pX++) - (float)(*pY++);
        diff += c1 * c1;
    }

    return diff;
}  // + => 03/05/24

float DistanceUtils::ComputeL2Distance_NEON(const float* pX, const float* pY, DimensionType length) {  // + => 03/05/24
    const float* pEnd4 = pX + (length & ~3); // Align to process blocks of 4
    const float* pEnd1 = pX + length;

    float32x4_t diffVec = vdupq_n_f32(0.0f); // Accumulator vector for squared differences

    // Process blocks of 4
    while (pX < pEnd4) {
        float32x4_t xVec = vld1q_f32(pX); // Load 4 float elements from pX
        float32x4_t yVec = vld1q_f32(pY); // Load 4 float elements from pY

        float32x4_t diff = vsubq_f32(xVec, yVec); // Compute the difference of the vectors
        diffVec = vmlaq_f32(diffVec, diff, diff); // Square the differences and accumulate

        pX += 4;
        pY += 4;
    }

    // Reduce the vector sum into a single float
    float diff = vaddvq_f32(diffVec);

    // Handle any remaining elements
    while (pX < pEnd1) {
        float c1 = (*pX++) - (*pY++);
        diff += c1 * c1;
    }

    return diff;
}  // + => 03/05/24

float DistanceUtils::ComputeCosineDistance_NEON(const std::int8_t* pX, const std::int8_t* pY, DimensionType length) {  // + => 03/05/24
    const std::int8_t* pEnd16 = pX + (length & ~15); // Process blocks of 16
    const std::int8_t* pEnd1 = pX + length;

    int32x4_t accVec = vdupq_n_s32(0); // Accumulator for the dot products

    // Process in blocks of 16
    while (pX < pEnd16) {
        int8x16_t xVec = vld1q_s8(pX); // Load 16 elements from pX
        int8x16_t yVec = vld1q_s8(pY); // Load 16 elements from pY

        // Extend the 8-bit integers to 16-bit integers to prevent overflow during multiplication
        int16x8_t xVec_low = vmovl_s8(vget_low_s8(xVec));
        int16x8_t yVec_low = vmovl_s8(vget_low_s8(yVec));
        int16x8_t xVec_high = vmovl_s8(vget_high_s8(xVec));
        int16x8_t yVec_high = vmovl_s8(vget_high_s8(yVec));

        // Perform multiplication
        int32x4_t prod_low = vmull_s16(vget_low_s16(xVec_low), vget_low_s16(yVec_low));
        int32x4_t prod_high = vmull_s16(vget_high_s16(xVec_low), vget_high_s16(yVec_low));
        accVec = vaddq_s32(accVec, prod_low);
        accVec = vaddq_s32(accVec, prod_high);

        prod_low = vmull_s16(vget_low_s16(xVec_high), vget_low_s16(yVec_high));
        prod_high = vmull_s16(vget_high_s16(xVec_high), vget_high_s16(yVec_high));
        accVec = vaddq_s32(accVec, prod_low);
        accVec = vaddq_s32(accVec, prod_high);

        pX += 16;
        pY += 16;
    }

    // Reduce the accumulated vector to a single integer
    int32_t dot_product = vaddvq_s32(accVec);

    // Handle any remaining elements
    while (pX < pEnd1) {
        dot_product += (*pX++) * (*pY++);
    }

    // Subtract the accumulated dot product from a constant, similar to the AVX version
    return 16129 - dot_product;
}  // + => 03/05/24
float DistanceUtils::ComputeCosineDistance_NEON(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length) {  // + => 03/05/24
    const std::uint8_t* pEnd16 = pX + (length & ~15); // Process blocks of 16
    const std::uint8_t* pEnd1 = pX + length;

    uint32x4_t accVec = vdupq_n_u32(0); // Accumulator for the dot products

    // Process in blocks of 16
    while (pX < pEnd16) {
        uint8x16_t xVec = vld1q_u8(pX); // Load 16 elements from pX
        uint8x16_t yVec = vld1q_u8(pY); // Load 16 elements from pY

        // Convert the 8-bit integers to 16-bit to prevent overflow during multiplication
        uint16x8_t xVec_low = vmovl_u8(vget_low_u8(xVec));
        uint16x8_t yVec_low = vmovl_u8(vget_low_u8(yVec));
        uint16x8_t xVec_high = vmovl_u8(vget_high_u8(xVec));
        uint16x8_t yVec_high = vmovl_u8(vget_high_u8(yVec));

        // Perform multiplication
        uint32x4_t prod_low = vmull_u16(vget_low_u16(xVec_low), vget_low_u16(yVec_low));
        uint32x4_t prod_high = vmull_u16(vget_high_u16(xVec_low), vget_high_u16(yVec_low));
        accVec = vaddq_u32(accVec, prod_low);
        accVec = vaddq_u32(accVec, prod_high);

        prod_low = vmull_u16(vget_low_u16(xVec_high), vget_low_u16(yVec_high));
        prod_high = vmull_u16(vget_high_u16(xVec_high), vget_high_u16(yVec_high));
        accVec = vaddq_u32(accVec, prod_low);
        accVec = vaddq_u32(accVec, prod_high);

        pX += 16;
        pY += 16;
    }

    // Reduce the accumulated vector to a single integer
    uint32_t dot_product = vaddvq_u32(accVec);

    // Handle any remaining elements
    while (pX < pEnd1) {
        dot_product += (*pX++) * (*pY++);
    }

    // Subtract the accumulated dot product from a constant
    return 65025 - dot_product;
}  // + => 03/05/24
float DistanceUtils::ComputeCosineDistance_NEON(const std::int16_t* pX, const std::int16_t* pY, DimensionType length) {  // + => 03/05/24
    const std::int16_t* pEnd16 = pX + (length & ~15); // Process blocks of 16
    const std::int16_t* pEnd1 = pX + length;

    int32x4_t accVec = vdupq_n_s32(0); // Accumulator for the dot products

    // Process in blocks of 16
    while (pX < pEnd16) {
        int16x8_t xVec1 = vld1q_s16(pX); // Load 8 int16 elements from pX
        int16x8_t yVec1 = vld1q_s16(pY); // Load 8 int16 elements from pY
        pX += 8;
        pY += 8;
        int16x8_t xVec2 = vld1q_s16(pX); // Load the next 8 int16 elements from pX
        int16x8_t yVec2 = vld1q_s16(pY); // Load the next 8 int16 elements from pY
        pX += 8;
        pY += 8;

        // Perform multiplication and accumulate
        accVec = vmlal_s16(accVec, vget_low_s16(xVec1), vget_low_s16(yVec1));
        accVec = vmlal_s16(accVec, vget_high_s16(xVec1), vget_high_s16(yVec1));
        accVec = vmlal_s16(accVec, vget_low_s16(xVec2), vget_low_s16(yVec2));
        accVec = vmlal_s16(accVec, vget_high_s16(xVec2), vget_high_s16(yVec2));
    }

    // Reduce the accumulated vector to a single integer
    int32_t dot_product = vaddvq_s32(accVec);

    // Handle any remaining elements
    while (pX < pEnd1) {
        dot_product += (*pX++) * (*pY++);
    }

    // Subtract the accumulated dot product from a constant, analogous to the AVX version
    return 1073676289 - dot_product;
}  // + => 03/05/24

float DistanceUtils::ComputeCosineDistance_NEON(const float* pX, const float* pY, DimensionType length) {  // + => 03/05/24
    const float* pEnd4 = pX + (length & ~3); // Align to process blocks of 4
    const float* pEnd1 = pX + length;

    float32x4_t accVec = vdupq_n_f32(0.0f); // Accumulator for the dot products

    // Process in blocks of 4
    while (pX < pEnd4) {
        float32x4_t xVec = vld1q_f32(pX); // Load 4 float elements from pX
        float32x4_t yVec = vld1q_f32(pY); // Load 4 float elements from pY

        accVec = vmlaq_f32(accVec, xVec, yVec); // Multiply and accumulate

        pX += 4;
        pY += 4;
    }

    // Reduce the accumulated vector to a single float
    float dot_product = vaddvq_f32(accVec);

    // Handle any remaining elements
    while (pX < pEnd1) {
        dot_product += (*pX++) * (*pY++);
    }

    // Subtract the accumulated dot product from 1
    return 1 - dot_product;
}  // + => 03/05/24

#endif