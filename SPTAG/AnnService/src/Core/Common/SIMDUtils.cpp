// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "inc/Core/Common/SIMDUtils.h"

using namespace SPTAG;
using namespace SPTAG::COMMON;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86) // + => 04/05/2024

void SIMDUtils::ComputeSum_SSE(std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX(std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd1 = pX + length;

     while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX512(std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}


void SIMDUtils::ComputeSum_SSE(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX512(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
     const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi8(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}


void SIMDUtils::ComputeSum_SSE(std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd1 = pX + length;

    while (pX < pEnd8) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi16(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 8;
        pY += 8;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX(std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
     const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd1 = pX + length;

    while (pX < pEnd8) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi16(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 8;
        pY += 8;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX512(std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
     const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd1 = pX + length;

    while (pX < pEnd8) {
        __m128i x_part = _mm_loadu_si128((__m128i*) pX);
        __m128i y_part = _mm_loadu_si128((__m128i*) pY);
        x_part = _mm_add_epi16(x_part, y_part);
        _mm_storeu_si128((__m128i*) pX, x_part);
        pX += 8;
        pY += 8;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}


void SIMDUtils::ComputeSum_SSE(float* pX, const float* pY, DimensionType length)
{
    const float* pEnd4= pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    while (pX < pEnd4) {
        __m128 x_part = _mm_loadu_ps(pX);
        __m128 y_part = _mm_loadu_ps(pY);
        x_part = _mm_add_ps(x_part, y_part);
        _mm_storeu_ps(pX, x_part);
        pX += 4;
        pY += 4;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX(float* pX, const float* pY, DimensionType length)
{
    const float* pEnd4= pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    while (pX < pEnd4) {
        __m128 x_part = _mm_loadu_ps(pX);
        __m128 y_part = _mm_loadu_ps(pY);
        x_part = _mm_add_ps(x_part, y_part);
        _mm_storeu_ps(pX, x_part);
        pX += 4;
        pY += 4;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_AVX512(float* pX, const float* pY, DimensionType length)
{
    const float* pEnd4= pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    while (pX < pEnd4) {
        __m128 x_part = _mm_loadu_ps(pX);
        __m128 y_part = _mm_loadu_ps(pY);
        x_part = _mm_add_ps(x_part, y_part);
        _mm_storeu_ps(pX, x_part);
        pX += 4;
        pY += 4;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}


#elif defined(__arm__) || defined(__aarch64__) // + => 04/05/24

void SIMDUtils::ComputeSum_NEON(std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        int8x16_t x_part = vld1q_s8(pX);
        int8x16_t y_part = vld1q_s8(pY);
        x_part = vaddq_s8(x_part, y_part);
        vst1q_s8(pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_NEON(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd1 = pX + length;

    while (pX < pEnd16) {
        uint8x16_t x_part = vld1q_u8(pX);
        uint8x16_t y_part = vld1q_u8(pY);
        x_part = vaddq_u8(x_part, y_part);
        vst1q_u8(pX, x_part);
        pX += 16;
        pY += 16;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_NEON(std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd1 = pX + length;

    while (pX < pEnd8) {
        int16x8_t x_part = vld1q_s16(pX);
        int16x8_t y_part = vld1q_s16(pY);
        x_part = vaddq_s16(x_part, y_part);
        vst1q_s16(pX, x_part);
        pX += 8;
        pY += 8;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

void SIMDUtils::ComputeSum_NEON(float* pX, const float* pY, DimensionType length)
{
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

    while (pX < pEnd4) {
        float32x4_t x_part = vld1q_f32(pX);
        float32x4_t y_part = vld1q_f32(pY);
        x_part = vaddq_f32(x_part, y_part);
        vst1q_f32(pX, x_part);
        pX += 4;
        pY += 4;
    }

    while (pX < pEnd1) {
        *pX++ += *pY++;
    }
}

#endif