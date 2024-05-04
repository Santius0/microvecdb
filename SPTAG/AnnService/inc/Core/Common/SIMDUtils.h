// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_SIMDUTILS_H_
#define _SPTAG_COMMON_SIMDUTILS_H_

#include <functional>
#include <iostream>

#include "CommonUtils.h"
#include "InstructionUtils.h"

namespace SPTAG
{
    namespace COMMON
    {
        template <typename T>
        using SumCalcReturn = void(*)(T*, const T*, DimensionType);
        template<typename T>
        inline SumCalcReturn<T> SumCalcSelector();

        class SIMDUtils
        {
        public:
            template <typename T>
            static void ComputeSum_Naive(T* pX, const T* pY, DimensionType length)
            {
                const T* pEnd1 = pX + length;
                while (pX < pEnd1) {
                    *pX++ += *pY++;
                }
            }

            #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86) // + => 04/05/2024

            static void ComputeSum_SSE(std::int8_t* pX, const std::int8_t* pY, DimensionType length);
            static void ComputeSum_AVX(std::int8_t* pX, const std::int8_t* pY, DimensionType length);
            static void ComputeSum_AVX512(std::int8_t* pX, const std::int8_t* pY, DimensionType length);

            static void ComputeSum_SSE(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static void ComputeSum_AVX(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static void ComputeSum_AVX512(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);

            static void ComputeSum_SSE(std::int16_t* pX, const std::int16_t* pY, DimensionType length);
            static void ComputeSum_AVX(std::int16_t* pX, const std::int16_t* pY, DimensionType length);
            static void ComputeSum_AVX512(std::int16_t* pX, const std::int16_t* pY, DimensionType length);

            static void ComputeSum_SSE(float* pX, const float* pY, DimensionType length);
            static void ComputeSum_AVX(float* pX, const float* pY, DimensionType length);
            static void ComputeSum_AVX512(float* pX, const float* pY, DimensionType length);

            #elif defined(__arm__) || defined(__aarch64__) // + => 04/05/24

            static void ComputeSum_NEON(std::int8_t* pX, const std::int8_t* pY, DimensionType length);
            static void ComputeSum_NEON(std::uint8_t* pX, const std::uint8_t* pY, DimensionType length);
            static void ComputeSum_NEON(std::int16_t* pX, const std::int16_t* pY, DimensionType length);
            static void ComputeSum_NEON(float* pX, const float* pY, DimensionType length);

            #endif // + => 04/05/24

            template<typename T>
            static inline void ComputeSum(T* p1, const T* p2, DimensionType length)
            {
                auto func = SumCalcSelector<T>();
                return func(p1, p2, length);
            }
        };

        template<typename T>
        inline SumCalcReturn<T> SumCalcSelector()
        {
            #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86) // + => 04/05/2024
            if (InstructionSet::AVX512())
            {
                return &(SIMDUtils::ComputeSum_AVX512);
            }
            bool isSize4 = (sizeof(T) == 4);
            if (InstructionSet::AVX2() || (isSize4 && InstructionSet::AVX()))
            {
                return &(SIMDUtils::ComputeSum_AVX);
            }
            if (InstructionSet::SSE2() || (isSize4 && InstructionSet::SSE()))
            {
                return &(SIMDUtils::ComputeSum_SSE);
            }
            #elif defined(__arm__) || defined(__aarch64__) // + => 04/05/24
            if(InstructionSet::NEON())
            {
                return &(SIMDUtils::ComputeSum_NEON);
            }
            #endif
            return &(SIMDUtils::ComputeSum_Naive);
        }
    }
}

#endif // _SPTAG_COMMON_SIMDUTILS_H_
