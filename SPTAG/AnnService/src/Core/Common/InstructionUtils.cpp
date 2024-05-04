#include "inc/Core/Common/InstructionUtils.h"
#include "inc/Core/Common.h"

#ifndef _MSC_VER
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86) // 03/05/24 - Sergio
void cpuid(int info[4], int InfoType) {
    __cpuid_count(InfoType, 0, info[0], info[1], info[2], info[3]);
}
#endif
#endif

namespace SPTAG {
    namespace COMMON {
        const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

        bool InstructionSet::SSE(void) { return CPU_Rep.HW_SSE; }
        bool InstructionSet::SSE2(void) { return CPU_Rep.HW_SSE2; }
        bool InstructionSet::AVX(void) { return CPU_Rep.HW_AVX; }
        bool InstructionSet::AVX2(void) { return CPU_Rep.HW_AVX2; }
        bool InstructionSet::AVX512(void) { return CPU_Rep.HW_AVX512; }
        bool InstructionSet::NEON(void) { return CPU_Rep.HW_NEON; }
        
        void InstructionSet::PrintInstructionSet(void) 
        {
            if (CPU_Rep.HW_AVX512)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX512 InstructionSet!\n");
            else if (CPU_Rep.HW_AVX2)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX2 InstructionSet!\n");
            else if (CPU_Rep.HW_AVX)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX InstructionSet!\n");
            else if (CPU_Rep.HW_SSE2)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE2 InstructionSet!\n");
            else if (CPU_Rep.HW_SSE)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE InstructionSet!\n");
            else
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using NONE InstructionSet!\n");
        }

        // from https://stackoverflow.com/a/7495023/5053214
        InstructionSet::InstructionSet_Internal::InstructionSet_Internal() :
            HW_SSE{ false },
            HW_SSE2{ false },
            HW_AVX{ false },
            HW_AVX512{ false },
            HW_AVX2{ false },
            HW_NEON{ false } // + => 03/05/24
        {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86) // + => 03/05/24
            int info[4];
            cpuid(info, 0);
            int nIds = info[0];

            //  Detect Features
            if (nIds >= 0x00000001) {
                cpuid(info, 0x00000001);
                HW_SSE = (info[3] & ((int)1 << 25)) != 0;
                HW_SSE2 = (info[3] & ((int)1 << 26)) != 0;
                HW_AVX = (info[2] & ((int)1 << 28)) != 0;
            }
            if (nIds >= 0x00000007) {
                cpuid(info, 0x00000007);
                HW_AVX2 = (info[1] & ((int)1 << 5)) != 0;
                HW_AVX512 = (info[1] & (((int)1 << 16) | ((int) 1 << 30)));

                // If we are not compiling support for AVX-512 due to old compiler version, we should not call it
                #ifdef _MSC_VER
                    #if _MSC_VER < 1920
                        HW_AVX512 = false;
                    #endif
                #endif
            }
        #elif defined(__arm__) || defined(__aarch64__)  // + => 03/05/24
            unsigned long hwcap = getauxval(AT_HWCAP);
                #ifdef __aarch64__
                    HW_NEON = (hwcap & HWCAP_ASIMD) != 0; // ASIMD is the equivalent of NEON in AArch64
                #else
                    HW_NEON = (hwcap & HWCAP_NEON) != 0;
                #endif
        #endif                                          // + => 03/05/24

            if (HW_AVX512)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX512 InstructionSet!\n");
            else if (HW_AVX2)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX2 InstructionSet!\n");
            else if (HW_AVX)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using AVX InstructionSet!\n");
            else if (HW_SSE2)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE2 InstructionSet!\n");
            else if (HW_SSE)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using SSE InstructionSet!\n");
            else if(HW_NEON)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using NEON InstructionSet!\n");
            else
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Using NONE InstructionSet!\n");
        }
    }
}
