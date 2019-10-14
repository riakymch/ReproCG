/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file ExSUM.FPE.hpp
 *  \brief A set of routines concerning floating-point expansions
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#ifndef FPEXPANSIONVECT_HPP_INCLUDED
#define FPEXPANSIONVECT_HPP_INCLUDED
#include "accumulate.h"

namespace exblas
{
namespace cpu
{

/**
 * \struct FPExpansionTraits
 * \ingroup lowlevel
 * \brief This struct is meant to specify optimization or other technique used
 */
template<bool EX=false, bool FLUSHHI=false, bool H2SUM=false, bool CRF=false, bool CSWAP=false, bool B2SUM=true, bool SORT=false, bool VICT=false>
struct FPExpansionTraits
{
    static bool constexpr EarlyExit = EX;
    static bool constexpr FlushHi = FLUSHHI;
    static bool constexpr Horz2Sum = H2SUM;
    static bool constexpr CheckRangeFirst = CRF;
    static bool constexpr ConditionalSwap = CSWAP;
    static bool constexpr Biased2Sum = B2SUM;
    static bool constexpr Sort = SORT;
    static bool constexpr Victimcache = VICT;
};

/**
 * \struct FPExpansionVect
 * \ingroup lowlevel
 * \brief This struct is meant to introduce functionality for working with
 *  floating-point expansions in conjuction with superaccumulators
 */
template<typename T, int N, typename TRAITS=FPExpansionTraits<false,false> >
struct FPExpansionVect
{
    /**
     * Constructor
     * \param fpe
     */
    FPExpansionVect(T* fpeinit);
    
    /**
     * This function accumulates value x to the floating-point expansion
     * \param x input value
     */
    void Accumulate(T x);

private:
    static T twosum(T a, T b, T & s);

    T* fpe;
};

template<typename T, int N, typename TRAITS>
FPExpansionVect<T,N,TRAITS>::FPExpansionVect(T* fpeinit) :
    fpe(fpeinit)
{}

// Knuth 2Sum.
template<typename T>
inline static T KnuthTwoSum(T a, T b, T & s)
{
    T r = a + b;
    T z = r - a;
    s = (a - (r - z)) + (b - z);
    return r;
}
template<typename T>
inline static T TwoProductFMA(T a, T b, T &d) {
    T p = a * b;
#ifdef _WITHOUT_VCL
    d = a*b-p;
#else
    d = vcl::mul_sub_x(a, b, p); //extra precision even if FMA is not available
#endif//_WITHOUT_VCL
    return p;
}

// Knuth 2Sum with FMAs
template<typename T>
inline static T FMA2Sum(T a, T b, T & s)
{
#ifndef _WITHOUT_VCL
    T r = a + b;
    T z = vcl::mul_sub(1., r, a);
    s = vcl::mul_add(1., a - vcl::mul_sub(1., r, z), b - z);
    return r;
#else
    T r = a + b;
    T z = r - a;
    s = (a - (r - z)) + (b - z);
    return r;
#endif//_WITHOUT_VCL
}

template<typename T, int N, typename TRAITS> UNROLL_ATTRIBUTE
void FPExpansionVect<T,N,TRAITS>::Accumulate(T x)
{
    // Experimental
    if(TRAITS::CheckRangeFirst && horizontal_or(abs(x) < abs(fpe[N-1]))) {
        return;
    }

    T s;
    for(unsigned int i = 0; i != N; ++i) {
        fpe[i] = twosum(fpe[i], x, s);
        x = s;
        if(TRAITS::EarlyExit && !horizontal_or(x))
	    return;
    }

    if (horizontal_or(x)) {
        fprintf(stderr, "WARN: with the FPE-based implementation we cannot keep every bit of information for this problem due to either high condition number (ill-cond), too broad dynamic range, or both. Thus, we cannot ensure correct-rounding and bitwise reproducibility. Hence, we advise to switch to the ExBLAS-based implementation for this particular (rather rare) case.\n");
    }
}

template<typename T, int N, typename TRAITS>
T FPExpansionVect<T,N,TRAITS>::twosum(T a, T b, T & s)
{
//#if INSTRSET > 7                       // AVX2 and later
	// Assume Haswell-style architecture with parallel Add and FMA pipelines
	return FMA2Sum(a, b, s);
//#else
    //if(TRAITS::Biased2Sum) {
    //    return BiasedSIMD2Sum(a, b, s);
    //}
    //else {
    //    return KnuthTwoSum(a, b, s);
    //}
//#endif
}

/**
* @brief Convert a fpe to the nearest double precision number (CPU version)
*
* @ingroup highlevel
* @param fpe a pointer to N doubles on the CPU (representing the fpe)
* @return the double precision number nearest to the fpe
*/
template<typename T>
inline static T Round( const T *fpe ) {

    // Now add3(hi, mid, lo)
    // Adapted from:
    // Sylvie Boldo, and Guillaume Melquiond. "Emulation of a FMA and correctly rounded sums: proved algorithms using rounding to odd." IEEE Transactions on Computers, 57, no. 4 (2008): 462-471.
    union {
        T d;
        int64_t l;
    } thdb;

    T tl;
    T th = FMA2Sum(fpe[1], fpe[2], tl);
   
    if (tl != 0.0) {
        thdb.d = th;
        // if the mantissa of th is odd, there is nothing to do
        if (!(thdb.l & 1)) {
            // choose the rounding direction
            // depending of the signs of th and tl
            if ((tl > 0.0) ^ (th < 0.0))
                thdb.l++;
            else
                thdb.l--;
            th = thdb.d;
        }
        
    } 

    // final addition rounded to nearest
    return fpe[0] + th;
}

#undef IACA
#undef IACA_START
#undef IACA_END

}//namespace cpu
}//namespace exblas
#endif
