/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */

/**
 *  @file exdot_serial.h
 *  @brief Serial version of exdot
 *
 *  @authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>

#include "fpexpansionvect.hpp"

namespace exblas{
///@cond
namespace cpu{

template<typename CACHE, typename PointerOrValue1, typename PointerOrValue2>
void ExDOTFPE_cpu(int N, PointerOrValue1 a, PointerOrValue2 b, double* fpe) {
    CACHE cache(fpe);
#ifndef _WITHOUT_VCL
    int r = (( int64_t(N) ) & ~7ul);
    for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
        asm ("# myloop");
#endif
        vcl::Vec8d r1 ;
        vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,i), make_vcl_vec8d(b,i), r1);
        cache.Accumulate(x);
        cache.Accumulate(r1);
    }
    if( r != N) {
        //accumulate remainder
        vcl::Vec8d r1;
        vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,r,N-r), make_vcl_vec8d(b,r,N-r), r1);
        cache.Accumulate(x);
        cache.Accumulate(r1);
    }
#else// _WITHOUT_VCL
    for(int i = 0; i < N; i++) {
        double r1;
        double x = TwoProductFMA(a[i],b[i],r1);
        cache.Accumulate(x);
        cache.Accumulate(r1);
    }
#endif// _WITHOUT_VCL
}
}//namespace cpu
///@endcond

/*!@brief serial version of exact dot product
 *
 * Computes the exact dot \f[ \sum_{i=0}^{N-1} x_i y_i \f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @sa \c exblas::cpu::Round  to convert the superaccumulator into a double precision number
*/
template<class PointerOrValue1, class PointerOrValue2, size_t NBFPE=3>
void exdot_cpu(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, double* fpe){
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
    for( int i=0; i<NBFPE; i++)
        fpe[i] = 0;
#ifndef _WITHOUT_VCL
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, fpe);
#else
    cpu::ExDOTFPE_cpu<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, fpe);
#endif//_WITHOUT_VCL
}


}//namespace exblas
