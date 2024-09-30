/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <mute/config.hpp>

#include <mute/util/type_traits.hpp>  // iterator_traits
#include <mute/container/array_subbyte.hpp>

#include <mute/pointer_base.hpp>
#include <mute/swizzle.hpp>

/* This implements a swizzle pointer of the form
 *   InvolutionFn o PtrAdd
 * where the InvolutionFn need not be linear.
 *
 * This differs subtly from swizzle_layout because the smem pointer is used
 * as the offset. That means that swizzle_layout will implement position-independent
 * swizzle layouts, while swizzle_ptr implements position-dependent swizzle tensors.
 * Arch chose to design hardware with position-dependent swizzles.
 *
 * For clarity:
 *   NormalLayout  : DeRef <- PtrAdd <- [Layout]
 *   ComposedLayout: DeRef <- PtrAdd <- [Swizzle <- OffsetAdd <- Layout]
 *   SwizzlePtr    : [DeRef <- Swizzle <- PtrAdd] <- Layout
 *
 * Furthermore, for known swizzles, this pointer attempts to decay itself
 *    to a normal-pointer with a new layout containing dynamic or static strides.
 * This is possible by determining the subdomain of the InvolutionFn
 *    that is identity and testing if the Layout's codomain is contained
 *    within it.
 */

namespace mute
{

// concept SwizzleFn {
//   MUTE_HOST_DEVICE constexpr static uint apply(uint);
// }
// See Swizzle<B,M,S> in swizzle.hpp for common swizzle-functions.

template <class SwizzleFn, class Iterator>
struct swizzle_ptr : iter_adaptor<Iterator,swizzle_ptr<SwizzleFn,Iterator>>
{
  using iterator     = Iterator;
  using reference    = typename iterator_traits<iterator>::reference;
  using element_type = typename iterator_traits<iterator>::element_type;
  using value_type   = typename iterator_traits<iterator>::value_type;

  using iter_adaptor<Iterator,swizzle_ptr<SwizzleFn,Iterator>>::iter_adaptor;

  template <class Iter>
  MUTE_HOST_DEVICE constexpr static
  Iter apply_swizzle(Iter ptr) {
    if constexpr (mute::is_gmem_v<Iter>) {
      return {apply_swizzle<typename iterator_traits<decltype(ptr.get())>::value_type, 1>(ptr.get())};
    } else if constexpr (mute::is_smem_v<Iter>) {
      return {apply_swizzle<typename iterator_traits<decltype(ptr.get())>::value_type, 3>(ptr.get())};
    } else {
      return {apply_swizzle<typename iterator_traits<decltype(ptr.get())>::value_type, 0>(ptr.get())};
    }
  }

  template <class T, int I = 0>
  MUTE_HOST_DEVICE constexpr static
  T* apply_swizzle(T* ptr) {
    // reinterpret_cast can't be used to cast to a different address space
    // compiler only allows c-style pointer cast
    return (T*)(reinterpret_cast<T __attribute__((address_space(I)))*>(
                  SwizzleFn::apply(reinterpret_cast<uintptr_t>(ptr))));
  }

  template <class T, int I = 0>
  MUTE_HOST_DEVICE constexpr static
  subbyte_iterator<T> apply_swizzle(subbyte_iterator<T> ptr) {
    return {apply_swizzle<T, I>(ptr.ptr_), ptr.idx_};
  }

  MUTE_HOST_DEVICE constexpr
  reference operator*() const {
    return *apply_swizzle(this->get());
  }

  template <class Int>
  MUTE_HOST_DEVICE constexpr
  reference operator[](Int const& i) const {
    return *apply_swizzle(this->get() + i);
  }
};

template <class T, class = void>                      // Default No-Swizzle
struct get_swizzle { using type = Swizzle<0,4,3>; };
template <class SwizzleFn, class P>                   // Found the SwizzleFn
struct get_swizzle<swizzle_ptr<SwizzleFn,P>> { using type = SwizzleFn; };
template <class T>                                    // Recurse into anything with a ::iterator
struct get_swizzle<T, void_t<typename T::iterator>> : get_swizzle<typename T::iterator> {};

template <class Iter>
using get_swizzle_t = typename get_swizzle<Iter>::type;

template <class Iterator, class SwizzleFn>
MUTE_HOST_DEVICE constexpr
swizzle_ptr<SwizzleFn,Iterator>
make_swizzle_ptr(Iterator ptr, SwizzleFn) {
  return {ptr};
}

// Swizzle-0 specialization for immediate decay
template <class Iterator, int M, int S>
MUTE_HOST_DEVICE constexpr
Iterator
make_swizzle_ptr(Iterator ptr, Swizzle<0,M,S>) {
  return ptr;
}

//
// Recast
//

template <class SwizzleFn, class P>
MUTE_HOST_DEVICE constexpr
auto
raw_pointer_cast(swizzle_ptr<SwizzleFn,P> const& ptr) {
  return raw_pointer_cast(ptr.get());
}

// SwizzleFn operates on the pointer address, so it doesn't care about the type
template <class NewT, class SwizzleFn, class P>
MUTE_HOST_DEVICE constexpr
auto
recast_ptr(swizzle_ptr<SwizzleFn,P> const& ptr) {
  return make_swizzle_ptr(recast_ptr<NewT>(ptr.get()), SwizzleFn{});
}

//
// Display utilities
//

template <class SwizzleFn, class P>
MUTE_HOST_DEVICE void print(swizzle_ptr<SwizzleFn,P> ptr)
{
  print(SwizzleFn{}); printf("_"); print(ptr.get());
}

#if !defined(__MUSACC_RTC__)
template <class SwizzleFn, class P>
MUTE_HOST std::ostream& operator<<(std::ostream& os, swizzle_ptr<SwizzleFn,P> ptr)
{
  return os << SwizzleFn{} << "_" << ptr.get();
}
#endif

} // end namespace mute
