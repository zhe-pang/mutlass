# Copyright (c) 2024 - 2024 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if (NOT CUSTOM_MUSA_PATH)
  set(CUSTOM_MUSA_PATH "/usr/local/musa")
endif()
set(MUSA_PATH "${CUSTOM_MUSA_PATH}")
set(MUSA_MCC_EXECUTABLE "${MUSA_PATH}/bin/mcc")
set(MUSA_CMAKE_PATH "${MUSA_PATH}/cmake")

if(NOT EXISTS ${MUSA_MCC_EXECUTABLE})
  message(FATAL_ERROR "${MUSA_MCC_EXECUTABLE} does not exist, please check!" )
endif()
set(CMAKE_CXX_COMPILER ${MUSA_MCC_EXECUTABLE})

if(NOT EXISTS ${MUSA_CMAKE_PATH})
  message(FATAL_ERROR "${MUSA_CMAKE_PATH} does not exist, please check!" )
endif()
set(CMAKE_MODULE_PATH ${MUSA_CMAKE_PATH})

find_package(MUSA REQUIRED)
if(MUSA_FOUND)
  message(STATUS "MUSA_FOUND")
  add_definitions(-DMUSA_FOUND)
  include_directories(${MUSA_INCLUDE_DIRS})
  set(MUSA_LINK_LIBRARIES_KEYWORD PRIVATE)
else ()
  message(FATAL_ERROR "MUSA_NOT_FOUND")
endif()

function(mutlass_add_library NAME)

  set(options SKIP_GENCODE_FLAGS)
  set(oneValueArgs EXPORT_NAME)
  set(multiValueArgs MP_ARCHS_ MCC_FLAGS_)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(ARG ${__UNPARSED_ARGUMENTS})
    if(NOT ${ARG} MATCHES ".*.mu$")
      set_source_files_properties(${ARG}
      PROPERTIES
      MUSA_SOURCE_PROPERTY_FORMAT OBJ
      )
    endif()
  endforeach()

  string(REPLACE ";" " " MCC_FLAGS_ "${__MCC_FLAGS_}")
  mutlass_apply_musa_gencode_flags(MUSA_MCC_FLAGS)
  list(APPEND MUSA_MCC_FLAGS ${MUTLASS_MUSA_MCC_FLAGS} ${MCC_FLAGS_})

  set(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})
  MUSA_ADD_LIBRARY(${NAME} ${TARGET_SOURCE_ARGS} "")

  if(__EXPORT_NAME)
    add_library(mt::mutlass::${__EXPORT_NAME} ALIAS ${NAME})
    set_target_properties(${NAME} PROPERTIES EXPORT_NAME ${__EXPORT_NAME})
  endif()

endfunction()

function(mutlass_add_executable NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs MP_ARCHS_ MCC_FLAGS_)

  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(ARG ${__UNPARSED_ARGUMENTS})
    if(NOT ${ARG} MATCHES ".*.mu$")
      set_source_files_properties(${ARG}
      PROPERTIES
      MUSA_SOURCE_PROPERTY_FORMAT OBJ
      )
    endif()
  endforeach()
  string(REPLACE ";" " " MCC_FLAGS_ "${__MCC_FLAGS_}")
  mutlass_apply_musa_gencode_flags(MUSA_MCC_FLAGS)
  list(APPEND MUSA_MCC_FLAGS ${MUTLASS_MUSA_MCC_FLAGS} ${MCC_FLAGS_})

  set(TARGET_SOURCE_ARGS ${__UNPARSED_ARGUMENTS})

  set(MUSA_LINK_LIBRARIES_KEYWORD PRIVATE)
  MUSA_ADD_EXECUTABLE(${NAME} ${TARGET_SOURCE_ARGS})
  target_compile_features(${NAME} PRIVATE cxx_std_17)

endfunction()

function(mutlass_target_sources NAME)

  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(ARG ${__UNPARSED_ARGUMENTS})
      set_source_files_properties(${ARG}
      PROPERTIES
      MUSA_SOURCE_PROPERTY_FORMAT OBJ
      COMPILE_FLAGS "-x musa -fPIC"
      )
    set_source_files_properties(${ARG} PROPERTIES LANGUAGE CXX)
  endforeach()

  target_sources(${NAME} ${__UNPARSED_ARGUMENTS})
endfunction()

