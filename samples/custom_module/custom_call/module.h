// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_CUSTOM_MODULE_TENSOR_ASYNC_MODULE_H_
#define IREE_SAMPLES_CUSTOM_MODULE_TENSOR_ASYNC_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/call.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void Double(void* out, void** in);
void Triple(void* out, void** in);
void iree_custom_module_custom_call_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_runtime_session_t* session,
    const char* name, void (*fn)(void*, void**));

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_SAMPLES_CUSTOM_MODULE_TENSOR_ASYNC_MODULE_H_
