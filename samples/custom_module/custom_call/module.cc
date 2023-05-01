// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "module.h"

#include <cstdio>
#include <functional>
#include <thread>

#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"
#include "iree/runtime/call.h"
#include "iree/vm/native_module_cc.h"

namespace {

using namespace iree;

class CustomModuleState final {
 public:
  explicit CustomModuleState(vm::ref<iree_hal_device_t> device,
                             iree_allocator_t host_allocator,
                             std::function<void(void*, void**)> fn)
      : device_(std::move(device)), host_allocator_(host_allocator), fn_(fn) {}
  ~CustomModuleState() = default;

  StatusOr<vm::ref<iree_hal_buffer_view_t>> CallAsync(
      const vm::ref<iree_hal_buffer_view_t> arg_view,
      vm::ref<iree_hal_buffer_view_t> result_view,
      const vm::ref<iree_hal_fence_t> wait_fence,
      const vm::ref<iree_hal_fence_t> signal_fence) {
    vm::ref<iree_hal_semaphore_t> semaphore;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_create(device_.get(), 0ull, &semaphore));
    Status status =
        iree_hal_fence_wait(wait_fence.get(), iree_infinite_timeout());
    if (status.ok()) {
      iree_hal_buffer_mapping_t source_mapping = {{0}};
      status = iree_hal_buffer_map_range(
          iree_hal_buffer_view_buffer(arg_view.get()),
          IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0,
          IREE_WHOLE_BUFFER, &source_mapping);
      iree_hal_buffer_mapping_t target_mapping = {{0}};
      if (status.ok()) {
        status =
            iree_hal_buffer_map_range(
                iree_hal_buffer_view_buffer(result_view.get()),
                IREE_HAL_MAPPING_MODE_SCOPED,
                IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0, IREE_WHOLE_BUFFER,
                &target_mapping);
      }

      if (status.ok()) {
        const int32_t *source_ptr =
            reinterpret_cast<const int32_t*>(source_mapping.contents.data);
        int32_t *target_ptr =
            reinterpret_cast<int32_t*>(target_mapping.contents.data);

        //  Actual computation with tensors and sizes.
        int rank = iree_hal_buffer_view_shape_rank(arg_view.get());
        const iree_hal_dim_t* dims = iree_hal_buffer_view_shape_dims(arg_view.get());
        void *in[] = {(void*)source_ptr, (void*)(&rank), (void*)dims};
        void *out[] = {(void*)target_ptr, (void*)(&rank), (void*)dims};
        fn_((void*)out, (void**)in);
        for (int i = 0; i < 10; i++) {
          printf("Stuff: %d: %d %d\n", i, source_ptr[i], target_ptr[i]);
        }
      }

      iree_status_ignore(iree_hal_buffer_unmap_range(&source_mapping));
      iree_status_ignore(iree_hal_buffer_unmap_range(&target_mapping));
    }
    if (status.ok()) {
      status = iree_hal_fence_signal(signal_fence.get());
    }
    if (!status.ok()) {
      iree_hal_fence_fail(signal_fence.get(), status.release());
    }

    return result_view;
  }

 private:
  vm::ref<iree_hal_device_t> device_;

  iree_allocator_t host_allocator_;

  std::function<void(void*, void**)> fn_;
};

static const vm::NativeFunction<CustomModuleState> kCustomModuleFunctions[] = {
    vm::MakeNativeFunction("function", &CustomModuleState::CallAsync),
};

class CustomModule final : public vm::NativeModule<CustomModuleState> {
 public:
  using vm::NativeModule<CustomModuleState>::NativeModule;
  CustomModule(const char* name, uint32_t version, iree_vm_instance_t* instance,
             iree_allocator_t allocator,
             std::function<void(void*, void**)> fn)
    : NativeModule(
          name, version, instance, allocator, 
          iree::span<const vm::NativeFunction<CustomModuleState>>(
              kCustomModuleFunctions)),
      fn_(fn) {}

  void SetDevice(vm::ref<iree_hal_device_t> device) {
    device_ = std::move(device);
  }

  StatusOr<std::unique_ptr<CustomModuleState>> CreateState(
      iree_allocator_t host_allocator) override {
    auto state = std::make_unique<CustomModuleState>(vm::retain_ref(device_),
                                                     host_allocator, fn_);
    return state;
  }

 private:
  vm::ref<iree_hal_device_t> device_;

  std::function<void(void*, void**)> fn_;
};

}  // namespace

void Internal(int s, void* out, void** in) {
  void** outVal = reinterpret_cast<void**>(out);
  int32_t *outBuffer = reinterpret_cast<int32_t*>(outVal[0]);
  int *outRank = reinterpret_cast<int*>(outVal[1]);
  int64_t *outShape = reinterpret_cast<int64_t*>(outVal[2]);
  int32_t *inBuffer = reinterpret_cast<int32_t*>(in[0]);
  int *inRank = reinterpret_cast<int*>(in[1]);
  int64_t *inShape = reinterpret_cast<int64_t*>(in[2]);
  assert(*inRank == *outRank);
  printf("Murali times: %d: inRank: %d\n", s, *inRank);
  int totalCount = 1;
  for (auto i = 0; i < *outRank; i++) {
    assert(outShape[i] == inShape[i]);
    printf("Rajani times: %d, index: %d, outShape: %ld\n", s, i, outShape[i]);
    totalCount *= outShape[i];
  }
  for (int i = 0; i < totalCount; i++) {
    printf("Mud times: %d, index: %d, inBuffer: %d, outBuffer: %d\n", s, i, inBuffer[i], outBuffer[i]);
    outBuffer[i] = inBuffer[i]*s;
    printf("Daran times: %d, index: %d, inBuffer: %d, outBuffer: %d\n", s, i, inBuffer[i], outBuffer[i]);
  }
}

void Double(void* out, void** in) {
  return Internal(2, out, in);
}

void Triple(void* out, void** in) {
  return Internal(3, out, in);
}

// Note that while we are using C++ bindings internally we still expose the
// module as a C instance. This hides the details of our implementation.
extern "C" void iree_custom_module_custom_call_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_runtime_session_t* session,
    const char* name, void (*fn)(void*, void**)) {
  iree_vm_module_t* custom_module = NULL;
  auto module = std::make_unique<CustomModule>(
      name, /*version=*/0, instance, host_allocator, fn);
  module->SetDevice(vm::retain_ref(device));
  custom_module = module.release()->interface();
  IREE_CHECK_OK(iree_runtime_session_append_module(session, custom_module));
  iree_vm_module_release(custom_module);
}
