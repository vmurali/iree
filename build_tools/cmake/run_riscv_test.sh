#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper script to run the artifact on RISC-V 32/64-bit Linux device.
# This script checks if QEMU emulator is set, and use either the emulator or
# the actual device to run the cross-compiled RISC-V linux artifacts.

set -x
set -e

RISCV_ARCH="${RISCV_ARCH:-rv64}"

# A QEMU Linux emulator must be available Within the system that matches the
# processor architecturue. The emulators are at the path specified by the
# `QEMU_RV64_BIN` or `QEMU_RV32_BIN` environment variable to run the artifacts
# under the emulator.
if [[ "${RISCV_ARCH}" == "rv64" ]] && [[ ! -z "${QEMU_RV64_BIN}" ]]; then
  "${QEMU_RV64_BIN}" "-cpu" "rv64,x-v=true,x-k=true,vlen=512,elen=64,vext_spec=v1.0" \
  "-L" "${RISCV_RV64_LINUX_TOOLCHAIN_ROOT}/sysroot" "$@"
elif [[ "${RISCV_ARCH}" == "rv32-linux" ]] && [[ ! -z "${QEMU_RV32_BIN}" ]]; then
  "${QEMU_RV32_BIN}" "-cpu" "rv32,x-v=true,x-k=true,vlen=512,elen=32,vext_spec=v1.0" \
  "-L" "${RISCV_RV64_LINUX_TOOLCHAIN_ROOT}/sysroot" "$@"
else
# TODO(dcaballe): Add on-device run commands.
  "$@"
fi
