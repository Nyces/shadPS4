// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core {
class MemoryManager;
}

namespace Shader {
struct Info;
}

namespace VideoCore {

/**
 * Writes the resource descriptors of one shader stage of one draw to
 * ``<user>/resource_dump``.
 *
 * A RenderDoc capture records what the GPU was told to do, but not which guest
 * address a descriptor resolves to, so the constants the character shaders read
 * out of storage buffers stay out of reach of a capture. Inside the emulator
 * both halves are available: the recompiled stage carries the V# descriptors
 * and the guest memory behind them can be read directly. Each dump appends one
 * JSONL line (stage, shader hash, user data, every bound buffer with its
 * address, size and stride) plus one raw ``.bin`` per buffer.
 *
 * The dump runs for the frames requested by
 * ``DebugState.RequestResourceDump()``.
 */
void DumpDrawResources(const Shader::Info& stage, Core::MemoryManager* memory);

} // namespace VideoCore
