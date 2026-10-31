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
 * Writes the resource descriptors of one shader stage of one draw into the log
 * (``<user>/log/shad_log.txt``) with a ``RESDUMP|`` prefix.
 *
 * A RenderDoc capture records what the GPU was told to do, but not which guest
 * address a descriptor resolves to, so the constants the character shaders read
 * out of storage buffers stay out of reach of a capture. Inside the emulator
 * both halves are available: the recompiled stage carries the V# descriptors
 * and the guest memory behind them can be read directly.
 *
 * Three line kinds are emitted:
 *  * ``RESDUMP|meta|f<frame>|e<entry>|<stage>|hash=..|ud=..`` once per stage,
 *    with the 16 user-data SGPRs the shader received;
 *  * ``RESDUMP|buf|..|b<binding>|addr=..|size=..|stride=..|rec=..|w=..`` per
 *    bound guest buffer;
 *  * ``RESDUMP|hex|..|c<first dword>|<dwords>`` for buffers small enough to
 *    hold constants (see MAX_CONTENT_BYTES), split over several lines.
 *
 * The dump runs for the frames requested by
 * ``DebugState.RequestResourceDump()``.
 */
void DumpDrawResources(const Shader::Info& stage, Core::MemoryManager* memory);

} // namespace VideoCore
