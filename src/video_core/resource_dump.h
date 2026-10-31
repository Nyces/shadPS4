// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/types.h"

namespace Core {
class MemoryManager;
}

namespace Shader {
struct Info;
}

namespace VideoCore {

/**
 * True when the current frame should have its shader resources dumped.
 *
 * The automatic capture window fires for a couple of frames every
 * ``AUTO_PERIOD`` frames once the run has warmed up, so a single unassisted
 * playthrough is enough; there is no hotkey to press.
 */
bool ShouldDumpDrawResources();

/**
 * Writes one draw call's shader resources into the log
 * (``<user>/log/shad_log.txt``) with a ``RESDUMP|`` prefix.
 *
 * A RenderDoc capture records what the GPU was told to do, but not which guest
 * address a descriptor resolves to, so the constants the character shaders read
 * out of storage buffers stay out of reach of a capture. Inside the emulator
 * both halves are available: the recompiled stage carries the V#/T# descriptors
 * and the guest memory behind them can be read directly.
 *
 * Lines emitted per draw:
 *  * ``RESDUMP|draw|f<frame>|d<draw>|ps=..;vs=..`` one line tying the stages of
 *    this draw together, which a per-stage log cannot express;
 *  * ``RESDUMP|ud|f<frame>|d<draw>|<stage>|hash=..|<16 dwords>`` the user-data
 *    SGPRs, logged per draw because they change with the draw;
 *  * ``RESDUMP|buf|..|b<binding>|addr=..|size=..|stride=..|rec=..|w=..`` and
 *    ``RESDUMP|img|..|i<binding>|addr=..|w=..|h=..|fmt=..|nf=..|pitch=..|type=..``
 *    for every bound guest buffer and texture;
 *  * ``RESDUMP|hex|..|c<first dword>|<dwords>`` for buffers small enough to
 *    hold constants (see MAX_CONTENT_BYTES), split over several lines.
 *
 * Image dimensions are the raw T# fields, i.e. one less than the real size.
 * Buffer contents are written once per distinct block, so a material shared by
 * a hundred draws and by every captured frame costs one copy, not a hundred.
 */
void DumpDrawResources(std::span<const Shader::Info* const> stages, Core::MemoryManager* memory);

} // namespace VideoCore
