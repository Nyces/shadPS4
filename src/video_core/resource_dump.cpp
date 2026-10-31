// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/resource_dump.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "shader_recompiler/info.h"

namespace VideoCore {

namespace {

/// Only small blocks are printed in full: material constants and uniform
/// buffers fit, while vertex, index and bone-matrix buffers only get their
/// descriptor logged.
constexpr u64 MAX_CONTENT_BYTES = 2_KB;

/// Ceiling for one frame, so a runaway dump cannot eat the log file budget.
constexpr u64 MAX_FRAME_CONTENT = 4_MB;

/// Dwords per printed line, so no single log line turns into a wall of hex.
constexpr u64 DWORDS_PER_LINE = 16;

struct FrameState {
    u32 frame = ~0u;
    u32 entries = 0;
    u64 content = 0;
    std::unordered_set<u64> seen;
};

FrameState& State() {
    static FrameState state;
    return state;
}

const char* StageName(Shader::SwStage stage) {
    switch (stage) {
    case Shader::SwStage::Fragment:
        return "ps";
    case Shader::SwStage::TessellationControl:
        return "hs";
    case Shader::SwStage::TessellationEval:
        return "ls";
    case Shader::SwStage::Vertex:
        return "vs";
    case Shader::SwStage::Geometry:
        return "gs";
    case Shader::SwStage::Compute:
        return "cs";
    default:
        return "??";
    }
}

std::string UserDataString(const Shader::Info& stage) {
    std::string text;
    for (const u32 value : stage.flattened_ud_buf) {
        if (!text.empty()) {
            text += ',';
        }
        text += fmt::format("{:08x}", value);
    }
    return text;
}

/// One key per distinct (stage, shader, buffer), so the same material block
/// bound by a hundred draws is printed once per frame instead of a hundred
/// times.
u64 BufferKey(Shader::SwStage stage, u64 hash, u64 address, u64 size) {
    u64 key = static_cast<u64>(stage) + 1;
    key = key * 0x100000001b3ULL ^ hash;
    key = key * 0x100000001b3ULL ^ address;
    key = key * 0x100000001b3ULL ^ size;
    return key;
}

void LogContent(u32 frame, const char* stage, u32 binding, u64 hash, const u8* data, u64 size) {
    const u64 dwords = size / sizeof(u32);
    for (u64 first = 0; first < dwords; first += DWORDS_PER_LINE) {
        std::string text;
        const u64 last = std::min(dwords, first + DWORDS_PER_LINE);
        for (u64 index = first; index < last; ++index) {
            if (index != first) {
                text += ',';
            }
            u32 value;
            std::memcpy(&value, data + index * sizeof(u32), sizeof(u32));
            text += fmt::format("{:08x}", value);
        }
        LOG_INFO(Render_Vulkan, "RESDUMP|hex|f{}|{}|b{}|hash={:#x}|c{}|{}", frame, stage, binding,
                 hash, first, text);
    }
}

} // namespace

void DumpDrawResources(const Shader::Info& stage, Core::MemoryManager* memory) {
    auto& state = State();
    const u32 frame = DebugState.GetFrameNum();
    if (state.frame != frame) {
        state.frame = frame;
        state.entries = 0;
        state.content = 0;
        state.seen.clear();
    }
    const u32 entry = ++state.entries;

    const char* name = StageName(stage.sw_stage);
    LOG_INFO(Render_Vulkan, "RESDUMP|meta|f{}|e{}|{}|hash={:#x}|ud={}", frame, entry, name,
             stage.pgm_hash, UserDataString(stage));

    for (size_t index = 0; index < stage.buffers.size(); ++index) {
        const auto& desc = stage.buffers[index];
        if (desc.IsSpecial()) {
            continue;
        }
        const auto vsharp = desc.GetSharp(stage);
        if (!vsharp || vsharp.base_address == 0) {
            continue;
        }

        const u32 binding = static_cast<u32>(index);
        const u64 address = vsharp.base_address;
        LOG_INFO(Render_Vulkan,
                 "RESDUMP|buf|f{}|e{}|{}|b{}|hash={:#x}|addr={:#x}|size={}|stride={}|rec={}|w={}",
                 frame, entry, name, binding, stage.pgm_hash, address, vsharp.GetSize(),
                 vsharp.GetStride(), vsharp.num_records, desc.is_written);

        const u64 size = std::min<u64>(vsharp.GetSize(), MAX_CONTENT_BYTES);
        if (size < sizeof(u32) || size % sizeof(u32) != 0) {
            continue;
        }
        const u64 key = BufferKey(stage.sw_stage, stage.pgm_hash, address, size);
        if (!state.seen.insert(key).second) {
            continue;
        }
        if (state.content + size > MAX_FRAME_CONTENT) {
            continue;
        }

        std::vector<u8> bytes(size);
        memory->CopySparseMemory(address, bytes.data(), size);
        state.content += size;
        LogContent(frame, name, binding, stage.pgm_hash, bytes.data(), size);
    }
}

} // namespace VideoCore
