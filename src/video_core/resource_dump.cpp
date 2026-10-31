// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/resource_dump.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/info.h"

namespace VideoCore {

namespace {

/// Frames to wait before the automatic capture starts, so the splash screens
/// and the menus do not fill the log.
constexpr u32 AUTO_DELAY = 600;

/// One capture window every this many frames, and how many frames it lasts.
constexpr u32 AUTO_PERIOD = 900;
constexpr u32 AUTO_WINDOW = 2;

/// Only small blocks are printed in full: material constants and uniform
/// buffers fit, while vertex, index and bone-matrix buffers only get their
/// descriptor logged.
constexpr u64 MAX_CONTENT_BYTES = 2_KB;

/// Total amount of buffer bytes printed over the whole run.
constexpr u64 MAX_TOTAL_CONTENT = 24_MB;

/// Dwords per printed line, so no single log line turns into a wall of hex.
constexpr u64 DWORDS_PER_LINE = 16;

struct CaptureState {
    std::mutex mutex;
    /// Distinct buffer blocks already printed, keyed by their content hash.
    std::unordered_set<u64> printed;
    /// Image descriptors already printed for the current frame.
    std::unordered_set<u64> images;
    /// Vertex attribute bindings already printed for the current frame.
    std::unordered_set<u64> vertices;
    u32 frame = ~0u;
    u32 draws = 0;
    std::atomic<u64> content{0};
    bool auto_started = false;
};

CaptureState& State() {
    static CaptureState state;
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

u64 Mix(u64 value, u64 add) {
    return (value ^ add) * 0x100000001b3ULL;
}

u64 ContentHash(const u8* data, u64 size) {
    u64 hash = 0xcbf29ce484222325ULL;
    for (u64 index = 0; index < size; ++index) {
        hash = (hash ^ data[index]) * 0x100000001b3ULL;
    }
    return hash;
}

void LogContent(u32 frame, u32 draw, const char* stage, u32 binding, u64 hash, const u8* data,
                u64 size) {
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
        LOG_INFO(Render_Vulkan, "RESDUMP|hex|f{}|d{}|{}|b{}|hash={:#x}|c{}|{}", frame, draw, stage,
                 binding, hash, first, text);
    }
}

bool IsAutoCaptureFrame() {
    if (!EmulatorSettings.IsResourceDumpAuto()) {
        return false;
    }
    const u32 frame = DebugState.GetFrameNum();
    if (frame < AUTO_DELAY) {
        return false;
    }
    return (frame - AUTO_DELAY) % AUTO_PERIOD < AUTO_WINDOW;
}

} // namespace

bool ShouldDumpDrawResources() {
    return IsAutoCaptureFrame();
}

void DumpVertexInput(u32 location, u64 address, u64 stride, u64 records, u32 data_format,
                     u32 num_format, u32 vk_format) {
    if (!ShouldDumpDrawResources()) {
        return;
    }

    auto& state = State();
    const u32 frame = DebugState.GetFrameNum();

    std::lock_guard lock{state.mutex};
    if (state.frame != frame) {
        state.frame = frame;
        state.draws = 0;
        state.images.clear();
        state.vertices.clear();
    }
    const u64 key = Mix(Mix(Mix(Mix(3, location), address), stride), records);
    if (!state.vertices.insert(key).second) {
        return;
    }
    LOG_INFO(Render_Vulkan, "RESDUMP|vtx|f{}|a{}|addr={:#x}|stride={}|rec={}|fmt={}|nf={}|vk={}",
             frame, location, address, stride, records, data_format, num_format, vk_format);
}

void LogVaryingDump(u32 frame, u64 vs_hash, const u8* data) {
    constexpr u32 slots = Shader::VARYING_DUMP_SLOTS;
    constexpr u32 components = slots * 4;
    for (u32 vertex = 0; vertex < Shader::VARYING_DUMP_VERTICES; ++vertex) {
        const u8* record = data + vertex * components * sizeof(float);
        std::string text;
        bool empty = true;
        for (u32 index = 0; index < components; ++index) {
            float value;
            std::memcpy(&value, record + index * sizeof(float), sizeof(float));
            if (value != 0.0f) {
                empty = false;
            }
            if (!text.empty()) {
                text += ',';
            }
            text += fmt::format("{:.5g}", value);
        }
        if (empty) {
            // Nothing was exported for this vertex index (the draw is shorter
            // than the window), which is different from exporting zeros.
            continue;
        }
        LOG_INFO(Render_Vulkan, "RESDUMP|vary|f{}|vs={:#x}|v{}|{}", frame, vs_hash, vertex, text);
    }
}

void DumpDrawResources(std::span<const Shader::Info* const> stages, Core::MemoryManager* memory) {
    if (!ShouldDumpDrawResources()) {
        return;
    }

    auto& state = State();
    const u32 frame = DebugState.GetFrameNum();
    const bool auto_capture = IsAutoCaptureFrame();

    std::lock_guard lock{state.mutex};
    if (state.frame != frame) {
        state.frame = frame;
        state.draws = 0;
        state.images.clear();
        state.vertices.clear();
    }
    if (auto_capture && !state.auto_started) {
        state.auto_started = true;
        LOG_INFO(Render_Vulkan, "RESDUMP|auto|automatic capture starts at frame {}", frame);
    }
    const u32 draw = ++state.draws;

    // One line per draw: the per-stage lines below cannot say which pixel and
    // vertex shader ran together, and the pairing is what identifies a shader.
    std::string summary;
    for (const auto* stage : stages) {
        if (stage == nullptr) {
            continue;
        }
        if (!summary.empty()) {
            summary += ';';
        }
        summary += fmt::format("{}={:#x}", StageName(stage->sw_stage), stage->pgm_hash);
    }
    LOG_INFO(Render_Vulkan, "RESDUMP|draw|f{}|d{}|{}", frame, draw, summary);

    for (const auto* stage : stages) {
        if (stage == nullptr) {
            continue;
        }
        const char* name = StageName(stage->sw_stage);
        LOG_INFO(Render_Vulkan, "RESDUMP|ud|f{}|d{}|{}|hash={:#x}|{}", frame, draw, name,
                 stage->pgm_hash, UserDataString(*stage));

        for (size_t index = 0; index < stage->buffers.size(); ++index) {
            const auto& desc = stage->buffers[index];
            if (desc.IsSpecial()) {
                continue;
            }
            const auto vsharp = desc.GetSharp(*stage);
            if (!vsharp || vsharp.base_address == 0) {
                continue;
            }

            const u32 binding = static_cast<u32>(index);
            const u64 address = vsharp.base_address;
            LOG_INFO(Render_Vulkan,
                     "RESDUMP|buf|f{}|d{}|{}|b{}|hash={:#x}|addr={:#x}|size={}|stride={}|rec={}"
                     "|w={}",
                     frame, draw, name, binding, stage->pgm_hash, address, vsharp.GetSize(),
                     vsharp.GetStride(), vsharp.num_records, desc.is_written);

            const u64 size = std::min<u64>(vsharp.GetSize(), MAX_CONTENT_BYTES);
            if (size < sizeof(u32) || size % sizeof(u32) != 0) {
                continue;
            }
            if (state.content.load(std::memory_order_relaxed) + size > MAX_TOTAL_CONTENT) {
                continue;
            }
            std::vector<u8> bytes(size);
            memory->CopySparseMemory(address, bytes.data(), size);

            // One copy per distinct block: materials are bound by many draws and
            // by every captured frame, but the bytes only need to be in the log
            // once.
            if (!state.printed.insert(ContentHash(bytes.data(), size)).second) {
                continue;
            }
            state.content.fetch_add(size, std::memory_order_relaxed);
            LogContent(frame, draw, name, binding, stage->pgm_hash, bytes.data(), size);
        }

        for (size_t index = 0; index < stage->images.size(); ++index) {
            const auto& desc = stage->images[index];
            const auto image = desc.GetSharp(*stage);
            if (!image || image.base_address == 0) {
                continue;
            }
            const u64 image_address = image.Address();
            const u64 key =
                Mix(Mix(Mix(2, static_cast<u64>(stage->sw_stage)), stage->pgm_hash), image_address);
            if (!state.images.insert(Mix(key, index)).second) {
                continue;
            }
            LOG_INFO(Render_Vulkan,
                     "RESDUMP|img|f{}|d{}|{}|i{}|hash={:#x}|addr={:#x}|w={}|h={}|fmt={}|nf={}"
                     "|pitch={}|type={}",
                     frame, draw, name, static_cast<u32>(index), stage->pgm_hash, image_address,
                     static_cast<u32>(image.width), static_cast<u32>(image.height),
                     static_cast<u32>(image.data_format), static_cast<u32>(image.num_format),
                     static_cast<u32>(image.pitch), static_cast<u32>(image.type));
        }
    }
}

} // namespace VideoCore
