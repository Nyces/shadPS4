// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/resource_dump.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/io_file.h"
#include "common/path_util.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "shader_recompiler/info.h"

namespace VideoCore {

namespace {

/// One entry per stage per draw, so the manifest can be ordered against the
/// drawcall list of a RenderDoc capture.
std::atomic<u32> entry_index{0};

/// Raw bytes written since the process started; the manifest keeps growing
/// after the budget is spent so the descriptors are never lost.
std::atomic<u64> bytes_written{0};
constexpr u64 MAX_TOTAL_BYTES = 256_MB;

constexpr u64 MAX_BUFFER_BYTES = 1_MB;

std::filesystem::path DumpDirectory() {
    return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "resource_dump";
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
            text += ' ';
        }
        text += fmt::format("{:08x}", value);
    }
    return text;
}

} // namespace

void DumpDrawResources(const Shader::Info& stage, Core::MemoryManager* memory) {
    const u32 entry = entry_index.fetch_add(1) + 1;
    const u32 frame = DebugState.GetFrameNum();
    const char* name = StageName(stage.sw_stage);

    const auto directory = DumpDirectory();
    std::filesystem::create_directories(directory);

    std::string buffers;
    for (size_t index = 0; index < stage.buffers.size(); ++index) {
        const auto& desc = stage.buffers[index];
        if (desc.IsSpecial()) {
            continue;
        }
        const auto vsharp = desc.GetSharp(stage);
        if (!vsharp || vsharp.base_address == 0) {
            continue;
        }

        const u64 requested = std::min<u64>(vsharp.GetSize(), MAX_BUFFER_BYTES);
        const u64 size =
            requested == 0 ? 0 : memory->ClampRangeSize(vsharp.base_address, requested);
        std::string file;
        if (size != 0) {
            const u64 budget = bytes_written.load(std::memory_order_relaxed);
            if (budget + size <= MAX_TOTAL_BYTES) {
                std::vector<u8> bytes(size);
                memory->CopySparseMemory(vsharp.base_address, bytes.data(), size);
                file = fmt::format("f{:05d}_e{:05d}_{}_b{:02d}.bin", frame, entry, name,
                                   static_cast<u32>(index));
                Common::FS::IOFile out(directory / file, Common::FS::FileAccessMode::Create);
                out.Write(bytes);
                bytes_written.fetch_add(size, std::memory_order_relaxed);
            }
        }

        if (!buffers.empty()) {
            buffers += ',';
        }
        buffers += fmt::format(
            R"({{"b":{},"addr":"{:#x}","size":{},"stride":{},"records":{},"written":{},"file":"{}"}})",
            static_cast<u32>(index), static_cast<u64>(vsharp.base_address), size,
            vsharp.GetStride(), vsharp.num_records, desc.is_written, file);
    }

    const std::string user_data = UserDataString(stage);
    const std::string line = fmt::format(
        R"({{"frame":{},"entry":{},"stage":"{}","hash":"{:#x}","ud":"{}","buffers":[{}]}})"
        "\n",
        frame, entry, name, stage.pgm_hash, user_data, buffers);

    Common::FS::IOFile manifest(directory / "manifest.jsonl", Common::FS::FileAccessMode::Append);
    manifest.Write(line);
}

} // namespace VideoCore
