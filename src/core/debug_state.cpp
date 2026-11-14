// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/native_clock.h"
#include "common/singleton.h"
#include "common/string_util.h"
#include "common/thread.h"
#include "core/aerolib/aerolib.h"
#include "core/signals.h"
#include "debug_state.h"
#include "devtools/widget/common.h"
#include "libraries/kernel/time.h"
#include "libraries/system/msgdialog.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

using namespace DebugStateType;

DebugStateImpl& DebugState = *Common::Singleton<DebugStateImpl>::Instance();

bool DebugStateType::showing_debug_menu_bar = false;

static ThreadID ThisThreadID() {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return pthread_self();
#endif
}

static void PauseThread(ThreadID id) {
#ifdef _WIN32
    auto handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE, id);
    SuspendThread(handle);
    CloseHandle(handle);
#else
    pthread_kill(id, SIGSLEEP);
#endif
}

namespace {

struct HleSymbolEntry {
    u64 address;
    const char* name;
};

// Every HLE export is registered here at startup so a sampled program counter can be named.
std::mutex g_hle_symbols_mutex;
std::vector<HleSymbolEntry> g_hle_symbols;
bool g_hle_symbols_sorted = true;

#ifdef _WIN32
struct StallSampleState {
    u64 pc{};
    int identical_samples{};
    bool reported{};
};

// Only ever touched by the watchdog thread.
std::unordered_map<ThreadID, StallSampleState> g_stall_samples;

u64 SampleThreadPc(ThreadID id) {
    const HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, id);
    if (handle == nullptr) {
        return 0;
    }
    u64 pc{};
    if (SuspendThread(handle) != static_cast<DWORD>(-1)) {
        CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL;
        if (GetThreadContext(handle, &context)) {
            pc = context.Rip;
        }
        ResumeThread(handle);
    }
    CloseHandle(handle);
    return pc;
}

const HleSymbolEntry* FindHleSymbol(u64 pc) {
    if (!g_hle_symbols_sorted) {
        std::ranges::sort(g_hle_symbols, {}, &HleSymbolEntry::address);
        g_hle_symbols_sorted = true;
    }
    const auto it = std::upper_bound(
        g_hle_symbols.begin(), g_hle_symbols.end(), pc,
        [](const u64 value, const HleSymbolEntry& entry) { return value < entry.address; });
    if (it == g_hle_symbols.begin()) {
        return nullptr;
    }
    return &*(it - 1);
}

std::string DescribePc(u64 pc) {
    // A HLE thunk is tiny, so anything further than this belongs to a different export.
    constexpr u64 MaxThunkDistance = 0x1000;
    if (const auto* symbol = FindHleSymbol(pc);
        symbol != nullptr && pc - symbol->address < MaxThunkDistance) {
        return fmt::format("HLE {} +{:#x}", symbol->name, pc - symbol->address);
    }
    // Name the module so a wait in a system dll can be told apart from code spinning in the
    // emulator or in the recompiled guest.
    HMODULE module{};
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(pc), &module) != 0 &&
        module != nullptr) {
        wchar_t path[MAX_PATH]{};
        if (GetModuleFileNameW(module, path, MAX_PATH) != 0) {
            const std::wstring_view name{path};
            const auto slash = name.find_last_of(L"\\/");
            return fmt::format("{}+{:#x}",
                               Common::UTF16ToUTF8(slash == std::wstring_view::npos
                                                       ? name
                                                       : name.substr(slash + 1)),
                               pc - reinterpret_cast<u64>(module));
        }
    }
    constexpr u64 GuestAddressLimit = 0x10'0000'0000;
    if (pc < GuestAddressLimit) {
        return fmt::format("guest code at {:#x}", pc);
    }
    return fmt::format("host code at {:#x}", pc);
}
#endif

} // namespace

void RegisterHleFunctionSymbol(u64 host_address, const char* nid) {
    const auto* entry = Core::AeroLib::FindByNid(nid);
    const char* name = entry != nullptr ? entry->name : nid;
    std::scoped_lock lock{g_hle_symbols_mutex};
    g_hle_symbols.push_back({host_address, name});
    g_hle_symbols_sorted = g_hle_symbols.size() == 1;
}

void DebugStateImpl::ReportStalledGuestThreads() {
#ifdef _WIN32
    constexpr int SampleIntervalSec = 5;
    constexpr int SamplesUntilStall = 3;

    std::vector<std::pair<ThreadID, std::string>> threads;
    {
        std::scoped_lock lock{guest_threads_mutex};
        threads.reserve(guest_threads.size() + stall_watchdog_threads.size());
        for (const ThreadID id : guest_threads) {
            const auto it = guest_thread_names.find(id);
            threads.emplace_back(id,
                                 it != guest_thread_names.end() ? it->second : std::string{"?"});
        }
        for (const ThreadID id : stall_watchdog_threads) {
            const auto it = guest_thread_names.find(id);
            threads.emplace_back(id,
                                 it != guest_thread_names.end() ? it->second : std::string{"?"});
        }
    }

    const ThreadID self = GetCurrentThreadId();
    std::vector<u64> pcs(threads.size(), 0);
    for (size_t i = 0; i < threads.size(); ++i) {
        if (threads[i].first != self) {
            pcs[i] = SampleThreadPc(threads[i].first);
        }
    }

    std::scoped_lock lock{g_hle_symbols_mutex};
    for (size_t i = 0; i < threads.size(); ++i) {
        auto& state = g_stall_samples[threads[i].first];
        if (pcs[i] == 0) {
            state = {};
            continue;
        }
        if (pcs[i] == state.pc) {
            ++state.identical_samples;
        } else {
            state = {.pc = pcs[i], .identical_samples = 1, .reported = false};
        }
        const bool stalled = state.identical_samples >= SamplesUntilStall;
        if (!stalled || state.reported) {
            continue;
        }
        state.reported = true;
        LOG_WARNING(Kernel, "Guest thread '{}' (tid {}) has not moved for {}s, stuck at {}",
                    threads[i].second, threads[i].first,
                    state.identical_samples * SampleIntervalSec, DescribePc(pcs[i]));
    }
#endif
}

void StartGuestStallWatchdog() {
#ifdef _WIN32
    static std::atomic_bool started{false};
    if (started.exchange(true)) {
        return;
    }
    std::thread([] {
        Common::SetCurrentThreadName("StallWatchdog");
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            DebugState.ReportStalledGuestThreads();
        }
    }).detach();
#endif
}

static void ResumeThread(ThreadID id) {
#ifdef _WIN32
    auto handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE, id);
    ResumeThread(handle);
    CloseHandle(handle);
#else
    pthread_kill(id, SIGSLEEP);
#endif
}

void DebugStateImpl::AddCurrentThreadToGuestList() {
    std::lock_guard lock{guest_threads_mutex};
    const ThreadID id = ThisThreadID();
    guest_threads.push_back(id);
    guest_thread_names.insert_or_assign(id, Common::GetCurrentThreadName());
}

void DebugStateImpl::RemoveCurrentThreadFromGuestList() {
    std::lock_guard lock{guest_threads_mutex};
    const ThreadID id = ThisThreadID();
    std::erase_if(guest_threads, [&](const ThreadID& v) { return v == id; });
    guest_thread_names.erase(id);
}

void DebugStateImpl::RegisterStallWatchdogThread() {
    std::lock_guard lock{guest_threads_mutex};
    const ThreadID id = ThisThreadID();
    stall_watchdog_threads.push_back(id);
    guest_thread_names.insert_or_assign(id, Common::GetCurrentThreadName());
}

void DebugStateImpl::PauseGuestThreads() {
    using namespace Libraries::MsgDialog;
    std::unique_lock lock{guest_threads_mutex};
    if (is_guest_threads_paused) {
        return;
    }
    if (ShouldPauseInSubmit()) {
        waiting_submit_pause = false;
        should_show_frame_dump = true;
    }
    bool self_guest = false;
    ThreadID self_id = ThisThreadID();
    for (const auto& id : guest_threads) {
        if (id == self_id) {
            self_guest = true;
        } else {
            PauseThread(id);
        }
    }
    pause_time = Libraries::Kernel::Dev::GetClock()->GetUptime();
    is_guest_threads_paused = true;
    lock.unlock();
    if (self_guest) {
        PauseThread(self_id);
    }
}

void DebugStateImpl::ResumeGuestThreads() {
    std::lock_guard lock{guest_threads_mutex};
    if (!is_guest_threads_paused) {
        return;
    }

    u64 delta_time = Libraries::Kernel::Dev::GetClock()->GetUptime() - pause_time;
    Libraries::Kernel::Dev::GetInitialPtc() += delta_time;
    for (const auto& id : guest_threads) {
        ResumeThread(id);
    }
    is_guest_threads_paused = false;
}

void DebugStateImpl::RequestFrameDump(s32 count) {
    ASSERT(!DumpingCurrentFrame());
    gnm_frame_dump_request_count = count;
    frame_dump_list.clear();
    frame_dump_list.resize(count);
    const auto f = gnm_frame_count.load() + 1;
    for (size_t i = 0; i < count; ++i) {
        frame_dump_list[i].frame_id = f + i;
    }
    waiting_submit_pause = true;
}

void DebugStateImpl::PushQueueDump(QueueDump dump) {
    ASSERT(DumpingCurrentFrame());
    std::unique_lock lock{frame_dump_list_mutex};
    auto& frame = GetFrameDump();
    { // Find draw calls
        auto data = std::span{dump.data};
        auto initial_data = data.data();
        while (!data.empty()) {
            const auto* header = reinterpret_cast<const AmdGpu::PM4Type3Header*>(data.data());
            const auto type = header->type;
            if (type == 2) {
                data = data.subspan(1);
            } else if (type != 3) {
                UNREACHABLE();
            }
            const AmdGpu::PM4ItOpcode opcode = header->opcode;
            if (Core::Devtools::Widget::IsDrawCall(opcode)) {
                const auto offset =
                    reinterpret_cast<uintptr_t>(header) - reinterpret_cast<uintptr_t>(initial_data);
                const auto addr = dump.base_addr + offset;
                waiting_reg_dumps.emplace(addr, &frame);
                waiting_reg_dumps_dbg.emplace(
                    addr,
                    fmt::format("#{} h({}) queue {} {} {}",
                                frame_dump_list.size() - gnm_frame_dump_request_count, addr,
                                magic_enum::enum_name(dump.type), dump.submit_num, dump.num2));
            }
            data = data.subspan(header->NumWords() + 1);
        }
    }
    frame.queues.push_back(std::move(dump));
}

std::optional<RegDump*> DebugStateImpl::GetRegDump(uintptr_t base_addr, uintptr_t header_addr) {
    const auto it = waiting_reg_dumps.find(header_addr);
    if (it == waiting_reg_dumps.end()) {
        return std::nullopt;
    }
    auto& frame = *it->second;
    waiting_reg_dumps.erase(it);
    waiting_reg_dumps_dbg.erase(waiting_reg_dumps_dbg.find(header_addr));
    return &frame.regs[header_addr - base_addr];
}

void DebugStateImpl::PushRegsDump(uintptr_t base_addr, uintptr_t header_addr,
                                  const AmdGpu::Regs& regs) {
    std::scoped_lock lock{frame_dump_list_mutex};

    auto dump = GetRegDump(base_addr, header_addr);
    if (!dump) {
        return;
    }

    (*dump)->regs = regs;

    for (int i = 0; i < RegDump::MaxShaderStages; i++) {
        if ((*dump)->regs.stage_enable.IsStageEnabled(i)) {
            auto stage = (*dump)->regs.ProgramForStage(i);
            if (stage->address) {
                const auto params = AmdGpu::GetParams(*stage);
                (*dump)->stages[i] = PipelineShaderProgramDump{
                    .name = Vulkan::PipelineCache::GetShaderName(Shader::StageFromIndex(i),
                                                                 params.hash),
                    .hash = params.hash,
                    .user_data = *stage,
                    .code = std::vector<u32>{params.code.begin(), params.code.end()},
                };
            }
        }
    }
}

void DebugStateImpl::PushRegsDumpCompute(uintptr_t base_addr, uintptr_t header_addr,
                                         const CsState& cs_state) {
    std::scoped_lock lock{frame_dump_list_mutex};

    auto dump = GetRegDump(base_addr, header_addr);
    if (!dump) {
        return;
    }

    (*dump)->is_compute = true;
    auto& cs = (*dump)->regs.cs_program;
    cs = cs_state;

    const auto params = AmdGpu::GetParams(cs);
    (*dump)->cs_data = PipelineComputerProgramDump{
        .name = Vulkan::PipelineCache::GetShaderName(Shader::HwStage::Compute, params.hash),
        .hash = params.hash,
        .cs_program = cs,
        .code = std::vector<u32>{params.code.begin(), params.code.end()},
    };
}

void DebugStateImpl::CollectShader(const std::string& name, Shader::SwStage l_stage,
                                   vk::ShaderModule module, std::span<const u32> spv,
                                   std::span<const u32> raw_code, std::span<const u32> patch_spv,
                                   bool is_patched) {
    shader_dump_list.emplace_back(name, l_stage, module, std::vector<u32>{spv.begin(), spv.end()},
                                  std::vector<u32>{raw_code.begin(), raw_code.end()},
                                  std::vector<u32>{patch_spv.begin(), patch_spv.end()}, is_patched);
}
