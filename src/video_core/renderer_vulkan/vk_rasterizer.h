// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <unordered_map>
#include <unordered_set>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class Scheduler;
class RenderState;
class GraphicsPipeline;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size);
    bool ReadMemory(VAddr addr, u64 size);
    void ProcessDownloadImages();
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    void CpSync();
    u64 Flush();
    void Finish();
    void OnSubmit();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

private:
    void RecordGuestWindow(u32 scissor_width, u32 scissor_height);
    float PresentationScale() const;
    void ApplyPresentationScale(VideoCore::TextureCache::ImageDesc& desc) const;
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

    void ResetBindings() {
        for (auto& image_id : bound_images) {
            texture_cache.GetImage(image_id).binding = {};
        }
        bound_images.clear();
    }

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;

    using BufferBindingInfo = std::tuple<VideoCore::BufferId, AmdGpu::Buffer, u64>;
    boost::container::static_vector<BufferBindingInfo, Shader::NUM_BUFFERS> buffer_bindings;
    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
    bool fault_process_pending{};
    bool attachment_feedback_loop{};

    // VideoOut output auto-fit: the render area and the scissor of every pass drawing
    // into a registered VideoOut surface are kept at the surface extent, so a stale
    // attachment or scissor register cannot crop the frame to the top-left corner.
    // Without a resolution patch the surface matches the game's window and this has no
    // effect. The remaining fields are observations used by the diagnostics.
    mutable u16 vo_surface_width{};
    mutable u16 vo_surface_height{};
    // True when the pass still clips to a window smaller than the surface, with the
    // ratio between the two.
    mutable bool output_upscaled{};
    mutable float vo_fit_x{1.0f};
    mutable float vo_fit_y{1.0f};
    // Ratio last observed on a pass that still clips to the game's original window.
    // It reveals the window the geometry of every output pass is built for, including
    // the passes whose own scissor registers were already updated to the full surface.
    float vo_known_fit_x{1.0f};
    float vo_known_fit_y{1.0f};
    // The window the game's own passes clip to, which is what its geometry is laid out
    // for. The ratio between it and the output surface is the scale a resolution patch
    // left the offscreen targets missing, and it is one when no patch is active.
    u32 guest_window_width{};
    u32 guest_window_height{};
    // Ratio applied to the current pass because it renders into an offscreen target that
    // was enlarged to the presentation scale.
    mutable float rt_fit_x{1.0f};
    mutable float rt_fit_y{1.0f};
    // Extent that target was enlarged to, used to tell whether the pass viewport already
    // covers it and must not be stretched again.
    mutable u32 rt_fit_width{};
    mutable u32 rt_fit_height{};
    // Addresses of the offscreen targets that are being rendered at the presentation
    // scale. Shaders sampling them describe the original size, so the same adjustment has
    // to be applied on the sampling path or the lookup would miss the enlarged image.
    std::unordered_set<VAddr> upscaled_targets;
    // Set while the current draw renders into a registered VideoOut surface, used to
    // scope the output-composition diagnostics.
    mutable bool vo_pass{};
    // Set while the current draw presents an offscreen target that was rendered at the
    // presentation scale. Its contents are already at the full size, so it must not
    // receive the window-to-surface ratio the other output passes need.
    mutable bool presents_upscaled{};
    // Vertex shader hash of the current draw, for the per-shader viewport rule below.
    mutable u64 current_vs_hash{};
    // Viewport register signatures observed per vertex shader on the output surface:
    // bit 1 marks window-sized registers, bit 2 surface-sized ones. The resolution
    // patch rewrites only some of the constants the game derives its registers and
    // vertex conversion bases from, so a shader whose geometry is laid out for the
    // original window can arrive with window-sized registers in one batch and
    // surface-sized registers in another. That dual signature is the fingerprint of
    // such a shader: a consistent one would never need both.
    std::unordered_map<u64, u8> vs_viewport_signatures{};
    // Vertex shaders confirmed to carry the dual signature. Their geometry is built on
    // the window basis (the matrices divide window pixels by the window half-size even
    // though the positions were already scaled to the surface), so their viewport is
    // pinned to the guest window instead of the surface or a stretched variant.
    std::unordered_set<u64> window_space_vs_hashes{};
    // Hash of the previous clip-enabled VideoOut pass and how many times it drew in a
    // row. Sprite batches (crowd, glowsticks, note effects) submit the same vertex
    // shader hundreds of times per frame, while the interface elements each draw a
    // handful of times, so a long run of one hash identifies a window-space sprite
    // shader without needing the dual-signature observation, which only completes
    // when both scenes using the shader are visited in the same session.
    u64 last_vo_vs_hash{};
    u32 vo_burst_count{};
};

} // namespace Vulkan
