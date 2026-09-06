// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "common/debug.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Vulkan {

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
}

Rasterizer::~Rasterizer() = default;

void Rasterizer::CpSync() {
    scheduler.EndRendering();
    auto cmdbuf = scheduler.CommandBuffer();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

// Returns whether any stage of the pipeline samples a texture that starts at the given
// address, which identifies a pass reading the same allocation it renders into.
static bool SamplesAddress(const GraphicsPipeline* pipeline, VAddr address) {
    if (address == 0) {
        return false;
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        for (const auto& image_desc : stage->images) {
            if (image_desc.GetSharp(*stage).Address() == address) {
                return true;
            }
        }
    }
    return false;
}

// Returns whether the pass must stretch its viewport to fill the output surface.
// Two geometry bases coexist under the 4K patch:
//  - passes whose viewport registers were NOT converted (their raw viewport width is
//    half of the surface) lay everything out for the original 1080p window and always
//    need the doubling. This is where the gameplay notes/stars and every other batch
//    the patch missed live, and no shader hash can decide them because the same vertex
//    shader also runs in converted batches: the raw viewport width does.
//  - converted passes (raw viewport width equals the surface) span either the full NDC
//    (the matrix-driven and procedural passes, which keep the 4K viewport) or half of
//    it (the 1080p UI-sheet binaries listed below, which need the doubled viewport).
// The glyph/font batches are NOT in the list: the patch converts their uScale transform
// too, so their quads already span the full NDC at the 4K viewport and stretching them
// misplaces every character text.
static bool OutputSpriteNeedsStretch(u64 vs_pgm_hash, float raw_vp_width, u32 surface_width) {
    if (raw_vp_width > 0.f && raw_vp_width < float(surface_width) * 0.75f) {
        return true; // unconverted viewport: geometry still laid out for the 1080p window
    }
    switch (vs_pgm_hash) {
    case 0x00000000788fc913ULL: // glyph quad shader: maps the 1080p window to half the
                                // NDC, so it needs the doubled 4K viewport (verified by
                                // post-VS geometry in a capture)
    case 0x000000000ec3717aULL: // UI layer quad: draws the 1080p UI sheet and widget textures
        return true;
    default:
        return false;
    }
}

// Returns whether any stage samples one of the offscreen targets that are rendered at
// the presentation scale. Such a pass is presenting the scene we already rasterized at
// the full size, so its geometry needs no further stretching, while a pass that reads
// none of them is drawing content still laid out for the game's original window.
static bool SamplesUpscaledTarget(const GraphicsPipeline* pipeline,
                                  const std::unordered_set<VAddr>& upscaled_targets) {
    if (upscaled_targets.empty()) {
        return false;
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        for (const auto& image_desc : stage->images) {
            if (upscaled_targets.contains(image_desc.GetSharp(*stage).Address())) {
                return true;
            }
        }
    }
    return false;
}

// Enlarges a color target descriptor to the presentation scale when it is one of the
// offscreen targets the game sized for its own window. The render path, the sampling
// path and the resolve path all have to agree on which targets are enlarged and by how
// much: each of them looks images up by extent, so a path that describes a target at a
// different size than the one it was created at looks up a second image over the same
// memory and reads or writes an allocation nobody else touches.
//
// The output surface itself is excluded: it is already sized for presentation, and the
// composition draws into it from the enlarged targets.
//
// Enlarging is only correct while a resolution patch is active, which is what makes the
// game render into a surface larger than the window its passes are built for. Without
// one, the surface and the window agree and nothing is missing any scale.
//
// The ratio is the surface against that one window, and the window has to be identified
// before the ratio means anything. Deriving it from each pass' own scissor does not
// work: the game deliberately renders its bloom chain at half, quarter and eighth of
// the window, so every one of those levels reports a ratio that would scale it to the
// full surface. Enlarging them blurred the whole frame, which is the ghosting seen at
// 1080p, where the surface equals the window and nothing should have been touched at
// all.
//
// The window is the region the game's own passes clip to. Take it only from passes
// drawing into an offscreen target: the ones drawing into the output surface clip to the
// surface itself and would report the window as being the surface, collapsing the ratio
// to one.
//
// Require the candidate to cover more than half the surface. The bloom chain clips to a
// half, a quarter and an eighth of the window on purpose, and any of those would answer
// with a whole ratio of its own (a 960-wide level reports 4x against a 3840 surface),
// which is what enlarged them to the full frame and blurred it.
void Rasterizer::RecordGuestWindow(u32 scissor_width, u32 scissor_height) {
    const auto vo_ext = liverpool->GetVideoOutExtent();
    if (!vo_ext.Valid() || scissor_width == 0 || scissor_height == 0) {
        return;
    }
    if (scissor_width > vo_ext.width || scissor_height > vo_ext.height) {
        return;
    }
    if (scissor_width * 2 < vo_ext.width || scissor_height * 2 < vo_ext.height) {
        return;
    }
    const u32 prev_width = guest_window_width;
    const u32 prev_height = guest_window_height;
    guest_window_width = std::max(guest_window_width, scissor_width);
    guest_window_height = std::max(guest_window_height, scissor_height);
    if (guest_window_width != prev_width || guest_window_height != prev_height) {
        LOG_INFO(Render_Vulkan, "Guest window: {}x{} against surface {}x{}, presentation scale {}",
                 guest_window_width, guest_window_height, vo_ext.width, vo_ext.height,
                 PresentationScale());
    }
}

float Rasterizer::PresentationScale() const {
    const auto vo_ext = liverpool->GetVideoOutExtent();
    if (!vo_ext.Valid() || guest_window_width == 0 || guest_window_height == 0) {
        return 1.0f;
    }
    const float fit_x = float(vo_ext.width) / float(guest_window_width);
    const float fit_y = float(vo_ext.height) / float(guest_window_height);
    // Only a uniform enlargement is handled, and only a whole one: the patch doubles
    // both axes together, and a fractional ratio would describe a window the geometry
    // was never laid out for.
    if (std::abs(fit_x - fit_y) > 0.01f || std::abs(fit_x - std::round(fit_x)) > 0.01f) {
        return 1.0f;
    }
    return fit_x;
}

void Rasterizer::ApplyPresentationScale(VideoCore::TextureCache::ImageDesc& desc) const {
    const float fit = PresentationScale();
    if (fit <= 1.001f) {
        return;
    }
    const auto vo_ext = liverpool->GetVideoOutExtent();
    if (!vo_ext.Valid() || desc.info.size.width == 0 || desc.info.size.height == 0) {
        return;
    }
    if (liverpool->FindVideoOutSurface(desc.info.guest_address)) {
        return;
    }
    // Only the targets the game sized for exactly that window are missing the scale.
    // A target it allocated at some other size is a deliberate choice, and the passes
    // reading it scale their coordinates themselves.
    if (desc.info.size.width != guest_window_width ||
        desc.info.size.height != guest_window_height) {
        return;
    }
    desc.info.size.width = vo_ext.width;
    desc.info.size.height = vo_ext.height;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    output_upscaled = false;
    vo_surface_width = 0;
    vo_surface_height = 0;
    vo_fit_x = 1.0f;
    vo_fit_y = 1.0f;
    vo_pass = false;
    presents_upscaled = false;
    rt_fit_x = 1.0f;
    rt_fit_y = 1.0f;
    rt_fit_width = 0;
    rt_fit_height = 0;
    AmdGpu::CbDbExtent vo_extent{};
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    // Identify the window the game's geometry is laid out for before any target is
    // looked up, because the enlargement rule is derived from it. Only offscreen passes
    // describe that window; a pass drawing into the output surface clips to the surface.
    if (const auto& cb0 = regs.color_buffers[0];
        cb0 && !liverpool->FindVideoOutSurface(cb0.Address())) {
        RecordGuestWindow(AmdGpu::Scissor::Clamp(regs.screen_scissor.bottom_right_x),
                          AmdGpu::Scissor::Clamp(regs.screen_scissor.bottom_right_y));
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        // The game sizes its offscreen scene targets for its own window, so the frame
        // stays at that resolution no matter how large the output surface is: the
        // composition can only upscale what it is given. Rasterize them at the
        // presentation scale instead to get real detail rather than a stretched image.
        //
        // Only the host extent may grow. The pitch, the guest size and the mip layout
        // describe how the allocation is laid out in guest memory and drive tiling,
        // uploads and the cache lookup, so scaling them would describe a stride the
        // guest never wrote and the detiler would unpack garbage. Uploads clamp their
        // copy extent to the guest pitch, so leaving them alone is safe.
        //
        // A clip-disabled pass that samples the very allocation it draws into is an
        // in-place blit. Those are harmless while the source and the destination are the
        // same size, because the mapping is the identity, but enlarging the target turns
        // it into a scaling blit that reads pixels it has already written, so the result
        // collapses to black. Leave such targets at the size the game chose.
        const bool in_place_blit =
            regs.IsClipDisabled() && SamplesAddress(pipeline, col_buf.Address());
        if (const u32 sharp_width = desc.info.size.width, sharp_height = desc.info.size.height;
            !in_place_blit) {
            // Go through the shared rule so the resolve path, which rebuilds these
            // descriptors from the same registers, enlarges exactly the same targets.
            ApplyPresentationScale(desc);
            if (desc.info.size.width != sharp_width) {
                rt_fit_x = float(desc.info.size.width) / float(sharp_width);
                rt_fit_y = float(desc.info.size.height) / float(sharp_height);
                rt_fit_width = desc.info.size.width;
                rt_fit_height = desc.info.size.height;
                upscaled_targets.insert(desc.info.guest_address);
            }
        }
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;

        {
            // Report every distinct render target with the register state it was derived
            // from, to locate the ones the game still allocates at its original size.
            // The address is part of the key: two allocations that happen to share a
            // size and a format are distinct targets, and merging their reports hid
            // the scene buffer behind another one of the same shape.
            static std::unordered_set<u64> logged_rt;
            const u64 rt_key = (u64(col_buf.Address() >> 12) << 40) ^
                               (u64(image.info.size.width) << 28) ^
                               (u64(image.info.size.height) << 16) ^
                               u64(static_cast<u32>(image.info.pixel_format));
            if (logged_rt.insert(rt_key).second) {
                LOG_INFO(Render_Vulkan,
                         "RT: {}x{} fmt={} addr={:#x} pitch={} regPitch={} regHeight={} "
                         "hint={}x{} valid={} scsr={}x{} tileMax={} sliceMax={} tileMode={} "
                         "sliceSize={:#x} guestSize={:#x} samples={} upscaled={}",
                         image.info.size.width, image.info.size.height,
                         vk::to_string(image.info.pixel_format), col_buf.Address(),
                         image.info.pitch, col_buf.Pitch(), col_buf.Height(), hint.width,
                         hint.height, hint.Valid(),
                         AmdGpu::Scissor::Clamp(regs.screen_scissor.bottom_right_x),
                         AmdGpu::Scissor::Clamp(regs.screen_scissor.bottom_right_y),
                         col_buf.pitch.tile_max, col_buf.slice.tile_max,
                         static_cast<u32>(col_buf.GetTileMode()), col_buf.GetColorSliceSize(),
                         image.info.guest_size, image.info.num_samples,
                         upscaled_targets.contains(image.info.guest_address));
            }
        }

        // A registered VideoOut surface defines the presentation window. When the game
        // still clips to a smaller window than the surface it renders into (e.g. a
        // resolution patch enlarged the output buffer and the vertex positions, but the
        // scissor registers kept the original size), the frame gets cropped to the
        // top-left corner. Record the surface extent so the scissor can be opened up.
        // The lookup goes through the registration table rather than the cached image
        // because the texture cache may replace the image object while the registration
        // stays valid.
        if (const auto* vo = liverpool->FindVideoOutSurface(col_buf.Address()); vo) {
            const auto& vp = regs.viewports[0];
            const u32 scsr_w = AmdGpu::Scissor::Clamp(regs.screen_scissor.bottom_right_x);
            const u32 scsr_h = AmdGpu::Scissor::Clamp(regs.screen_scissor.bottom_right_y);
            vo_pass = true;
            // Align every pass that renders into this surface to its full extent so
            // all of them share one depth attachment of a matching size, regardless
            // of whether the pass itself needs adjusting.
            vo_extent.width = std::max(vo_extent.width, vo->width);
            vo_extent.height = std::max(vo_extent.height, vo->height);
            vo_surface_width = std::max(vo_surface_width, vo->width);
            vo_surface_height = std::max(vo_surface_height, vo->height);
            if (scsr_w < vo->width || scsr_h < vo->height) {
                // This pass still clips to the original window, so the ratio between the
                // surface and that window is the scale its geometry is missing. Remember
                // both: the passes that already clip to the full surface cannot derive
                // the ratio themselves, and the offscreen targets the game allocates at
                // that window size have to be recognised later on.
                if (scsr_w > 0 && scsr_h > 0) {
                    vo_known_fit_x = float(vo->width) / float(scsr_w);
                    vo_known_fit_y = float(vo->height) / float(scsr_h);
                }
            }
            // Apply the ratio to every pass targeting the surface. Excluding the passes
            // that present an enlarged target was tried and moved the composition to the
            // top-left quarter of the screen, which is where it lands with no scaling at
            // all, so the ratio is what puts that blit over the whole surface and it is
            // correct for every output pass. Keep tracking what the pass reads for the
            // diagnostics, since that is the only thing separating the composition from
            // the interface.
            presents_upscaled = SamplesUpscaledTarget(pipeline, upscaled_targets);
            if (vo_known_fit_x > 1.001f || vo_known_fit_y > 1.001f) {
                vo_fit_x = vo_known_fit_x;
                vo_fit_y = vo_known_fit_y;
                output_upscaled = true;
            }
            // Report every pass targeting the output surface, including the ones that
            // need no adjusting, so the whole composition can be reconstructed.
            // The shader hashes AND the viewport registers are part of the key: the
            // resolution patch rewrites only some of the constants the game derives its
            // viewport registers and vertex conversion bases from, so one shader can
            // arrive with 1080p registers in one batch and 4K registers in another
            // (verified in a capture: the same sprite vertex shader appears with a
            // 1920-wide viewport in one draw and a 3840-wide one in the next). The
            // geometry basis of each shader family (1920, 3840 or 7680) decides the
            // viewport it needs, and that basis can only be mapped by logging every
            // shader-plus-register combination and joining it with the post-VS
            // geometry classification from a RenderDoc capture.
            static std::unordered_set<u64> logged;
            const u64 vs_hash = key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Vertex)];
            const u64 fs_hash = key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Fragment)];
            const u64 log_key = (u64(scsr_w) << 32) | (u64(scsr_h) << 16) |
                                (u64(static_cast<u32>(regs.primitive_type)) << 4) |
                                (regs.IsClipDisabled() ? 2u : 0u) |
                                (regs.viewport_control.xscale_enable ? 1u : 0u);
            const u64 log_id = log_key ^ (vs_hash * 0x9E3779B97F4A7C15ull >> 6) ^ (fs_hash >> 17) ^
                               (u64(std::bit_cast<u32>(vp.xoffset)) << 20) ^
                               (u64(std::bit_cast<u32>(vp.xscale)) >> 3);
            if (logged.insert(log_id).second) {
                LOG_INFO(Render_Vulkan,
                         "VideoOut pass: surface {}x{}, prim={}, clipDisabled={}, vte=({},{}), "
                         "vp=({},{},{},{}), screenScissor={}x{}, mrt={:#x}, opened={}, fit={}x{}, "
                         "vsHash={:#x}, fsHash={:#x}",
                         vo->width, vo->height, static_cast<u32>(regs.primitive_type),
                         regs.IsClipDisabled(), regs.viewport_control.xscale_enable,
                         regs.viewport_control.yscale_enable, vp.xoffset, vp.yoffset, vp.xscale,
                         vp.yscale, scsr_w, scsr_h, key.mrt_mask, output_upscaled, vo_fit_x,
                         vo_fit_y, vs_hash, fs_hash);
            }
        }
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        auto hint = liverpool->last_db_extent;
        if (vo_extent.Valid() && (vo_extent.width > hint.width || vo_extent.height > hint.height)) {
            // The game still allocates the depth buffer for its original window size.
            // Enlarge it to match the VideoOut target, otherwise the smaller depth
            // attachment would shrink the framebuffer back to the old resolution and
            // clip the upscaled pass to its top-left corner.
            hint = vo_extent;
        }
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        if (rt_fit_x > 1.001f) {
            // Follow the enlarged color target, otherwise the smaller depth attachment
            // would shrink the framebuffer back and crop the pass. As with the color
            // target, only the host extent grows: the guest layout has to keep
            // describing the allocation the game actually wrote.
            desc.info.size.width = u32(desc.info.size.width * rt_fit_x);
            desc.info.size.height = u32(desc.info.size.height * rt_fit_y);
        }
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const auto& regs = liverpool->regs;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(index_offset, buffer_barriers);
    }

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }

    ResetBindings();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(0, buffer_barriers);
    }

    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }
    if (count_buffer) {
        if (auto barrier = count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                    vk::PipelineStageFlagBits2::eDrawIndirect)) {
            buffer_barriers.emplace_back(*barrier);
        }
    }

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_base, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_base,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
    }

    ResetBindings();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);

    ResetBindings();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);

    ResetBindings();
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    if (!pipeline->IsCompute() && liverpool->regs.IsClipDisabled()) {
        // Only the graphics path may consult the fit ratios. A dispatch does not go
        // through PrepareRenderState, so on that path the ratios and the color buffer
        // registers still describe whichever draw ran before it, and scaling by them
        // would apply one pass' correction to an unrelated one.
        //
        // Clip-disabled passes pin the Vulkan viewport to the hardware window and turn
        // vertex positions into window coordinates inside the shader through push data
        // (see ConvertPositionToClipSpace), so the viewport stretch applied below can
        // never reach them and the conversion itself has to carry the scale.
        //
        // The composition shaders build their quad from VertexIndex as x in [-1, 1] and
        // y in [-1, 1] with normalized 0..1 texture coordinates (verified in the dumps:
        // the vertices are (-1,-1), (1,-1), (-1,1) with UV (0,1), (1,1), (0,0)), so the
        // conversion spans 2*scale on both axes and the game's viewport registers already
        // map that quad onto the whole target they were sized for. Multiplying the terms
        // by the same ratio the target was enlarged by maps the identical quad onto the
        // whole enlarged target instead; an extra factor would push half of it off-screen
        // and leave the source sampled only halfway across.
        if (output_upscaled) {
            // A pass drawing into the output surface works at the presentation scale.
            push_data.xoffset *= vo_fit_x;
            push_data.xscale *= vo_fit_x;
            push_data.yoffset *= vo_fit_y;
            push_data.yscale *= vo_fit_y;
        } else if (rt_fit_x > 1.001f) {
            // A pass drawing into an offscreen target we enlarged is rasterized at the
            // presentation scale as well: without the ratio its quad would only fill
            // the corner of the target the game sized, and every pass sampling that
            // target would read the shrunk scene next to uninitialized memory.
            push_data.xoffset *= rt_fit_x;
            push_data.xscale *= rt_fit_x;
            push_data.yoffset *= rt_fit_y;
            push_data.yscale *= rt_fit_y;
            // Report every distinct pass the ratio is applied to, with the conversion
            // it ends up with, to verify the quad then covers the enlarged target.
            static std::unordered_set<u64> logged_pp;
            const u64 k = (u64(liverpool->regs.color_buffers[0].Address()) << 24) ^
                          (u64(std::bit_cast<u32>(push_data.xoffset)) << 2) ^
                          u64(std::bit_cast<u32>(push_data.yscale));
            if (logged_pp.insert(k).second) {
                const auto* gp = dynamic_cast<const GraphicsPipeline*>(pipeline);
                const auto& key = gp->GetGraphicsKey();
                const u64 vs_h = key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Vertex)];
                const u64 fs_h = key.stage_hashes[static_cast<u32>(Shader::LogicalStage::Fragment)];
                LOG_INFO(Render_Vulkan,
                         "Upscaled RT clip-disabled pass: cb0={:#x}, fit={}x{}, "
                         "rtFitExtent={}x{}, push=({},{},{},{}), vsHash={:#x}, fsHash={:#x}",
                         liverpool->regs.color_buffers[0].Address(), rt_fit_x, rt_fit_y,
                         rt_fit_width, rt_fit_height, push_data.xoffset, push_data.yoffset,
                         push_data.xscale, push_data.yscale, vs_h, fs_h);
            }
        }
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        stage->PushUd(binding, push_data);
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        // We only use fault buffer for DMA right now.
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (auto& range : mapped_ranges) {
            buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
        }
        fault_process_pending = true;
    }

    return true;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
        dst_image.CopyImage(src_image);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    dst_image.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    image1.flags |= VideoCore::ImageFlagBits::GpuModified;
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    buffer_bindings.clear();

    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(stage);
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            const auto buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            buffer_bindings.emplace_back(buffer_id, vsharp, size);
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
        }
    }

    // Second pass to re-bind buffers that were updated after binding
    for (u32 i = 0; i < buffer_bindings.size(); i++) {
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        // Buffer is not from the cache, either a special buffer or unbound.
        if (!buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id);
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            ASSERT(adjust % 4 == 0);
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            const vk::AccessFlags2 access_mask =
                desc.is_written
                    ? vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
                    : vk::AccessFlagBits2::eShaderRead;
            if (auto barrier =
                    vk_buffer->GetBarrier(access_mask, vk::PipelineStageFlagBits2::eAllCommands)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (desc.is_written && desc.is_formatted) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eUniformBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the size of each consecutive array with the number of bindings consumed.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::tuple{tsharp, image_desc});

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            // A target that is rendered at the presentation scale is described by the
            // shader at its original size, so the lookup has to be adjusted the same way
            // or it would create a second image over the same memory and sample an empty
            // one instead of the rendered contents. Only the extent is adjusted here as
            // well, to match how the target itself was enlarged.
            //
            // Repeat the extent check the render path used rather than trusting the
            // address alone: these allocations are recycled for surfaces of other sizes,
            // and enlarging one of those would look up an image that was never rendered.
            bool report_sampling = false;
            bool sampling_adjusted = false;
            u32 sharp_width = 0;
            u32 sharp_height = 0;
            if (!upscaled_targets.empty() && upscaled_targets.contains(desc.info.guest_address)) {
                sharp_width = desc.info.size.width;
                sharp_height = desc.info.size.height;
                // Go through the same rule the render and the resolve paths use, so the
                // lookup describes the target at the extent it was created at.
                ApplyPresentationScale(desc);
                sampling_adjusted = desc.info.size.width != sharp_width;
                report_sampling = true;
            }

            image_id = texture_cache.FindImage(desc);
            auto* image = &texture_cache.GetImage(image_id);

            if (report_sampling) {
                // Report the image the lookup actually resolved to, and keep reporting
                // it periodically. Logging only the first occurrence hides the steady
                // state behind the first frame, where the source has not been rendered
                // yet and a black sample is expected. An adjusted descriptor can also
                // resolve to a different image than the one the render path created,
                // and a resolved extent that is not the enlarged one, or one without
                // GpuModified, would each sample black with correct geometry.
                static std::unordered_map<u64, u32> smp_hits;
                const u64 k = (u64(desc.info.guest_address) << 24) ^
                              (u64(desc.info.size.width) << 12) ^ desc.info.size.height;
                if (++smp_hits[k] % 600 == 1) {
                    LOG_INFO(Render_Vulkan,
                             "Sampling upscaled target: addr={:#x}, sharp={}x{}, pitch={}, "
                             "adjusted={}, lookup={}x{}, resolved={}x{}, gpuModified={}, "
                             "cpuDirty={}",
                             desc.info.guest_address, sharp_width, sharp_height, desc.info.pitch,
                             sampling_adjusted, desc.info.size.width, desc.info.size.height,
                             image->info.size.width, image->info.size.height,
                             True(image->flags & VideoCore::ImageFlagBits::GpuModified),
                             True(image->flags & VideoCore::ImageFlagBits::CpuDirty));
                }
            }

            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (auto& [image_id, desc] : image_bindings) {
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);

            if (rt_fit_x > 1.001f) {
                // Report every image a pass drawing into an enlarged target reads. The
                // scene has to reach the composition source through some chain of
                // post-process passes, and an earlier report of the sampling side showed
                // no pass ever reading the scene buffer itself, so the chain is not
                // visible in the log at all. Log each input periodically, alongside the
                // output it feeds, so the whole chain can be reconstructed.
                static std::unordered_map<u64, u32> pp_hits;
                const u64 k = (u64(liverpool->regs.color_buffers[0].Address()) << 24) ^
                              (u64(image.info.guest_address) << 12) ^ image.info.size.width;
                if (++pp_hits[k] % 600 == 1) {
                    LOG_INFO(Render_Vulkan,
                             "Post-process input: out={:#x}, reads {}x{} addr={:#x}, pitch={}, "
                             "gpuModified={}, upscaled={}",
                             liverpool->regs.color_buffers[0].Address(), image.info.size.width,
                             image.info.size.height, image.info.guest_address, image.info.pitch,
                             True(image.flags & VideoCore::ImageFlagBits::GpuModified),
                             upscaled_targets.contains(image.info.guest_address));
                }
            }

            if (vo_pass) {
                // Report what every pass targeting the output surface reads, and how its
                // positions are converted, to tell apart a geometry problem from a
                // sampling one. Periodically, so the steady state is visible and not only
                // the first frame, where the source has not been rendered yet.
                //
                // Clip-enabled passes are included. The game draws 3D content straight
                // into the surface with a real projection, and restricting this report to
                // the clip-disabled blits hid those passes entirely, which is what left
                // the source of the scene magnification invisible in the log.
                static std::unordered_map<u64, u32> blit_hits;
                const u64 k = (u64(image.info.guest_address) << 24) ^
                              (u64(image.info.size.width) << 12) ^ image.info.size.height ^
                              (u64(static_cast<u32>(liverpool->regs.primitive_type)) << 40);
                if (++blit_hits[k] % 600 == 1) {
                    const auto& vp = liverpool->regs.viewports[0];
                    LOG_INFO(Render_Vulkan,
                             "VideoOut source: reads {}x{} addr={:#x}, pitch={}, "
                             "guestSize={:#x}, gpuModified={}, upscaled={}, "
                             "vp=({},{},{},{}), fit={}x{}, numIndices={}, prim={}, "
                             "clipDisabled={}",
                             image.info.size.width, image.info.size.height,
                             image.info.guest_address, image.info.pitch, image.info.guest_size,
                             True(image.flags & VideoCore::ImageFlagBits::GpuModified),
                             upscaled_targets.contains(image.info.guest_address), vp.xoffset,
                             vp.yoffset, vp.xscale, vp.yscale, vo_fit_x, vo_fit_y,
                             liverpool->regs.num_indices,
                             static_cast<u32>(liverpool->regs.primitive_type),
                             liverpool->regs.IsClipDisabled());
                }
            }

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead |
                                  (image.info.props.is_depth
                                       ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                       : vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead),
                              {});
            } else {
                if (is_storage) {
                    image.Transit(vk::ImageLayout::eGeneral,
                                  vk::AccessFlagBits2::eShaderRead |
                                      vk::AccessFlagBits2::eShaderWrite,
                                  desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead,
                                  desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(stage);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        const auto vk_sampler = texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop = true;
        } else {
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear = regs.depth_render_control.depth_clear_enable ||
                                    texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    if (vo_pass && vo_surface_width > 0 && vo_surface_height > 0) {
        // The render area is the intersection of all attachments, so an attachment the
        // game still sized for the original window shrinks it and crops the frame no
        // matter how wide the scissor is. Keep the area at the output surface extent.
        state.width = std::max<u16>(state.width, vo_surface_width);
        state.height = std::max<u16>(state.height, vo_surface_height);
    }

    if (rt_fit_x > 1.001f) {
        // The render area is the intersection of all attachments, so any attachment that
        // was not enlarged along with the target shrinks it back and would crop the pass
        // to the area of the original window. Keep the area at the enlarged extent.
        if (cb_descs[0].first) {
            const auto& cb0 = texture_cache.GetImage(cb_descs[0].first).info;
            state.width = std::max<u32>(state.width, cb0.size.width);
            state.height = std::max<u32>(state.height, cb0.size.height);
        }

        // Report what a pass drawing into an enlarged target actually gets, so a target
        // that is never rendered at its new size can be told apart from one that is
        // rendered correctly but sampled wrong.
        static std::unordered_set<u64> logged_up;
        const auto& vp = liverpool->regs.viewports[0];
        const VAddr cb0_addr =
            cb_descs[0].first ? texture_cache.GetImage(cb_descs[0].first).info.guest_address : 0;
        const u64 up_key = (u64(cb0_addr) << 20) ^ (u64(state.width) << 12) ^
                           (u64(static_cast<u32>(liverpool->regs.primitive_type)) << 4) ^
                           (u64(u32(std::abs(vp.xscale))) << 32) ^
                           (state.depth_stencil_attachment.has_depth ? 1u : 0u);
        if (logged_up.insert(up_key).second) {
            LOG_INFO(
                Render_Vulkan,
                "Upscaled RT pass: area={}x{}, cb0={}x{} addr={:#x}, db={}x{}, depth={}, "
                "vp=({},{},{},{}), screenScissor={}x{}, prim={}, clipDisabled={}",
                state.width, state.height,
                cb_descs[0].first ? texture_cache.GetImage(cb_descs[0].first).info.size.width : 0,
                cb_descs[0].first ? texture_cache.GetImage(cb_descs[0].first).info.size.height : 0,
                cb_descs[0].first ? texture_cache.GetImage(cb_descs[0].first).info.guest_address
                                  : 0,
                db_desc.first ? texture_cache.GetImage(db_desc.first).info.size.width : 0,
                db_desc.first ? texture_cache.GetImage(db_desc.first).info.size.height : 0,
                state.depth_stencil_attachment.has_depth, vp.xoffset, vp.yoffset, vp.xscale,
                vp.yscale, AmdGpu::Scissor::Clamp(liverpool->regs.screen_scissor.bottom_right_x),
                AmdGpu::Scissor::Clamp(liverpool->regs.screen_scissor.bottom_right_y),
                static_cast<u32>(liverpool->regs.primitive_type), liverpool->regs.IsClipDisabled());
        }
    }

    if (vo_pass) {
        // The render area is the intersection of all attachments, so a stale
        // attachment shrinks it and crops the frame regardless of the scissor.
        static std::unordered_set<u64> logged_area;
        const u64 area_key = (u64(state.width) << 32) | (u64(state.height) << 16) |
                             (u64(state.num_color_attachments) << 2) |
                             (state.depth_stencil_attachment.has_depth ? 1u : 0u);
        if (logged_area.insert(area_key).second) {
            LOG_INFO(
                Render_Vulkan,
                "VideoOut render area: {}x{}, colors={}, depth={}, cb0={}x{}, db={}x{}",
                state.width, state.height, state.num_color_attachments,
                state.depth_stencil_attachment.has_depth,
                cb_descs[0].first ? texture_cache.GetImage(cb_descs[0].first).info.size.width : 0,
                cb_descs[0].first ? texture_cache.GetImage(cb_descs[0].first).info.size.height : 0,
                db_desc.first ? texture_cache.GetImage(db_desc.first).info.size.width : 0,
                db_desc.first ? texture_cache.GetImage(db_desc.first).info.size.height : 0);
        }
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    // The resolve pass hands the multisampled scene to the single-sampled buffer the
    // post-process chain reads. The render path enlarges the offscreen targets to the
    // presentation scale, so both sides of this transfer have to be looked up at that
    // scale as well: rebuilding the descriptors from the registers alone describes them
    // at the size the game chose, which would look up a second image over the same
    // memory for the source and leave the destination at the original size, so the
    // enlarged scene would be resolved into a quarter-sized buffer and everything
    // downstream of it would read an image that was never written.
    const u32 mrt0_sharp_width = mrt0_desc.info.size.width;
    const u32 mrt1_sharp_width = mrt1_desc.info.size.width;
    ApplyPresentationScale(mrt0_desc);
    ApplyPresentationScale(mrt1_desc);
    // Record both sides the way the render path records the targets it enlarges. The
    // destination of a resolve never goes through the render path, so nothing else
    // would ever record it, and the post-process passes reading it describe it at the
    // size the game chose: without the record the sampling path leaves those lookups
    // alone and they resolve to a second, never written image over the same memory,
    // which is what kept the scene black even once the resolve itself was correct.
    if (mrt0_desc.info.size.width != mrt0_sharp_width) {
        upscaled_targets.insert(mrt0_desc.info.guest_address);
    }
    if (mrt1_desc.info.size.width != mrt1_sharp_width) {
        upscaled_targets.insert(mrt1_desc.info.guest_address);
    }
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    {
        static std::unordered_set<u64> logged_resolve;
        const u64 k = (u64(mrt0_desc.info.guest_address >> 12) << 24) ^
                      (u64(mrt1_desc.info.guest_address >> 12) << 4) ^
                      u64(mrt0_image.info.size.width);
        if (logged_resolve.insert(k).second) {
            LOG_INFO(Render_Vulkan,
                     "Resolve: mrt0={}x{} addr={:#x} samples={}, mrt1={}x{} addr={:#x} "
                     "samples={}, requested {}x{} -> {}x{}, recorded={}",
                     mrt0_image.info.size.width, mrt0_image.info.size.height,
                     mrt0_image.info.guest_address, mrt0_image.info.num_samples,
                     mrt1_image.info.size.width, mrt1_image.info.size.height,
                     mrt1_image.info.guest_address, mrt1_image.info.num_samples,
                     mrt0_desc.info.size.width, mrt0_desc.info.size.height,
                     mrt1_desc.info.size.width, mrt1_desc.info.size.height,
                     upscaled_targets.contains(mrt1_desc.info.guest_address));
        }
    }

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = {write_image.info.size.width, write_image.info.size.height, 1},
    };
    scheduler.CommandBuffer().copyImage(read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        write_image.GetImage(),
                                        vk::ImageLayout::eTransferDstOptimal, region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size);
    return true;
}

void Rasterizer::ProcessDownloadImages() {
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState(pipeline);
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
            // A pass that draws into an offscreen target we enlarged is already working at
            // the presentation scale, so it must not also take the window-to-surface
            // ratio. Both conditions can hold at once when such a target is bound
            // alongside the output surface, and applying both magnified the geometry to
            // twice the surface and left the scene black.
            const bool draws_into_enlarged_target = rt_fit_x > 1.001f;
            if (output_upscaled && !draws_into_enlarged_target) {
                // The patch does not convert the game's passes as a whole. Passes the
                // patch missed still carry 1080p viewport registers (their raw width is
                // half of the surface): fonts, gameplay notes and stars live there, and
                // they all need the doubling. Passes it did convert (raw width equals
                // the surface) separate into the ones spanning the full NDC - the 3D
                // scene, the composition and the procedural rhythm circles, which keep
                // the 4K viewport - and the 1080p UI sprite families, whose quads span
                // half the NDC and need the doubled viewport. See
                // OutputSpriteNeedsStretch for both rules.
                const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
                const float raw_vp_width = viewport.width;
                const bool stretch =
                    OutputSpriteNeedsStretch(vs_info.pgm_hash, raw_vp_width, vo_surface_width);
                if (stretch) {
                    viewport.x *= vo_fit_x;
                    viewport.y *= vo_fit_y;
                    viewport.width *= vo_fit_x;
                    viewport.height *= vo_fit_y;
                }
                // Report every distinct output-surface pass with the stretch decision
                // and the vertex-program hash that decided it, so the sprite list can be
                // traced back to the batches it touched and extended as new UI shaders
                // show up.
                static std::unordered_set<u64> logged_sprite;
                const u64 sprite_k =
                    (u64(liverpool->regs.color_buffers[0].Address() >> 8) << 32) ^
                    (vs_info.pgm_hash << 12) ^ (u64(std::bit_cast<u32>(viewport.width)) << 14) ^
                    (u64(std::bit_cast<u32>(viewport.height)) << 2) ^
                    (u64(std::bit_cast<u32>(raw_vp_width)) << 4) ^ (stretch ? 1u : 0u);
                if (logged_sprite.insert(sprite_k).second) {
                    LOG_INFO(Render_Vulkan,
                             "Output-surface sprite classification: cb0={:#x}, spriteVS={:#x}, "
                             "rawVPW={}, stretch={}, stretchedVP=({},{} {}x{}), voFit={}x{}",
                             liverpool->regs.color_buffers[0].Address(), vs_info.pgm_hash,
                             raw_vp_width, stretch, viewport.x, viewport.y, viewport.width,
                             viewport.height, vo_fit_x, vo_fit_y);
                }
            } else if (draws_into_enlarged_target) {
                // The offscreen target of this pass is rendered at the presentation
                // scale, so the viewport has to cover the enlarged target.
                //
                // The patch does not convert a pass as a whole. The scene passes show
                // vp=(1920,1080,1920,-1080) with both axes converted, and
                // vp=(768,1080,768,-1080) with only the vertical one: 540 became 1080
                // while 768 stayed as it was. Scaling such a pass on both axes pushes
                // the converted one off the target, and skipping it entirely leaves the
                // missed one at half the width it should cover, which is what squeezed
                // the character horizontally.
                //
                // The scissor the game writes is the reference. It always describes the
                // region in the coordinates of the original window, so an axis whose
                // viewport already spans that region times the ratio is converted, and
                // one that still matches the bare region is not. Decide per axis on
                // that basis.
                const float scsr_w = float(regs.screen_scissor.GetWidth());
                const float scsr_h = float(regs.screen_scissor.GetHeight());
                const bool converted_x =
                    scsr_w > 0.0f && viewport.width >= scsr_w * rt_fit_x * 0.999f;
                const bool converted_y =
                    scsr_h > 0.0f && std::abs(viewport.height) >= scsr_h * rt_fit_y * 0.999f;
                if (!converted_x) {
                    viewport.x *= rt_fit_x;
                    viewport.width *= rt_fit_x;
                }
                if (!converted_y) {
                    viewport.y *= rt_fit_y;
                    viewport.height *= rt_fit_y;
                }
                // Report the raw register state of every distinct enlarged-target
                // clip-enabled pass so the viewport values handed to Vulkan can be
                // traced back to what the game actually wrote. The per-axis rule
                // keys on screen_scissor and rt_fit, and a viewport that exceeds
                // its own target is the signature of an over-applied scale.
                static std::unordered_set<u64> logged_rtvp;
                const u64 rtvp_k = (u64(liverpool->regs.color_buffers[0].Address() >> 8) << 28) ^
                                   (u64(std::bit_cast<u32>(vp.xscale)) << 14) ^
                                   u64(std::bit_cast<u32>(vp.yscale));
                if (logged_rtvp.insert(rtvp_k).second) {
                    LOG_INFO(Render_Vulkan,
                             "RT viewport raw: cb0={:#x}, rawVP=({},{},{},{}), "
                             "scissor=({},{}) {}x{}, rtFit={}x{}, convertedXY={}{}",
                             liverpool->regs.color_buffers[0].Address(), vp.xoffset, vp.yoffset,
                             vp.xscale, vp.yscale, regs.screen_scissor.top_left_x,
                             regs.screen_scissor.top_left_y, scsr_w, scsr_h, rt_fit_x, rt_fit_y,
                             converted_x, converted_y);
                }
            }
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        if (vo_pass && vo_surface_width > 0 && vo_surface_height > 0) {
            // Passes drawing into the output surface must not be clipped by the window
            // rectangle the game keeps in its scissor registers: the geometry and the
            // render area already define the bounds.
            vp_scsr.top_left_x = 0;
            vp_scsr.top_left_y = 0;
            vp_scsr.bottom_right_x = s16(std::min<u32>(vo_surface_width, 16384u));
            vp_scsr.bottom_right_y = s16(std::min<u32>(vo_surface_height, 16384u));
        } else if (rt_fit_x > 1.001f) {
            // Grow the scissor with the enlarged offscreen target, otherwise the render
            // would stay confined to the area of the original one.
            vp_scsr.top_left_x = s16(vp_scsr.top_left_x * rt_fit_x);
            vp_scsr.top_left_y = s16(vp_scsr.top_left_y * rt_fit_y);
            vp_scsr.bottom_right_x = s16(std::min<u32>(
                u32(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x) * rt_fit_x), 16384u));
            vp_scsr.bottom_right_y = s16(std::min<u32>(
                u32(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y) * rt_fit_y), 16384u));
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });

        if (i == 0 && (output_upscaled || rt_fit_x > 1.001f || presents_upscaled)) {
            // Report what is actually handed to Vulkan for the passes we adjust. The
            // register-level diagnostics above cannot show whether a pass ended up
            // covering its target, because the correction is applied here and in the
            // push data, so a geometry that leaves the target can only be told apart
            // from one that fills it by the final values.
            // The primitive type and what the pass reads are part of the key. Two passes
            // targeting the same surface can end up with an identical viewport while
            // needing opposite treatment: the interface and the pass presenting the
            // scene both arrive with the same registers, and keying on the extent alone
            // reported only whichever came first and hid the other completely.
            static std::unordered_set<u64> logged_vp;
            const u64 k = (u64(liverpool->regs.color_buffers[0].Address() >> 8) << 32) ^
                          (u64(std::bit_cast<u32>(viewport.width)) << 12) ^
                          (u64(std::bit_cast<u32>(viewport.height)) << 6) ^
                          (u64(static_cast<u32>(regs.primitive_type)) << 3) ^
                          (regs.IsClipDisabled() ? 2u : 0u) ^ (presents_upscaled ? 1u : 0u);
            if (logged_vp.insert(k).second) {
                LOG_INFO(Render_Vulkan,
                         "Final viewport: cb0={:#x}, prim={}, clipDisabled={}, vp=({},{} "
                         "{}x{}), scissor=({},{} {}x{}), voFit={}x{}, rtFit={}x{}, "
                         "rtFitExtent={}x{}, outputUpscaled={}, presentsUpscaled={}",
                         liverpool->regs.color_buffers[0].Address(),
                         static_cast<u32>(regs.primitive_type), regs.IsClipDisabled(), viewport.x,
                         viewport.y, viewport.width, viewport.height, vp_scsr.top_left_x,
                         vp_scsr.top_left_y, vp_scsr.GetWidth(), vp_scsr.GetHeight(), vo_fit_x,
                         vo_fit_y, rt_fit_x, rt_fit_y, rt_fit_width, rt_fit_height, output_upscaled,
                         presents_upscaled);
            }
        }
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        dynamic_state.SetStencilReferences(front.stencil_test_val, back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
