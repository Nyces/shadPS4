// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/buffer.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/tile.h"

namespace VideoCore {

using namespace Vulkan;
using Libraries::VideoOut::TilingMode;
using VideoOutFormat = Libraries::VideoOut::PixelFormat;

// Internal resolution scale of the emulator. This is a build time constant on purpose: it must be
// known before the very first image is created so that every surface in a framebuffer ends up with
// the same size. Change this constant to raise or lower the internal rendering resolution.
constexpr u32 kInternalResolutionScale = 2;

u32 GetResolutionScale() {
    // The 4K rendering is opt in through an empty patch that carries no guest changes. Without it
    // the emulator renders exactly at the guest resolution, which keeps the scaling risk away from
    // every other game.
    const bool enabled =
        kInternalResolutionScale > 1 && MemoryPatcher::Is4KResolutionPatchEnabled();
    static bool logged = false;
    if (!logged) {
        logged = true;
        if (enabled) {
            LOG_INFO(Render_Vulkan, "4K resolution patch enabled, host surfaces scaled {}x",
                     kInternalResolutionScale);
        } else {
            LOG_INFO(Render_Vulkan, "No 4K resolution patch, rendering at the guest resolution");
        }
    }
    return enabled ? kInternalResolutionScale : 1;
}

namespace {
struct ScaledSurface {
    VAddr base;
    u32 size;
    u32 width;
    u32 height;
    u32 pitch;
};
std::mutex g_scaled_mutex;
std::vector<ScaledSurface> g_scaled_ranges;

bool FindScaledSurface(VAddr base, ScaledSurface& out) {
    std::scoped_lock lock{g_scaled_mutex};
    for (const auto& entry : g_scaled_ranges) {
        if (entry.base == base) {
            out = entry;
            return true;
        }
    }
    return false;
}

// Forget a scaled surface once its host image is destroyed. The address can then be reused by an
// ordinary texture, and keeping the entry would make that texture adopt the enlarged layout of a
// surface it has nothing to do with.
static void ForgetScaledRangeUnlocked(VAddr base) {
    for (auto it = g_scaled_ranges.begin(); it != g_scaled_ranges.end(); ++it) {
        if (it->base == base) {
            g_scaled_ranges.erase(it);
            return;
        }
    }
}

constexpr size_t MaxScaledRanges = 16384;

void RecordScaledRange(VAddr address, u32 size, u32 width, u32 height, u32 pitch) {
    std::scoped_lock lock{g_scaled_mutex};
    // Surfaces are recreated constantly, so the same base address gets recorded over and over.
    // Appending those duplicates silently exhausted the table, and every surface created after the
    // table filled up was no longer recognized as scaled: its passes fell back to the guest
    // viewport and drew the 1080p layout into a 4K target, which is what turned the picture into an
    // upscaled 1080p frame. Keep one entry per base address instead.
    for (auto& entry : g_scaled_ranges) {
        if (entry.base == address) {
            entry = {address, size, width, height, pitch};
            return;
        }
    }
    if (g_scaled_ranges.size() < MaxScaledRanges) {
        g_scaled_ranges.push_back({address, size, width, height, pitch});
    } else {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOG_ERROR(Render_Vulkan, "Scaled surface table is full, {} entries", MaxScaledRanges);
        }
    }
}
} // namespace

void ForgetScaledRange(VAddr base) {
    std::scoped_lock lock{g_scaled_mutex};
    ForgetScaledRangeUnlocked(base);
}

bool IsScaledRange(VAddr address) {
    std::scoped_lock lock{g_scaled_mutex};
    for (const auto& entry : g_scaled_ranges) {
        if (address >= entry.base && address < entry.base + entry.size) {
            return true;
        }
    }
    return false;
}

// Grow only surfaces that shape the visible picture. Only the host extent of the surface grows;
// the guest side of the image keeps describing the small layout so that the memory tracker, the
// page watchers, the upload/download bounds and the buffer aliasing stay consistent with what the
// guest actually owns. Smaller textures and already large surfaces are left untouched.
static void ApplyResolutionScale(ImageInfo& info) {
    const u32 scale = GetResolutionScale();
    if (scale == 1) {
        return;
    }
    const u32 width = info.size.width;
    const u32 height = info.size.height;
    // The surfaces the game flips are 1920x1080 (or 1920x1080 with a taller tile alignment).
    const bool screen_sized = width == 1920 && height >= 1080 && height <= 1152;
    // Square offscreen plates hold the character and the other large renders. They are drawn with a
    // normal viewport into their own surface and then sampled by the composite, so enlarging them
    // adds real detail without moving anything on screen. Without this the plate is produced at
    // 840x840 or 1680x1680 and the 4K output is only an upscale, which is why the 3D content looks
    // exactly as aliased as it does at 1080p. Their mip levels (210x210, 52x52, ...) stay below the
    // bound and are left at the guest size.
    const bool square_plate = width == height && width >= 768 && width <= 1920;
    if (!screen_sized && !square_plate) {
        return;
    }
    const u32 guest_size = info.guest_size;
    info.size.width *= scale;
    info.size.height *= scale;
    info.pitch *= scale;
    // Rebuild the whole layout for the larger extent. Patching the fields by hand left the mip
    // heights, mip pitches and mip offsets describing the old 1080p layout, so a tiled surface with
    // mip levels (the UI blur pyramid for instance) addressed the wrong memory and each level
    // overwrote the next one.
    info.UpdateSize();
    // Only the host extent grows. Everything that indexes guest memory has to keep seeing the
    // guest layout: the page table, the overlap resolution, the uploads and downloads and the
    // buffer aliasing all use these fields, and letting a scaled surface claim four times the
    // memory it really owns made it overlap unrelated textures. Those were then invalidated and
    // re-uploaded from the wrong place, which showed up as black, missing or garbled sprites. The
    // mip table keeps describing the host image, which the host side copies rely on.
    info.guest_size = guest_size;
    info.props.is_scaled = 1;
    RecordScaledRange(info.guest_address, guest_size, info.size.width, info.size.height,
                      info.pitch);
}

static vk::Format ConvertPixelFormat(const VideoOutFormat format) {
    switch (format) {
    case VideoOutFormat::A8B8G8R8Srgb:
    // Remaining formats are mapped to RGBA for internal consistency and changed to BGRA in the
    // frame image view.
    case VideoOutFormat::A8R8G8B8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case VideoOutFormat::A2R10G10B10:
    case VideoOutFormat::A2R10G10B10Srgb:
    case VideoOutFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2B10G10R10UnormPack32;
    default:
        break;
    }
    UNREACHABLE_MSG("Unknown format={}", static_cast<u32>(format));
    return {};
}

ImageInfo::ImageInfo(const Libraries::VideoOut::BufferAttributeGroup& group,
                     VAddr cpu_address) noexcept {
    const auto& attrib = group.attrib;
    props.is_tiled = attrib.tiling_mode == TilingMode::Tile;
    tile_mode =
        props.is_tiled ? AmdGpu::TileMode::Display2DThin : AmdGpu::TileMode::DisplayLinearAligned;
    array_mode = AmdGpu::GetArrayMode(tile_mode);
    pixel_format = ConvertPixelFormat(attrib.pixel_format);
    type = AmdGpu::ImageType::Color2D;
    size.width = attrib.width;
    size.height = attrib.height;
    pitch = attrib.tiling_mode == TilingMode::Linear ? size.width : (size.width + 127) & (~127);
    num_bits = attrib.pixel_format != VideoOutFormat::A16R16G16B16Float ? 32 : 64;
    ASSERT(num_bits == 32);

    guest_address = cpu_address;
    UpdateSize();
    ApplyResolutionScale(*this);
}

ImageInfo::ImageInfo(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent hint) noexcept {
    props.is_tiled = buffer.IsTiled();
    tile_mode = buffer.GetTileMode();
    array_mode = AmdGpu::GetArrayMode(tile_mode);
    pixel_format = LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
    num_samples = buffer.NumSamples();
    num_bits = NumBitsPerBlock(buffer.GetDataFmt());
    type = AmdGpu::ImageType::Color2D;
    size.width = hint.Valid() ? hint.width : buffer.Pitch();
    size.height = hint.Valid() ? hint.height : buffer.Height();
    size.depth = 1;
    pitch = buffer.Pitch();
    resources.layers = buffer.NumSlices();
    meta_info.cmask_addr = buffer.info.fast_clear ? buffer.CmaskAddress() : 0;
    meta_info.fmask_addr = buffer.info.compression ? buffer.FmaskAddress() : 0;

    guest_address = buffer.Address();
    if (props.is_tiled) {
        guest_size = buffer.GetColorSliceSize() * resources.layers;
        mips_layout[0] = MipInfo(guest_size, pitch, buffer.Height(), 0);
    } else {
        std::tie(std::ignore, std::ignore, guest_size) =
            ImageSizeLinearAligned(pitch, size.height, num_bits, num_samples);
        guest_size *= resources.layers;
        mips_layout[0] = MipInfo(guest_size, pitch, size.height, 0);
    }
    alt_tile = Libraries::Kernel::sceKernelIsNeoMode() && buffer.info.alt_tile_mode;
    ApplyResolutionScale(*this);
}

ImageInfo::ImageInfo(const AmdGpu::DepthBuffer& buffer, u32 num_slices, VAddr htile_address,
                     AmdGpu::CbDbExtent hint, bool write_buffer) noexcept {
    tile_mode = buffer.GetTileMode();
    array_mode = AmdGpu::GetArrayMode(tile_mode);
    pixel_format = LiverpoolToVK::DepthFormat(buffer.z_info.format, buffer.stencil_info.format);
    type = AmdGpu::ImageType::Color2D;
    props.is_tiled = buffer.IsTiled();
    props.is_depth = true;
    props.has_stencil = buffer.stencil_info.format != AmdGpu::DepthBuffer::StencilFormat::Invalid;
    num_samples = buffer.NumSamples();
    num_bits = buffer.NumBits();
    size.width = hint.Valid() ? hint.width : buffer.Pitch();
    size.height = hint.Valid() ? hint.height : buffer.Height();
    size.depth = 1;
    pitch = buffer.Pitch();
    resources.layers = num_slices;
    meta_info.htile_addr = buffer.z_info.tile_surface_enable ? htile_address : 0;

    stencil_addr = write_buffer ? buffer.StencilWriteAddress() : buffer.StencilAddress();
    stencil_size = pitch * size.height * sizeof(u8);

    guest_address = write_buffer ? buffer.DepthWriteAddress() : buffer.DepthAddress();
    if (props.is_tiled) {
        guest_size = buffer.GetDepthSliceSize() * resources.layers;
        mips_layout[0] = MipInfo(guest_size, pitch, buffer.Height(), 0);
    } else {
        std::tie(std::ignore, std::ignore, guest_size) =
            ImageSizeLinearAligned(pitch, size.height, num_bits, num_samples);
        guest_size *= resources.layers;
        mips_layout[0] = MipInfo(guest_size, pitch, size.height, 0);
    }
    ApplyResolutionScale(*this);
}

ImageInfo::ImageInfo(const AmdGpu::Image& image, const Shader::ImageResource& desc) noexcept {
    tile_mode = image.GetTileMode();
    array_mode = AmdGpu::GetArrayMode(tile_mode);
    pixel_format = LiverpoolToVK::SurfaceFormat(image.GetDataFmt(), image.GetNumberFmt());
    if (desc.is_depth) {
        pixel_format = LiverpoolToVK::PromoteFormatToDepth(pixel_format);
        props.is_depth = true;
    }
    type = image.GetBaseType();
    props.is_tiled = image.IsTiled();
    props.is_volume = type == AmdGpu::ImageType::Color3D;
    props.is_pow2 = image.pow2pad;
    props.is_block = AmdGpu::IsBlockCoded(image.GetDataFmt());
    size.width = image.width + 1;
    size.height = image.height + 1;
    size.depth = props.is_volume ? image.depth + 1 : 1;
    pitch = image.Pitch();
    resources.levels = image.NumLevels();
    resources.layers = image.NumLayers();
    num_samples = image.NumSamples();
    num_bits = NumBitsPerBlock(image.GetDataFmt());
    bank_swizzle = image.GetBankSwizzle();

    guest_address = image.Address();

    alt_tile = Libraries::Kernel::sceKernelIsNeoMode() && image.alt_tile_mode;
    UpdateSize();
    // If this address already backs a scaled render target, adopt exactly the same extent, pitch
    // and layout the render path recorded. Scaling only the extent keeps the two values
    // disagreeing, and the texture cache then keeps a second, never written image over the same
    // memory, which is what leaves sampled layers such as the blurred background black.
    // Restrict this to the recorded surface itself: its descriptor starts exactly at the recorded
    // base and is the guest sized version of it. A texture that merely starts inside a scaled range
    // has nothing to do with the enlarged surface (the game packs many small character textures
    // into the same block), and adopting the larger extent made it sample the wrong memory, which
    // showed up as black, missing or garbled sprites.
    ScaledSurface scaled{};
    const u32 scale = GetResolutionScale();
    if (FindScaledSurface(image.Address(), scaled) && scaled.width == size.width * scale &&
        scaled.height == size.height * scale) {
        const u32 guest_footprint = guest_size;
        size.width = scaled.width;
        size.height = scaled.height;
        pitch = scaled.pitch;
        // Rebuild the host layout so the sampled image matches the render target region for
        // region, but keep the guest footprint of the descriptor: the sampled surface still owns
        // only its guest sized range of memory.
        UpdateSize();
        guest_size = guest_footprint;
        props.is_scaled = 1;
    }
}

bool ImageInfo::IsCompatible(const ImageInfo& info) const {
    return (IsVulkanFormatCompatible(pixel_format, info.pixel_format) ||
            IsVulkanFormatCompatible(info.pixel_format, pixel_format)) &&
           num_samples == info.num_samples && num_bits == info.num_bits;
}

void ImageInfo::UpdateSize() {
    if (array_mode == AmdGpu::ArrayMode::ArrayLinearGeneral) {
        UNREACHABLE_MSG("Unhandled array mode: ArrayLinearGeneral");
    }
    const u32 thickness = AmdGpu::GetMicroTileThickness(array_mode);
    const bool macro = AmdGpu::IsMacroTiled(array_mode);
    guest_size = 0;
    micro_tiled_mips = 0;
    for (s32 mip = 0; mip < resources.levels; ++mip) {
        u32 mip_w = pitch >> mip;
        u32 mip_h = size.height >> mip;
        if (props.is_block) {
            mip_w = (mip_w + 3) / 4;
            mip_h = (mip_h + 3) / 4;
        }
        mip_w = std::max(mip_w, 1u);
        mip_h = std::max(mip_h, 1u);
        u32 mip_d = std::max(size.depth >> mip, 1u);

        if (props.is_pow2) {
            mip_w = std::bit_ceil(mip_w);
            mip_h = std::bit_ceil(mip_h);
            mip_d = std::bit_ceil(mip_d);
        }

        auto& mip_info = mips_layout[mip];
        u32 mip_thickness = 1;
        if (array_mode == AmdGpu::ArrayMode::ArrayLinearAligned) {
            std::tie(mip_info.pitch, mip_info.height, mip_info.size) =
                ImageSizeLinearAligned(mip_w, mip_h, num_bits, num_samples);
        } else if (macro &&
                   IsMacroTiledMip(mip_w, mip_h, num_bits, num_samples, tile_mode, mip, alt_tile)) {
            mip_thickness = thickness;
            std::tie(mip_info.pitch, mip_info.height, mip_info.size) =
                ImageSizeMacroTiled(mip_w, mip_h, num_bits, num_samples, tile_mode, alt_tile);
        } else {
            mip_thickness = std::min(thickness, 4u);
            std::tie(mip_info.pitch, mip_info.height, mip_info.size) =
                ImageSizeMicroTiled(mip_w, mip_h, mip_thickness, num_bits, num_samples);
            if (macro) {
                micro_tiled_mips |= 1u << mip;
            }
        }
        if (props.is_block) {
            mip_info.pitch = std::max(mip_info.pitch * 4, 32u);
            mip_info.height = std::max(mip_info.height * 4, 32u);
        }
        u32 num_slices = mip_d * resources.layers;
        num_slices += (-num_slices) & (mip_thickness - 1);
        mip_info.size *= num_slices;
        mip_info.offset = guest_size;
        guest_size += mip_info.size;
    }
}

s32 ImageInfo::MipOf(const ImageInfo& info) const {
    if (!IsCompatible(info)) {
        return -1;
    }

    if (info.array_mode != array_mode) {
        return -1;
    }

    // Currently we expect only one level to be copied.
    if (resources.levels != 1) {
        return -1;
    }

    const auto info_dim = info.props.is_block ? 2 : 0;
    const auto this_dim = props.is_block ? 2 : 0;

    // Find mip
    auto mip = -1;
    for (auto m = 0; m < info.resources.levels; ++m) {
        const auto& [mip_size, mip_pitch, mip_height, mip_ofs] = info.mips_layout[m];
        const VAddr mip_base = info.guest_address + mip_ofs;
        const VAddr mip_end = mip_base + mip_size;
        const u32 slice_size = mip_size / info.resources.layers;
        if (guest_address >= mip_base && guest_address < mip_end &&
            (guest_address - mip_base) % slice_size == 0 &&
            (pitch >> this_dim) == (mip_pitch >> info_dim)) {
            mip = m;
            break;
        }
    }

    if (mip < 0) {
        return -1;
    }

    // 2D block dimensions of both images should be the same.
    const auto mip_w = std::max(info.size.width >> (mip + info_dim), 1u);
    const auto mip_h = std::max(info.size.height >> (mip + info_dim), 1u);
    const auto this_w = std::max(size.width >> this_dim, 1u);
    const auto this_h = std::max(size.height >> this_dim, 1u);
    if ((this_w != mip_w) || (this_h != mip_h)) {
        return -1;
    }

    const auto mip_d = std::max(info.size.depth >> mip, 1u);
    if (info.type == AmdGpu::ImageType::Color3D && type == AmdGpu::ImageType::Color2D) {
        // In case of 2D array to 3D copy, make sure we have proper number of layers.
        if (resources.layers != mip_d) {
            return -1;
        }
    } else {
        if (type != info.type) {
            return -1;
        }
    }

    return mip;
}

s32 ImageInfo::SliceOf(const ImageInfo& info, s32 mip) const {
    if (!IsCompatible(info)) {
        return -1;
    }

    // Array slices should be of the same type.
    if (type != info.type) {
        return -1;
    }

    // 2D block dimensions of both images should be the same.
    const auto info_dim = info.props.is_block ? 2 : 0;
    const auto mip_w = std::max(info.size.width >> (mip + info_dim), 1u);
    const auto mip_h = std::max(info.size.height >> (mip + info_dim), 1u);
    const auto mip_p = std::max(info.mips_layout[mip].pitch >> info_dim, 1u);

    const auto this_dim = props.is_block ? 2 : 0;
    const auto this_w = std::max(size.width >> this_dim, 1u);
    const auto this_h = std::max(size.height >> this_dim, 1u);
    const auto this_p = std::max(pitch >> this_dim, 1u);
    if ((this_w != mip_w) || (this_h != mip_h) || (this_p != mip_p)) {
        return -1;
    }

    // Check for size alignment.
    const u32 slice_size = info.mips_layout[mip].size / info.resources.layers;
    if (guest_size % slice_size != 0) {
        return -1;
    }

    // Ensure that address is aligned too.
    const auto addr_diff = guest_address - (info.guest_address + info.mips_layout[mip].offset);
    if ((addr_diff % guest_size) != 0) {
        return -1;
    }

    return addr_diff / guest_size;
}

} // namespace VideoCore
