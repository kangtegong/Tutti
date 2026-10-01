#pragma once

// tutti/data_paths/local_nvme/metadata/prp_buf_pool.h
//
// Host-pinned PRP-list buffer pool — replaces per-registration
// nvm_dma_map_data_host for pre-built PRP-list pages.
//
// R19 S3b REQUIRED 1: at 180GB KV / 128KiB tensor = 1.47M registrations,
// per-registration nvm_dma_map_data_host causes minute-level latency
// (each call is a kernel ioctl). This pool pre-maps a large host-pinned
// DMA buffer via one nvm_dma_map_data_host per segment, then sub-allocates
// PRP pages from it via a bump allocator.
//
// NOTE: This is NOT the legacy "prp_list_pool" that was cleaned up in
// R16 S6b. That was a fixed-IOVA scatter-patch mechanism (deleted).
// This pool is a host-pinned PRP buffer allocator — different purpose,
// different implementation. The name "prp_buf_pool" avoids confusion.
//
// Lifecycle:
//   - alloc_pages(n) returns a PrpBufRef (segment ptr + base page index);
//     grows if needed (new nvm_dma_map_data_host per segment).
//   - free is a no-op (segments are freed on pool shutdown).
//   - shutdown nvm_dma_unmap's all segments.
//
// Thread safety: none — caller must hold registry_mutex_ (same as
// DescPool, storage_runtime.h:374/401).

#include <cstdint>
#include <vector>

// nvm_dma.h pulls in nvm_types.h which defines nvm_dma_t and nvm_ctrl_t
// as anonymous-struct typedefs (no tag name). Include the full header so
// all types are available; cannot forward-declare anonymous typedefs.
#include <nvm_dma.h>

namespace tutti::data_paths::local_nvme {

// Opaque reference to a sub-allocated range within the pool.
struct PrpBufRef {
    nvm_dma_t* segment = nullptr;     // the nvm_dma segment (for ioaddrs/vaddr)
    std::uint64_t base_page = 0;     // page index within this segment
    std::uint64_t num_pages = 0;     // pages allocated
    bool valid = false;
};

class PrpBufPool {
public:
    // Default segment size: 256 MiB = 65536 pages (4KiB) = ~1.05M slices
    // (at 16 slices/page). 180GB KV / 128KiB = 1.47M slices → ~2 segments.
    static constexpr std::uint64_t DEFAULT_SEGMENT_PAGES = 65536ULL;

    PrpBufPool() = default;
    ~PrpBufPool();

    PrpBufPool(const PrpBufPool&) = delete;
    PrpBufPool& operator=(const PrpBufPool&) = delete;

    // Initialize with the controller handle (needed for nvm_dma_map_data_host).
    void init(nvm_ctrl_t* ctrl, std::uint64_t page_size);

    // Allocate n_pages from the pool. Returns a PrpBufRef; .valid=false on
    // failure. The caller uses segment->ioaddrs[base_page + i] and
    // segment->vaddr + (base_page + i) * page_size for IOVA/virtual access.
    PrpBufRef alloc_pages(std::uint64_t n_pages);

    // Total pages allocated across all segments.
    std::uint64_t total_pages() const { return total_pages_; }

    // Number of segments allocated (for diagnostics / dma_map count).
    std::size_t num_segments() const { return segments_.size(); }

private:
    struct Segment {
        nvm_dma_t* dma = nullptr;      // nvm_dma_map_data_host'd
        void* vaddr = nullptr;         // cudaHostAlloc'd backing (snvme requires pinned pages)
        std::uint64_t capacity_pages = 0;  // total pages in this segment
        std::uint64_t used_pages = 0;     // pages already sub-allocated
    };

    std::vector<Segment> segments_;
    nvm_ctrl_t* ctrl_ = nullptr;
    std::uint64_t page_size_ = 4096;
    std::uint64_t total_pages_ = 0;
    std::uint64_t segment_pages_ = DEFAULT_SEGMENT_PAGES;
};

} // namespace tutti::data_paths::local_nvme
