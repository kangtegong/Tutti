// tutti/data_paths/local_nvme/metadata/prp_buf_pool.cpp
//
// R19 S3b REQUIRED 1: host-pinned PRP-list buffer pool implementation.

#include <cstdio>
#include <cstring>
#include "tutti/data_paths/local_nvme/metadata/prp_buf_pool.h"

#include <nvm_dma.h>   // nvm_dma_map_data_host, nvm_dma_unmap
#include <cuda_runtime.h>  // cudaHostAlloc: snvme's host mapping requires pinned pages

namespace tutti::data_paths::local_nvme {

PrpBufPool::~PrpBufPool() {
    for (auto& seg : segments_) {
        if (seg.dma) nvm_dma_unmap(seg.dma);
        if (seg.vaddr) cudaFreeHost(seg.vaddr);
    }
}

void PrpBufPool::init(nvm_ctrl_t* ctrl, std::uint64_t page_size) {
    ctrl_ = ctrl;
    page_size_ = page_size;
}

PrpBufRef PrpBufPool::alloc_pages(std::uint64_t n_pages) {
    if (n_pages == 0 || !ctrl_) return {};

    // Try current segment first.
    if (!segments_.empty()) {
        Segment& cur = segments_.back();
        if (cur.used_pages + n_pages <= cur.capacity_pages) {
            PrpBufRef ref;
            ref.segment = cur.dma;
            ref.base_page = cur.used_pages;
            ref.num_pages = n_pages;
            ref.valid = true;
            cur.used_pages += n_pages;
            return ref;
        }
    }

    // Need a new segment. Size = max(segment_pages_, n_pages rounded up).
    // Segment sizing: snvme/libnvm host DMA maps fail above a few MB
    // (observed: 256 MB map -> rc=14). Cap segments at 4 MB (1024 pages);
    // need-driven growth still allocates exactly what a request requires.
    std::uint64_t seg_cap = segment_pages_;
    if (seg_cap > 1024) seg_cap = 1024;
    std::uint64_t seg_pages = seg_cap;
    if (n_pages > seg_pages) {
        seg_pages = ((n_pages + seg_cap - 1) / seg_cap) * seg_cap;
    }

    const std::uint64_t seg_bytes = seg_pages * page_size_;
    // snvme's host DMA mapping pins the caller's pages; an internal malloc'd
    // buffer (vaddr = nullptr path) fails with EFAULT. Allocate pinned memory
    // the same way PrpPageCache does and hand its vaddr to the mapping.
    void* vaddr = nullptr;
    cudaError_t ce = cudaHostAlloc(&vaddr, static_cast<size_t>(seg_bytes),
                                   cudaHostAllocDefault);
    if (ce != cudaSuccess || vaddr == nullptr) {
        std::fprintf(stderr, "[tutti] PrpBufPool: cudaHostAlloc(%zu) failed ce=%d\n",
                     (size_t)seg_bytes, (int)ce);
        return {};
    }
    std::memset(vaddr, 0, static_cast<size_t>(seg_bytes));
    nvm_dma_t* dma = nullptr;
    int rc = nvm_dma_map_data_host(&dma, ctrl_, vaddr,
                                   static_cast<size_t>(seg_bytes));
    if (rc != 0 || !dma) {
        std::fprintf(stderr, "[tutti] PrpBufPool: nvm_dma_map_data_host(%zu bytes) rc=%d dma=%p\n",
                     (size_t)seg_bytes, rc, (void*)dma);
        cudaFreeHost(vaddr);
        return {};
    }

    Segment seg;
    seg.dma = dma;
    seg.vaddr = vaddr;
    seg.capacity_pages = seg_pages;
    seg.used_pages = n_pages;
    segments_.push_back(seg);
    total_pages_ += seg_pages;

    PrpBufRef ref;
    ref.segment = dma;
    ref.base_page = 0;
    ref.num_pages = n_pages;
    ref.valid = true;
    return ref;
}

} // namespace tutti::data_paths::local_nvme
