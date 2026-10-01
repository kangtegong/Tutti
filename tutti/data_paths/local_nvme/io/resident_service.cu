// tutti/data_paths/local_nvme/io/resident_service.cu
//
// Resident I/O control kernel + host launcher.  See resident_service.cuh.

#include "tutti/data_paths/local_nvme/io/resident_service.cuh"

#include <tutti/cuda_like.h>

namespace tutti::data_paths::local_nvme {

TUTTI_GLOBAL
void resident_service_kernel(ResidentRing* ring)
{
    const std::uint32_t tid    = TUTTI_THREAD_IDX_X
                               + TUTTI_BLOCK_IDX_X * TUTTI_BLOCK_DIM_X;
    const std::uint32_t stride = TUTTI_BLOCK_DIM_X * gridDim.x;
    const std::uint32_t budget = ring->cq_poll_budget;
    const std::uint32_t backoff = ring->backoff_ns;

    // Idle polling must not hammer PCIe: 256 threads sweeping pinned slot
    // headers saturates the bus and starves every other host<->GPU transfer
    // (measured: 4 KiB sync H2D stalling for minutes, stores at 0.3 GB/s).
    // Only thread 0 of each block reads the single post counter; the block
    // sweeps the slots only when it changed.
    const std::uint64_t idle_exit = ring->idle_exit_ns;
    __shared__ std::uint32_t seen, cur, stopf;
    __shared__ unsigned long long idle_ns;
    if (TUTTI_THREAD_IDX_X == 0) {
        seen = 0; cur = 0; stopf = 0; idle_ns = 0;
        atomicAdd_system(const_cast<int*>(&ring->alive), 1);
        __threadfence_system();
    }
    __syncthreads();

    while (true) {
        if (TUTTI_THREAD_IDX_X == 0) {   // single reader: uniform view for the block
            cur = ring->post_count;
            stopf = (std::uint32_t)ring->stop;
            if (idle_exit && idle_ns > idle_exit) stopf = 2;   // idle self-exit
        }
        __syncthreads();
        if (stopf) {                     // uniform exit (shared flag)
            if (TUTTI_THREAD_IDX_X == 0)
                atomicAdd_system(const_cast<int*>(&ring->alive), -1);
            return;
        }
        if (cur == seen) {
            if (backoff) __nanosleep(backoff);
            if (TUTTI_THREAD_IDX_X == 0) idle_ns += backoff ? backoff : 1000;
            __syncthreads();
            continue;
        }
        if (TUTTI_THREAD_IDX_X == 0) { seen = cur; idle_ns = 0; }
        __syncthreads();
        bool worked = false;
        for (std::uint32_t i = tid; i < ring->num_slots; i += stride) {
            ResidentSlot* s = &ring->slots[i];
            const std::uint32_t req = s->seq_req;
            if (req == s->seq_done) continue;

            // Entry is fully written before seq_req was bumped (host TSO).
            const DeviceSubmitEntry e =
                *const_cast<const DeviceSubmitEntry*>(&s->entry);
            const AddressDescriptor* d = e.prp_entry;

            EntryCompletionStatus* out = s->status_out;
            out->result = 0;                 // submit helpers only write on error paths
            out->nvme_status_dword3 = 0;
            if (e.direction == 0) {
                submit_read_one(e.target, d->prp1, d->prp2,
                                e.target_offset, d->data_length,
                                out, budget, /*inject_flag=*/0);
            } else {
                submit_write_one(e.target, d->prp1, d->prp2,
                                 e.target_offset, d->data_length,
                                 out, budget, /*inject_flag=*/0);
            }
            __threadfence_system();      // publish status before seq_done
            s->seq_done = req;
            worked = true;
        }
        (void)worked;
        __syncthreads();                 // whole block rejoins before re-checking the counter
    }
}

int launch_resident_service(ResidentRing* d_ring,
                            std::uint32_t blocks,
                            std::uint32_t threads,
                            void*         stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    resident_service_kernel<<<blocks, threads, 0, s>>>(d_ring);
    return static_cast<int>(cudaGetLastError());
}

} // namespace tutti::data_paths::local_nvme
