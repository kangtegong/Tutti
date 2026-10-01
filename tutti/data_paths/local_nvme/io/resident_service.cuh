#pragma once

// tutti/data_paths/local_nvme/io/resident_service.cuh
//
// Resident I/O control service (gio_ring / Strata style).
//
// Motivation (measured on A30, driver 535): a per-wave submit_one_kernel
// launch is starved by concurrent compute kernels (~45 ms fixed latency per
// launch regardless of size), so GPU-issued NVMe I/O serializes with the
// forward pass.  A kernel with a SMALL number of blocks launched while the
// GPU is idle keeps its SMs for its whole lifetime (the hardware scheduler
// is non-preemptive) and keeps servicing requests in ~0.5 ms under full
// compute load, with ~6% compute interference (bench/persistent_probe.cu).
//
// The request ring lives in PINNED HOST memory: device-memory doorbells
// (tiny H2D copies) starve under compute because small copies are issued as
// kernels; the resident kernel polling host memory (with a nanosleep
// backoff) is the configuration that measured flat latency.
//
// One ring slot = one DeviceSubmitEntry.  Each resident thread statically
// owns the slots with index % total_threads == its global id, so no
// device-side atomics are needed on the ring.  The host posts to a slot by
// writing `entry` first, then bumping seq_req (x86 TSO makes the entry
// visible before the bump).  The kernel executes the IO with the same
// submit_read_one / submit_write_one primitives as submit_one_kernel and
// publishes {result, status_dw3, seq_done = seq_req}.

#define TUTTI_SUBMIT_ONE_NO_KERNEL 1  // we need DeviceSubmitEntry + primitives, not the kernel
#include "tutti/data_paths/local_nvme/io/submit_one.cuh"  // DeviceSubmitEntry

#include <cstdint>

namespace tutti::data_paths::local_nvme {

struct ResidentSlot {
    volatile std::uint32_t seq_req;     // host bumps to post
    volatile std::uint32_t seq_done;    // kernel sets = seq_req when terminal
    std::uint32_t _pad0, _pad1;
    // Per-entry completion status is written to THIS GPU address (the op's
    // arena d_status slot), NOT into the ring: the slot is reusable the
    // moment seq_done catches up, and the op's results live in memory the
    // op already owns (freed with its arena slot) -- no harvest ordering.
    EntryCompletionStatus* status_out;  // GPU pointer
    DeviceSubmitEntry entry;            // host-written BEFORE the seq_req bump
};

struct ResidentRing {
    volatile int   stop;
    volatile std::uint32_t post_count;  // host bumps AFTER each slot post: the kernel's
                                        // only idle-time PCIe read (1 thread per block)
    // A kernel that never terminates deadlocks every device-wide sync on the
    // host (torch.cuda.synchronize, pageable-memcpy internals, ...). The
    // service therefore EXITS after idle_exit_ns without work; the host
    // relaunches it on post when alive == 0.
    volatile int   alive;               // active block count (atomicAdd_system)
    std::uint32_t  idle_exit_ns;        // self-exit after this much idle time
    std::uint32_t  num_slots;
    std::uint32_t  cq_poll_budget;
    std::uint32_t  backoff_ns;          // idle-sweep nanosleep
    // slots follow immediately (allocated as one pinned block)
    ResidentSlot   slots[1];
};

inline std::size_t resident_ring_bytes(std::uint32_t num_slots) {
    return sizeof(ResidentRing) + (num_slots - 1) * sizeof(ResidentSlot);
}

// Launches the resident service kernel (blocks x threads) on `stream`.
// Returns cudaGetLastError() as int (0 = success).  The kernel runs until
// ring->stop becomes non-zero.  d_ring is the DEVICE alias of the pinned
// ring (cudaHostGetDevicePointer).
int launch_resident_service(ResidentRing* d_ring,
                            std::uint32_t blocks,
                            std::uint32_t threads,
                            void*         stream);

} // namespace tutti::data_paths::local_nvme
