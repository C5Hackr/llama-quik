#pragma once

#include "ggml.h"

#ifdef __cplusplus
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

// Experimental MoE per-expert CUDA slot-cache metadata.
//
// This is intentionally tiny and attached to a synthetic GPU expert-bank tensor
// via tensor->extra, guarded by tensor->op_params[0] == GGML_MOE_EXPERT_SLOT_CACHE_MAGIC.
// The tensor itself has the normal ggml_mul_mat_id expert-bank shape, except
// ne[2] is the number of GPU slots rather than the model's total expert count.
// The CUDA mul_mat_id backend then treats IDs as real model expert IDs, promotes
// missing experts from the CPU tensor into slots, remaps IDs to slot IDs, and
// dispatches the regular CUDA mul_mat_id implementation on the slot bank.
static constexpr int32_t GGML_MOE_EXPERT_SLOT_CACHE_MAGIC = 0x4d4f4543; // 'MOEC'

struct ggml_moe_expert_slot_cache {
    uint32_t magic = GGML_MOE_EXPERT_SLOT_CACHE_MAGIC;

    const ggml_tensor * cpu_tensor = nullptr; // original full CPU expert tensor
    int32_t n_experts = 0;                    // original expert count
    int32_t n_slots   = 0;                    // GPU cache slots in this tensor

    size_t cpu_expert_stride_bytes = 0;
    size_t gpu_slot_stride_bytes   = 0;
    size_t copy_bytes              = 0;

    uint64_t clock = 0;
    std::mutex mutex;
    std::unordered_map<int32_t, int32_t> expert_to_slot;
    std::vector<int32_t> slot_to_expert;
    std::vector<uint64_t> slot_last_access;
};
#endif
