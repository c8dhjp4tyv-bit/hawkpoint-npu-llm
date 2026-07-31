#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace {

constexpr int64_t kRouterInput = 2048;
constexpr int64_t kRouterOutput = 128;
constexpr int64_t kExpertBatch = 8;
constexpr int64_t kExpertDimension = 2048;

std::vector<uint32_t> load_instructions(const std::string & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open instruction file: " + path);
    }
    const auto bytes = stream.tellg();
    if (bytes < 0 || bytes % sizeof(uint32_t) != 0) {
        throw std::runtime_error("invalid instruction file: " + path);
    }
    stream.seekg(0);
    std::vector<uint32_t> result(static_cast<size_t>(bytes) / sizeof(uint32_t));
    stream.read(reinterpret_cast<char *>(result.data()), bytes);
    return result;
}

uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + rounding) >> 16U);
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct ggml_backend_xdna_context {
    xrt::device device;
    xrt::xclbin xclbin;
    xrt::hw_context hardware_context;
    xrt::kernel kernel;
    std::vector<uint32_t> instructions;
    xrt::bo instruction_bo;
    xrt::bo weights_bo;
    xrt::bo scales_bo;
    xrt::bo input_bo;
    xrt::bo output_bo;
    std::mutex mutex;
    uint64_t operations = 0;
    uint64_t router_operations = 0;
    uint64_t expert_operations = 0;
    uint64_t dense_operations = 0;

    ggml_backend_xdna_context(const std::string & xclbin_path, const std::string & inst_path)
        : device(0),
          xclbin(std::string(xclbin_path)),
          hardware_context(),
          kernel(),
          instructions(load_instructions(inst_path)),
          instruction_bo(),
          weights_bo(),
          scales_bo(),
          input_bo(),
          output_bo() {
        const auto kernels = xclbin.get_kernels();
        const auto found = std::find_if(
            kernels.begin(), kernels.end(), [](const xrt::xclbin::kernel & candidate) {
                return candidate.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) {
            throw std::runtime_error("MLIR_AIE kernel is missing from " + xclbin_path);
        }

        device.register_xclbin(xclbin);
        hardware_context = xrt::hw_context(device, xclbin.get_uuid());
        kernel = xrt::kernel(hardware_context, found->get_name());
        instruction_bo = xrt::bo(
            device,
            instructions.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE,
            kernel.group_id(1));
        weights_bo = xrt::bo(
            device,
            kExpertBatch * kExpertDimension * kExpertDimension * sizeof(int8_t),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(3));
        scales_bo = xrt::bo(
            device,
            kExpertBatch * kExpertDimension * sizeof(float),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(4));
        input_bo = xrt::bo(
            device,
            kExpertBatch * kExpertDimension * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(5));
        output_bo = xrt::bo(
            device,
            kExpertBatch * kExpertDimension * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(6));
        std::memcpy(
            instruction_bo.map<void *>(),
            instructions.data(),
            instructions.size() * sizeof(uint32_t));
        instruction_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    bool compute_experts(ggml_tensor * destination) {
        const ggml_tensor * weights = destination->src[0];
        const ggml_tensor * input = destination->src[1];
        const ggml_tensor * ids = destination->src[2];
        std::lock_guard<std::mutex> lock(mutex);

        const int64_t input_dimension = weights->ne[0];
        const int64_t output_dimension = weights->ne[1];
        const auto * traits = ggml_get_type_traits(weights->type);
        if (traits == nullptr || traits->to_float == nullptr) {
            return false;
        }

        auto * quantized = weights_bo.map<int8_t *>();
        auto * scales = scales_bo.map<float *>();
        auto * input_bf16 = input_bo.map<uint16_t *>();
        const auto * output_bf16 = output_bo.map<const uint16_t *>();
        const auto * selected = static_cast<const int32_t *>(ids->data);
        std::vector<float> row_f32(static_cast<size_t>(input_dimension));

        const int64_t selected_count = ids->ne[0];
        for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
            if (slot >= selected_count) {
                auto * slot_scales = scales + slot * kExpertDimension;
                std::fill(
                    slot_scales,
                    slot_scales + kExpertDimension,
                    0.0f);
                auto * slot_input = input_bf16 + slot * kExpertDimension;
                std::fill(
                    slot_input,
                    slot_input + kExpertDimension,
                    static_cast<uint16_t>(0));
                continue;
            }
            const int32_t expert = selected[slot];
            if (expert < 0 || expert >= weights->ne[2]) {
                return false;
            }
            auto * slot_weights =
                quantized + slot * kExpertDimension * kExpertDimension;
            auto * slot_scales = scales + slot * kExpertDimension;
            const auto * expert_data =
                static_cast<const char *>(weights->data) + expert * weights->nb[2];

            for (int64_t row = 0; row < output_dimension; ++row) {
                traits->to_float(
                    expert_data + row * weights->nb[1],
                    row_f32.data(),
                    input_dimension);
                float maximum = 0.0f;
                for (int64_t column = 0; column < input_dimension; ++column) {
                    maximum = std::max(maximum, std::abs(row_f32[column]));
                }
                const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
                slot_scales[row] = scale;
                auto * output_row = slot_weights + row * kExpertDimension;
                for (int64_t column = 0; column < input_dimension; ++column) {
                    output_row[column] = static_cast<int8_t>(
                        std::nearbyint(row_f32[column] / scale));
                }
            }
            std::fill(
                slot_scales + output_dimension,
                slot_scales + kExpertDimension,
                0.0f);

            const int64_t input_slot = input->ne[1] == 1 ? 0 : slot;
            const auto * input_f32 = reinterpret_cast<const float *>(
                static_cast<const char *>(input->data) + input_slot * input->nb[1]);
            auto * slot_input = input_bf16 + slot * kExpertDimension;
            for (int64_t index = 0; index < input_dimension; ++index) {
                slot_input[index] = float_to_bf16(input_f32[index]);
            }
            std::fill(
                slot_input + input_dimension,
                slot_input + kExpertDimension,
                static_cast<uint16_t>(0));
        }

        weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        scales_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = kernel(
            3,
            instruction_bo,
            instructions.size(),
            weights_bo,
            scales_bo,
            input_bo,
            output_bo);
        if (run.wait() != ERT_CMD_STATE_COMPLETED) {
            return false;
        }
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        for (int64_t slot = 0; slot < selected_count; ++slot) {
            auto * output_f32 = reinterpret_cast<float *>(
                static_cast<char *>(destination->data) + slot * destination->nb[1]);
            const auto * slot_output =
                output_bf16 + slot * kExpertDimension;
            for (int64_t index = 0; index < output_dimension; ++index) {
                output_f32[index] = bf16_to_float(slot_output[index]);
            }
        }

        ++operations;
        ++expert_operations;
        if (expert_operations == 1 || expert_operations % 144 == 0) {
            std::fprintf(
                stderr,
                "XDNA expert offload count: %" PRIu64 "\n",
                expert_operations);
            GGML_LOG_INFO(
                "XDNA expert offload count: %" PRIu64 "\n",
                expert_operations);
        }
        return true;
    }

    bool compute_dense(ggml_tensor * destination) {
        const ggml_tensor * weights = destination->src[0];
        const ggml_tensor * input = destination->src[1];
        std::lock_guard<std::mutex> lock(mutex);

        const int64_t input_dimension = weights->ne[0];
        const int64_t output_dimension = weights->ne[1];
        const auto * traits = ggml_get_type_traits(weights->type);
        if (traits == nullptr || traits->to_float == nullptr) {
            return false;
        }

        auto * quantized = weights_bo.map<int8_t *>();
        auto * scales = scales_bo.map<float *>();
        auto * input_bf16 = input_bo.map<uint16_t *>();
        const auto * output_bf16 = output_bo.map<const uint16_t *>();
        const auto * input_f32 = static_cast<const float *>(input->data);
        std::vector<float> row_f32(static_cast<size_t>(input_dimension));
        std::vector<float> accumulated(
            static_cast<size_t>(output_dimension), 0.0f);

        for (int64_t output_base = 0;
             output_base < output_dimension;
             output_base += kExpertBatch * kExpertDimension) {
            const int64_t output_group = std::min<int64_t>(
                output_dimension - output_base,
                kExpertBatch * kExpertDimension);
            for (int64_t input_base = 0;
                 input_base < input_dimension;
                 input_base += kExpertDimension) {
                const int64_t input_count = std::min<int64_t>(
                    input_dimension - input_base, kExpertDimension);

                for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
                    const int64_t row_base =
                        output_base + slot * kExpertDimension;
                    const int64_t row_count = std::max<int64_t>(
                        0,
                        std::min<int64_t>(
                            kExpertDimension,
                            output_dimension - row_base));
                    auto * slot_weights =
                        quantized + slot * kExpertDimension * kExpertDimension;
                    auto * slot_scales = scales + slot * kExpertDimension;
                    auto * slot_input = input_bf16 + slot * kExpertDimension;

                    for (int64_t index = 0; index < input_count; ++index) {
                        slot_input[index] =
                            float_to_bf16(input_f32[input_base + index]);
                    }
                    std::fill(
                        slot_input + input_count,
                        slot_input + kExpertDimension,
                        static_cast<uint16_t>(0));

                    for (int64_t row = 0; row < row_count; ++row) {
                        traits->to_float(
                            static_cast<const char *>(weights->data) +
                                (row_base + row) * weights->nb[1],
                            row_f32.data(),
                            input_dimension);
                        float maximum = 0.0f;
                        for (int64_t column = 0;
                             column < input_count;
                             ++column) {
                            maximum = std::max(
                                maximum,
                                std::abs(row_f32[input_base + column]));
                        }
                        const float scale =
                            maximum > 0.0f ? maximum / 127.0f : 1.0f;
                        slot_scales[row] = scale;
                        auto * output_row =
                            slot_weights + row * kExpertDimension;
                        for (int64_t column = 0;
                             column < input_count;
                             ++column) {
                            output_row[column] = static_cast<int8_t>(
                                std::nearbyint(
                                    row_f32[input_base + column] / scale));
                        }
                    }
                    std::fill(
                        slot_scales + row_count,
                        slot_scales + kExpertDimension,
                        0.0f);
                }

                weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                scales_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                auto run = kernel(
                    3,
                    instruction_bo,
                    instructions.size(),
                    weights_bo,
                    scales_bo,
                    input_bo,
                    output_bo);
                if (run.wait() != ERT_CMD_STATE_COMPLETED) {
                    return false;
                }
                output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

                for (int64_t row = 0; row < output_group; ++row) {
                    const int64_t slot = row / kExpertDimension;
                    const int64_t slot_row = row % kExpertDimension;
                    accumulated[output_base + row] += bf16_to_float(
                        output_bf16[slot * kExpertDimension + slot_row]);
                }
            }
        }

        auto * result = static_cast<float *>(destination->data);
        std::copy(accumulated.begin(), accumulated.end(), result);
        ++operations;
        ++dense_operations;
        if (dense_operations == 1 || dense_operations % 100 == 0) {
            std::fprintf(
                stderr,
                "XDNA dense Qwen offload count: %" PRIu64 "\n",
                dense_operations);
            GGML_LOG_INFO(
                "XDNA dense Qwen offload count: %" PRIu64 "\n",
                dense_operations);
        }
        return true;
    }
};

bool supports_quantized_weight(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

bool supports_experts(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * input = op->src[1];
    const ggml_tensor * ids = op->src[2];
    if (weights == nullptr || input == nullptr || ids == nullptr) {
        return false;
    }
    const bool supported =
           weights->ne[0] > 0 &&
           weights->ne[0] <= kExpertDimension &&
           weights->ne[1] > 0 &&
           weights->ne[1] <= kExpertDimension &&
           supports_quantized_weight(weights->type) &&
           input->type == GGML_TYPE_F32 &&
           ids->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           weights->ne[2] >= kExpertBatch &&
           weights->ne[3] == 1 &&
           input->ne[0] == weights->ne[0] &&
           (input->ne[1] == 1 || input->ne[1] == ids->ne[0]) &&
           input->ne[2] == 1 &&
           input->ne[3] == 1 &&
           ids->ne[0] >= 1 &&
           ids->ne[0] <= kExpertBatch &&
           ids->ne[1] == 1 &&
           op->ne[0] == weights->ne[1] &&
           op->ne[1] == ids->ne[0] &&
           op->ne[2] == 1 &&
           ggml_is_contiguous(weights);
    return supported;
}

bool supports_dense_qwen(const ggml_tensor * op) {
    const char * architecture = std::getenv("GGML_XDNA_MODEL_ARCH");
    if (architecture == nullptr ||
        std::string(architecture).rfind("qwen", 0) != 0 ||
        std::string(architecture).find("moe") != std::string::npos) {
        return false;
    }
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * input = op->src[1];
    if (weights == nullptr || input == nullptr) {
        return false;
    }
    return supports_quantized_weight(weights->type) &&
           input->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           weights->ne[0] > 0 &&
           weights->ne[1] > 0 &&
           weights->ne[1] <= 32768 &&
           weights->ne[2] == 1 &&
           weights->ne[3] == 1 &&
           input->ne[0] == weights->ne[0] &&
           input->ne[1] == 1 &&
           input->ne[2] == 1 &&
           input->ne[3] == 1 &&
           op->ne[0] == weights->ne[1] &&
           op->ne[1] == 1 &&
           ggml_is_contiguous(weights) &&
           ggml_is_contiguous(input);
}

const char * backend_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "XDNA";
}

void backend_free(ggml_backend_t backend) {
    delete static_cast<ggml_backend_xdna_context *>(backend->context);
    delete backend;
}

ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * context = static_cast<ggml_backend_xdna_context *>(backend->context);
    try {
        for (int index = 0; index < graph->n_nodes; ++index) {
            ggml_tensor * node = graph->nodes[index];
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    if (!context->compute_dense(node)) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                case GGML_OP_MUL_MAT_ID:
                    if (!context->compute_experts(node)) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    break;
                default:
                    GGML_ABORT(
                        "%s: unsupported operation %s\n", __func__, ggml_op_desc(node));
            }
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::exception & exception) {
        GGML_LOG_ERROR("XDNA graph execution failed: %s\n", exception.what());
        return GGML_STATUS_FAILED;
    }
}

const ggml_backend_i backend_interface = {
    /* .get_name                = */ backend_name,
    /* .free                    = */ backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

ggml_guid_t backend_guid() {
    static const char guid[] = "AMD-XDNA1-GGML";
    return reinterpret_cast<ggml_guid_t>(const_cast<char *>(guid));
}

const char * device_name(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    return "XDNA0";
}

const char * device_description(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    return "AMD Ryzen AI XDNA1 NPU";
}

void device_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_UNUSED(device);
    *free = 0;
    *total = 0;
}

enum ggml_backend_dev_type device_type(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void device_properties(
    ggml_backend_dev_t device, ggml_backend_dev_props * properties) {
    properties->name = device_name(device);
    properties->description = device_description(device);
    properties->type = device_type(device);
    device_memory(device, &properties->memory_free, &properties->memory_total);
    properties->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

ggml_backend_t device_init(ggml_backend_dev_t device, const char * parameters);

ggml_backend_buffer_type_t device_buffer_type(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    return ggml_backend_cpu_buffer_type();
}

ggml_backend_buffer_t device_buffer_from_host(
    ggml_backend_dev_t device, void * pointer, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(device);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(pointer, size);
}

bool device_supports_op(ggml_backend_dev_t device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return supports_experts(op) || supports_dense_qwen(op);
    }
}

bool device_supports_buffer(
    ggml_backend_dev_t device, ggml_backend_buffer_type_t buffer_type) {
    GGML_UNUSED(device);
    return ggml_backend_buft_is_host(buffer_type);
}

bool device_offload_op(ggml_backend_dev_t device, const ggml_tensor * op) {
    GGML_UNUSED(device);
    return supports_experts(op) || supports_dense_qwen(op);
}

const ggml_backend_device_i device_interface = {
    /* .get_name               = */ device_name,
    /* .get_description        = */ device_description,
    /* .get_memory             = */ device_memory,
    /* .get_type               = */ device_type,
    /* .get_props              = */ device_properties,
    /* .init_backend           = */ device_init,
    /* .get_buffer_type        = */ device_buffer_type,
    /* .get_host_buffer_type   = */ nullptr,
    /* .buffer_from_host_ptr   = */ device_buffer_from_host,
    /* .supports_op            = */ device_supports_op,
    /* .supports_buft          = */ device_supports_buffer,
    /* .offload_op             = */ device_offload_op,
    /* .event_new              = */ nullptr,
    /* .event_free             = */ nullptr,
    /* .event_synchronize      = */ nullptr,
};

const char * registry_name(ggml_backend_reg_t registry) {
    GGML_UNUSED(registry);
    return "XDNA";
}

size_t registry_device_count(ggml_backend_reg_t registry) {
    GGML_UNUSED(registry);
    // Only advertise the XDNA device when it is actually configured. Without
    // the xclbin/instruction blobs device_init() cannot succeed, and
    // advertising a device whose init fails aborts context creation for every
    // model -- even pure CPU/GPU inference that never asked for XDNA. Report
    // zero devices so ggml transparently falls back when XDNA is unconfigured.
    if (std::getenv("GGML_XDNA_XCLBIN") == nullptr ||
        std::getenv("GGML_XDNA_INSTS") == nullptr) {
        return 0;
    }
    return 1;
}

ggml_backend_dev_t registry_device(ggml_backend_reg_t registry, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = {
        /* .iface   = */ device_interface,
        /* .reg     = */ registry,
        /* .context = */ nullptr,
    };
    return &device;
}

void * registry_proc_address(ggml_backend_reg_t registry, const char * name) {
    GGML_UNUSED(registry);
    GGML_UNUSED(name);
    return nullptr;
}

const ggml_backend_reg_i registry_interface = {
    /* .get_name         = */ registry_name,
    /* .get_device_count = */ registry_device_count,
    /* .get_device       = */ registry_device,
    /* .get_proc_address = */ registry_proc_address,
};

ggml_backend_reg_t xdna_registry() {
    static ggml_backend_reg registry = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ registry_interface,
        /* .context     = */ nullptr,
    };
    return &registry;
}

ggml_backend_t device_init(ggml_backend_dev_t device, const char * parameters) {
    GGML_UNUSED(parameters);
    const char * xclbin_path = std::getenv("GGML_XDNA_XCLBIN");
    const char * inst_path = std::getenv("GGML_XDNA_INSTS");
    if (xclbin_path == nullptr || inst_path == nullptr) {
        GGML_LOG_ERROR(
            "XDNA backend requires GGML_XDNA_XCLBIN and GGML_XDNA_INSTS\n");
        return nullptr;
    }
    try {
        auto * context = new ggml_backend_xdna_context(xclbin_path, inst_path);
        return new ggml_backend {
            /* .guid    = */ backend_guid(),
            /* .iface   = */ backend_interface,
            /* .device  = */ device,
            /* .context = */ context,
        };
    } catch (const std::exception & exception) {
        GGML_LOG_ERROR("failed to initialize XDNA backend: %s\n", exception.what());
        return nullptr;
    }
}

}  // namespace

GGML_BACKEND_DL_IMPL(xdna_registry)
