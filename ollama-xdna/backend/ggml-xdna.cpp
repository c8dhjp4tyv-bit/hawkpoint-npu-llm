#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
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
constexpr size_t kDefaultWeightCacheMiB = 512;

struct expert_cache_key {
    const ggml_tensor * tensor;
    const void * data;
    enum ggml_type type;
    int64_t input_dimension;
    int64_t output_dimension;
    int64_t expert_count;
    int32_t expert;

    bool operator==(const expert_cache_key & other) const {
        return tensor == other.tensor && data == other.data &&
               type == other.type && input_dimension == other.input_dimension &&
               output_dimension == other.output_dimension &&
               expert_count == other.expert_count && expert == other.expert;
    }
};

struct expert_cache_key_hash {
    size_t operator()(const expert_cache_key & key) const {
        const size_t tensor_hash = std::hash<const void *>{}(key.tensor);
        const size_t data_hash = std::hash<const void *>{}(key.data);
        const size_t type_hash = std::hash<int>{}(static_cast<int>(key.type));
        const size_t expert_hash = std::hash<int32_t>{}(key.expert);
        size_t result = tensor_hash ^ (data_hash + 0x9e3779b9U +
                                       (tensor_hash << 6U) +
                                       (tensor_hash >> 2U));
        result ^= type_hash + 0x9e3779b9U + (result << 6U) + (result >> 2U);
        result ^= std::hash<int64_t>{}(key.input_dimension) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        result ^= std::hash<int64_t>{}(key.output_dimension) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        result ^= std::hash<int64_t>{}(key.expert_count) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        return result ^ (expert_hash + 0x9e3779b9U + (result << 6U) +
                         (result >> 2U));
    }
};

struct dense_cache_key {
    const ggml_tensor * tensor;
    const void * data;
    enum ggml_type type;
    int64_t input_dimension;
    int64_t output_dimension;
    int64_t output_base;
    int64_t input_base;

    bool operator==(const dense_cache_key & other) const {
        return tensor == other.tensor && data == other.data &&
               type == other.type && input_dimension == other.input_dimension &&
               output_dimension == other.output_dimension &&
               output_base == other.output_base && input_base == other.input_base;
    }
};

struct dense_cache_key_hash {
    size_t operator()(const dense_cache_key & key) const {
        size_t result = std::hash<const void *>{}(key.tensor);
        result ^= std::hash<const void *>{}(key.data) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        result ^= std::hash<int>{}(static_cast<int>(key.type)) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        result ^= std::hash<int64_t>{}(key.input_dimension) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        result ^= std::hash<int64_t>{}(key.output_dimension) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        result ^= std::hash<int64_t>{}(key.output_base) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        result ^= std::hash<int64_t>{}(key.input_base) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        return result;
    }
};

size_t weight_cache_limit_bytes() {
    const char * value = std::getenv("GGML_XDNA_WEIGHT_CACHE_MB");
    if (value == nullptr || *value == '\0') {
        return kDefaultWeightCacheMiB * 1024U * 1024U;
    }
    char * end = nullptr;
    const unsigned long long mib = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        return kDefaultWeightCacheMiB * 1024U * 1024U;
    }
    constexpr unsigned long long kMaxMiB = 64ULL * 1024ULL;
    return static_cast<size_t>(std::min(mib, kMaxMiB) * 1024ULL * 1024ULL);
}

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
    struct packed_expert {
        std::vector<int8_t> weights;
        std::vector<float> scales;
        size_t bytes = 0;
    };

    struct native_packed_expert {
        std::vector<uint8_t> weights;
        size_t bytes = 0;
    };

    struct native_quant_program {
        xrt::device device;
        xrt::xclbin xclbin;
        xrt::hw_context hardware_context;
        xrt::kernel kernel;
        std::vector<uint32_t> instructions;
        xrt::bo instruction_bo;
        xrt::bo weights_bo;
        xrt::bo input_bo;
        xrt::bo output_bo;
        uint8_t * weights_host = nullptr;
        uint16_t * input_host = nullptr;
        const uint16_t * output_host = nullptr;
        size_t row_bytes;
        size_t block_bytes;
        size_t source_block_bytes;

        native_quant_program(
            const xrt::device & source_device,
            const std::string & xclbin_path,
            const std::string & inst_path,
            size_t native_row_bytes,
            size_t native_block_bytes,
            size_t native_source_block_bytes)
            : device(source_device),
              xclbin(std::string(xclbin_path)),
              hardware_context(),
              kernel(),
              instructions(load_instructions(inst_path)),
              instruction_bo(),
              weights_bo(),
              input_bo(),
              output_bo(),
              row_bytes(native_row_bytes),
              block_bytes(native_block_bytes),
              source_block_bytes(native_source_block_bytes) {
            const auto kernels = xclbin.get_kernels();
            const auto found = std::find_if(
                kernels.begin(), kernels.end(),
                [](const xrt::xclbin::kernel & candidate) {
                    return candidate.get_name().rfind("MLIR_AIE", 0) == 0;
                });
            if (found == kernels.end()) {
                throw std::runtime_error(
                    "native quantized MLIR_AIE kernel is missing from " +
                    xclbin_path);
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
                kExpertBatch * kExpertDimension * row_bytes,
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(3));
            input_bo = xrt::bo(
                device,
                kExpertBatch * kExpertDimension * sizeof(uint16_t),
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(4));
            output_bo = xrt::bo(
                device,
                kExpertBatch * kExpertDimension * sizeof(uint16_t),
                XRT_BO_FLAGS_HOST_ONLY,
                kernel.group_id(5));
            std::memset(
                weights_bo.map<void *>(),
                0,
                kExpertBatch * kExpertDimension * row_bytes);
            std::memset(
                input_bo.map<void *>(),
                0,
                kExpertBatch * kExpertDimension * sizeof(uint16_t));
            weights_host = weights_bo.map<uint8_t *>();
            input_host = input_bo.map<uint16_t *>();
            output_host = output_bo.map<const uint16_t *>();
            std::memcpy(
                instruction_bo.map<void *>(),
                instructions.data(),
                instructions.size() * sizeof(uint32_t));
            instruction_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
    };

    struct native_persistent_dense_tile {
        xrt::bo weights;
        uint8_t * weights_host = nullptr;
        size_t bytes = 0;

        native_persistent_dense_tile(
            const xrt::device & device,
            const xrt::kernel & kernel,
            size_t weight_bytes)
            : weights(
                  device,
                  weight_bytes,
                  XRT_BO_FLAGS_HOST_ONLY,
                  kernel.group_id(3)),
              weights_host(weights.map<uint8_t *>()),
              bytes(weight_bytes) {}
    };

    struct persistent_dense_tile {
        xrt::bo weights;
        xrt::bo scales;
        int8_t * weights_host = nullptr;
        float * scales_host = nullptr;
        size_t bytes = 0;

        persistent_dense_tile(
            const xrt::device & device,
            const xrt::kernel & kernel,
            size_t weight_bytes,
            size_t scale_count)
            : weights(
                  device,
                  weight_bytes,
                  XRT_BO_FLAGS_HOST_ONLY,
                  kernel.group_id(3)),
              scales(
                  device,
                  scale_count * sizeof(float),
                  XRT_BO_FLAGS_HOST_ONLY,
                  kernel.group_id(4)),
              weights_host(weights.map<int8_t *>()),
              scales_host(scales.map<float *>()),
              bytes(weight_bytes + scale_count * sizeof(float)) {}
    };

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
    std::unique_ptr<native_quant_program> native_quant;
    std::string native_quant_format;
    std::mutex mutex;
    uint64_t operations = 0;
    uint64_t router_operations = 0;
    uint64_t expert_operations = 0;
    uint64_t dense_operations = 0;
    uint64_t expert_cache_hits = 0;
    uint64_t expert_cache_misses = 0;
    uint64_t dense_cache_hits = 0;
    uint64_t dense_cache_misses = 0;
    uint64_t weight_uploads = 0;
    uint64_t weight_upload_bytes = 0;
    uint64_t native_operations = 0;
    uint64_t native_uploads = 0;
    uint64_t native_upload_bytes = 0;

    const size_t weight_cache_limit = weight_cache_limit_bytes();
    size_t expert_cache_bytes = 0;
    size_t dense_cache_bytes = 0;
    std::unordered_map<
        expert_cache_key,
        std::shared_ptr<packed_expert>,
        expert_cache_key_hash>
        expert_cache;
    std::list<expert_cache_key> expert_lru;
    std::unordered_map<
        dense_cache_key,
        std::shared_ptr<persistent_dense_tile>,
        dense_cache_key_hash>
        dense_cache;
    std::list<dense_cache_key> dense_lru;
    std::unordered_map<
        expert_cache_key,
        std::shared_ptr<native_packed_expert>,
        expert_cache_key_hash>
        native_cache;
    std::list<expert_cache_key> native_lru;
    size_t native_cache_bytes = 0;
    std::unordered_map<
        dense_cache_key,
        std::shared_ptr<native_persistent_dense_tile>,
        dense_cache_key_hash>
        native_dense_cache;
    std::list<dense_cache_key> native_dense_lru;
    size_t native_dense_cache_bytes = 0;
    std::array<std::shared_ptr<packed_expert>, kExpertBatch> uploaded_experts{};
    const ggml_tensor * uploaded_expert_owner = nullptr;
    const void * uploaded_expert_data = nullptr;
    enum ggml_type uploaded_expert_type = GGML_TYPE_COUNT;
    int64_t uploaded_expert_input_dimension = 0;
    int64_t uploaded_expert_output_dimension = 0;
    std::array<std::shared_ptr<native_packed_expert>, kExpertBatch>
        uploaded_native_experts{};
    const ggml_tensor * uploaded_native_owner = nullptr;
    const void * uploaded_native_data = nullptr;
    enum ggml_type uploaded_native_type = GGML_TYPE_COUNT;
    int64_t uploaded_native_input_dimension = 0;
    int64_t uploaded_native_output_dimension = 0;

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
        std::memset(
            weights_bo.map<void *>(),
            0,
            kExpertBatch * expert_weight_slot_bytes());
        std::memset(
            scales_bo.map<void *>(),
            0,
            kExpertBatch * expert_scale_slot_bytes());
        weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        scales_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::memcpy(
            instruction_bo.map<void *>(),
            instructions.data(),
            instructions.size() * sizeof(uint32_t));
        instruction_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const char * native_xclbin =
            std::getenv("GGML_XDNA_NATIVE_QUANT_XCLBIN");
        const char * native_insts =
            std::getenv("GGML_XDNA_NATIVE_QUANT_INSTS");
        const char * native_format =
            std::getenv("GGML_XDNA_NATIVE_QUANT_FORMAT");
        if (native_xclbin != nullptr && native_insts != nullptr &&
            native_format != nullptr) {
            size_t block_bytes = 0;
            size_t source_block_bytes = 0;
            if (std::string(native_format) == "q4_k") {
                block_bytes = 160;
                source_block_bytes = 144;
            } else if (std::string(native_format) == "q6_k") {
                block_bytes = 224;
                source_block_bytes = 210;
            }
            if (block_bytes == 0) {
                GGML_LOG_WARN(
                    "ignoring native XDNA quant program with unknown format %s\n",
                    native_format);
            } else {
                try {
                    native_quant = std::make_unique<native_quant_program>(
                        device,
                        native_xclbin,
                        native_insts,
                        static_cast<size_t>(kExpertDimension / 256) *
                            block_bytes,
                        block_bytes,
                        source_block_bytes);
                    native_quant_format = native_format;
                    GGML_LOG_INFO(
                        "XDNA native %s quantized kernel enabled\n",
                        native_format);
                } catch (const std::exception & exception) {
                    GGML_LOG_WARN(
                        "native XDNA quant program disabled: %s\n",
                        exception.what());
                }
            }
        }

        if (weight_cache_limit == 0) {
            GGML_LOG_INFO(
                "XDNA persistent weight cache disabled (GGML_XDNA_WEIGHT_CACHE_MB=0)\n");
        } else {
            GGML_LOG_INFO(
                "XDNA persistent weight cache limit: %zu MiB\n",
                weight_cache_limit / (1024U * 1024U));
        }
    }

    static constexpr size_t expert_weight_slot_bytes() {
        return static_cast<size_t>(kExpertDimension) *
               static_cast<size_t>(kExpertDimension);
    }

    static constexpr size_t expert_scale_slot_bytes() {
        return static_cast<size_t>(kExpertDimension) * sizeof(float);
    }

    void touch_expert(const expert_cache_key & key) {
        const auto found = std::find(expert_lru.begin(), expert_lru.end(), key);
        if (found != expert_lru.end()) {
            expert_lru.erase(found);
        }
        expert_lru.push_back(key);
    }

    void touch_native_expert(const expert_cache_key & key) {
        const auto found = std::find(native_lru.begin(), native_lru.end(), key);
        if (found != native_lru.end()) {
            native_lru.erase(found);
        }
        native_lru.push_back(key);
    }

    void touch_dense(const dense_cache_key & key) {
        const auto found = std::find(dense_lru.begin(), dense_lru.end(), key);
        if (found != dense_lru.end()) {
            dense_lru.erase(found);
        }
        dense_lru.push_back(key);
    }

    void evict_experts(size_t required) {
        while (expert_cache_bytes + required > weight_cache_limit &&
               !expert_lru.empty()) {
            const expert_cache_key key = expert_lru.front();
            expert_lru.pop_front();
            const auto found = expert_cache.find(key);
            if (found == expert_cache.end()) {
                continue;
            }
            expert_cache_bytes -= found->second->bytes;
            expert_cache.erase(found);
        }
    }

    void evict_dense(size_t required) {
        while (dense_cache_bytes + required > weight_cache_limit &&
               !dense_lru.empty()) {
            const dense_cache_key key = dense_lru.front();
            dense_lru.pop_front();
            const auto found = dense_cache.find(key);
            if (found == dense_cache.end()) {
                continue;
            }
            dense_cache_bytes -= found->second->bytes;
            dense_cache.erase(found);
        }
    }

    void evict_native_experts(size_t required) {
        while (native_cache_bytes + required > weight_cache_limit &&
               !native_lru.empty()) {
            const expert_cache_key key = native_lru.front();
            native_lru.pop_front();
            const auto found = native_cache.find(key);
            if (found == native_cache.end()) {
                continue;
            }
            native_cache_bytes -= found->second->bytes;
            native_cache.erase(found);
        }
    }

    std::shared_ptr<packed_expert> pack_expert(
        const ggml_tensor * weights,
        int32_t expert) {
        const int64_t input_dimension = weights->ne[0];
        const int64_t output_dimension = weights->ne[1];
        const auto * traits = ggml_get_type_traits(weights->type);
        if (traits == nullptr || traits->to_float == nullptr) {
            throw std::runtime_error("GGML weight type cannot be dequantized");
        }

        auto packed = std::make_shared<packed_expert>();
        packed->weights.assign(
            expert_weight_slot_bytes(),
            static_cast<int8_t>(0));
        packed->scales.assign(
            kExpertDimension,
            0.0f);
        packed->bytes = packed->weights.size() * sizeof(int8_t) +
                        packed->scales.size() * sizeof(float);

        auto * slot_weights = packed->weights.data();
        auto * slot_scales = packed->scales.data();
        const auto * expert_data = static_cast<const char *>(weights->data) +
                                   static_cast<size_t>(expert) * weights->nb[2];
        std::vector<float> row_f32(static_cast<size_t>(input_dimension));
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
        return packed;
    }

    std::shared_ptr<packed_expert> get_expert(
        const ggml_tensor * weights,
        int32_t expert) {
        const expert_cache_key key{
            weights,
            weights->data,
            weights->type,
            weights->ne[0],
            weights->ne[1],
            weights->ne[2],
            expert};
        const auto found = expert_cache.find(key);
        if (found != expert_cache.end()) {
            ++expert_cache_hits;
            touch_expert(key);
            return found->second;
        }

        ++expert_cache_misses;
        auto packed = pack_expert(weights, expert);
        if (weight_cache_limit == 0 || packed->bytes > weight_cache_limit) {
            return packed;
        }

        evict_experts(packed->bytes);
        expert_cache.emplace(key, packed);
        expert_cache_bytes += packed->bytes;
        expert_lru.push_back(key);
        return packed;
    }

    std::shared_ptr<persistent_dense_tile> pack_dense_tile(
        const ggml_tensor * weights,
        int64_t output_base,
        int64_t input_base) {
        const int64_t input_dimension = weights->ne[0];
        const int64_t output_dimension = weights->ne[1];
        const int64_t input_count = std::min<int64_t>(
            input_dimension - input_base, kExpertDimension);
        const auto * traits = ggml_get_type_traits(weights->type);
        if (traits == nullptr || traits->to_float == nullptr) {
            throw std::runtime_error("GGML weight type cannot be dequantized");
        }

        auto tile = std::make_shared<persistent_dense_tile>(
            device,
            kernel,
            static_cast<size_t>(kExpertBatch) * expert_weight_slot_bytes(),
            static_cast<size_t>(kExpertBatch) * kExpertDimension);
        std::fill(
            tile->weights_host,
            tile->weights_host +
                static_cast<size_t>(kExpertBatch) * expert_weight_slot_bytes(),
            static_cast<int8_t>(0));
        std::fill(
            tile->scales_host,
            tile->scales_host +
                static_cast<size_t>(kExpertBatch) * kExpertDimension,
            0.0f);

        std::vector<float> row_f32(static_cast<size_t>(input_dimension));
        for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
            const int64_t row_base = output_base + slot * kExpertDimension;
            const int64_t row_count = std::max<int64_t>(
                0,
                std::min<int64_t>(
                    kExpertDimension, output_dimension - row_base));
            auto * slot_weights =
                tile->weights_host + slot * expert_weight_slot_bytes();
            auto * slot_scales = tile->scales_host + slot * kExpertDimension;
            for (int64_t row = 0; row < row_count; ++row) {
                traits->to_float(
                    static_cast<const char *>(weights->data) +
                        (row_base + row) * weights->nb[1],
                    row_f32.data(),
                    input_dimension);
                float maximum = 0.0f;
                for (int64_t column = 0; column < input_count; ++column) {
                    maximum = std::max(
                        maximum,
                        std::abs(row_f32[input_base + column]));
                }
                const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
                slot_scales[row] = scale;
                auto * output_row = slot_weights + row * kExpertDimension;
                for (int64_t column = 0; column < input_count; ++column) {
                    output_row[column] = static_cast<int8_t>(
                        std::nearbyint(
                            row_f32[input_base + column] / scale));
                }
            }
        }
        tile->weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        tile->scales.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return tile;
    }

    std::shared_ptr<persistent_dense_tile> get_dense_tile(
        const ggml_tensor * weights,
        int64_t output_base,
        int64_t input_base) {
        const dense_cache_key key{
            weights,
            weights->data,
            weights->type,
            weights->ne[0],
            weights->ne[1],
            output_base,
            input_base};
        const auto found = dense_cache.find(key);
        if (found != dense_cache.end()) {
            ++dense_cache_hits;
            touch_dense(key);
            return found->second;
        }

        ++dense_cache_misses;
        const size_t tile_bytes =
            static_cast<size_t>(kExpertBatch) * expert_weight_slot_bytes() +
            static_cast<size_t>(kExpertBatch) * kExpertDimension *
                sizeof(float);
        if (weight_cache_limit == 0 || tile_bytes > weight_cache_limit) {
            return nullptr;
        }

        evict_dense(tile_bytes);
        auto tile = pack_dense_tile(weights, output_base, input_base);
        dense_cache.emplace(key, tile);
        dense_cache_bytes += tile->bytes;
        dense_lru.push_back(key);
        return tile;
    }

    void reset_uploaded_experts(const ggml_tensor * owner) {
        if (uploaded_expert_owner != owner ||
            uploaded_expert_data != owner->data ||
            uploaded_expert_type != owner->type ||
            uploaded_expert_input_dimension != owner->ne[0] ||
            uploaded_expert_output_dimension != owner->ne[1]) {
            uploaded_experts.fill(nullptr);
            uploaded_expert_owner = owner;
            uploaded_expert_data = owner->data;
            uploaded_expert_type = owner->type;
            uploaded_expert_input_dimension = owner->ne[0];
            uploaded_expert_output_dimension = owner->ne[1];
        }
    }

    void reset_uploaded_native_experts(const ggml_tensor * owner) {
        if (uploaded_native_owner != owner ||
            uploaded_native_data != owner->data ||
            uploaded_native_type != owner->type ||
            uploaded_native_input_dimension != owner->ne[0] ||
            uploaded_native_output_dimension != owner->ne[1]) {
            uploaded_native_experts.fill(nullptr);
            uploaded_native_owner = owner;
            uploaded_native_data = owner->data;
            uploaded_native_type = owner->type;
            uploaded_native_input_dimension = owner->ne[0];
            uploaded_native_output_dimension = owner->ne[1];
        }
    }

    std::shared_ptr<native_packed_expert> pack_native_expert(
        const ggml_tensor * weights,
        int32_t expert) {
        if (!native_quant || weights->ne[0] % 256 != 0 ||
            weights->nb[1] > native_quant->row_bytes) {
            return nullptr;
        }
        auto packed = std::make_shared<native_packed_expert>();
        packed->weights.assign(
            static_cast<size_t>(kExpertDimension) * native_quant->row_bytes,
            static_cast<uint8_t>(0));
        packed->bytes = packed->weights.size();
        const auto * expert_data = static_cast<const char *>(weights->data) +
                                   static_cast<size_t>(expert) * weights->nb[2];
        const size_t source_row_bytes = static_cast<size_t>(weights->nb[1]);
        const size_t source_block_count =
            static_cast<size_t>(weights->ne[0] / 256);
        if (source_row_bytes !=
            source_block_count * native_quant->source_block_bytes) {
            return nullptr;
        }
        for (int64_t row = 0; row < weights->ne[1]; ++row) {
            auto * destination = packed->weights.data() +
                                 static_cast<size_t>(row) *
                                     native_quant->row_bytes;
            const auto * source = expert_data + row * weights->nb[1];
            for (size_t block = 0; block < source_block_count; ++block) {
                std::memcpy(
                    destination + block * native_quant->block_bytes,
                    source + block * native_quant->source_block_bytes,
                    native_quant->source_block_bytes);
            }
        }
        return packed;
    }

    std::shared_ptr<native_packed_expert> get_native_expert(
        const ggml_tensor * weights,
        int32_t expert) {
        const expert_cache_key key{
            weights,
            weights->data,
            weights->type,
            weights->ne[0],
            weights->ne[1],
            weights->ne[2],
            expert};
        const auto found = native_cache.find(key);
        if (found != native_cache.end()) {
            touch_native_expert(key);
            return found->second;
        }
        auto packed = pack_native_expert(weights, expert);
        if (!packed || weight_cache_limit == 0 ||
            packed->bytes > weight_cache_limit) {
            return packed;
        }
        evict_native_experts(packed->bytes);
        if (native_cache_bytes + packed->bytes > weight_cache_limit) {
            return packed;
        }
        native_cache.emplace(key, packed);
        native_cache_bytes += packed->bytes;
        native_lru.push_back(key);
        return packed;
    }

    bool supports_native_experts(const ggml_tensor * weights) const {
        if (!native_quant || weights == nullptr || weights->ne[0] % 256 != 0) {
            return false;
        }
        if (native_quant_format == "q4_k") {
            return weights->type == GGML_TYPE_Q4_K;
        }
        if (native_quant_format == "q6_k") {
            return weights->type == GGML_TYPE_Q6_K;
        }
        return false;
    }

    bool supports_native_dense(const ggml_tensor * weights) const {
        if (!native_quant || weights == nullptr || weights->ne[0] % 256 != 0 ||
            weights->ne[2] != 1 || weights->ne[3] != 1) {
            return false;
        }
        if (native_quant_format == "q4_k") {
            return weights->type == GGML_TYPE_Q4_K;
        }
        if (native_quant_format == "q6_k") {
            return weights->type == GGML_TYPE_Q6_K;
        }
        return false;
    }

    void touch_native_dense(const dense_cache_key & key) {
        const auto found = std::find(
            native_dense_lru.begin(), native_dense_lru.end(), key);
        if (found != native_dense_lru.end()) {
            native_dense_lru.erase(found);
        }
        native_dense_lru.push_back(key);
    }

    void evict_native_dense(size_t required) {
        while (native_dense_cache_bytes + required > weight_cache_limit &&
               !native_dense_lru.empty()) {
            const dense_cache_key key = native_dense_lru.front();
            native_dense_lru.pop_front();
            const auto found = native_dense_cache.find(key);
            if (found == native_dense_cache.end()) {
                continue;
            }
            native_dense_cache_bytes -= found->second->bytes;
            native_dense_cache.erase(found);
        }
    }

    std::shared_ptr<native_persistent_dense_tile> pack_native_dense_tile(
        const ggml_tensor * weights,
        int64_t output_base,
        int64_t input_base) {
        if (!native_quant || weights->ne[0] % 256 != 0 ||
            weights->nb[1] > native_quant->row_bytes) {
            return nullptr;
        }
        const int64_t input_count = std::min<int64_t>(
            weights->ne[0] - input_base, kExpertDimension);
        if (input_count % 256 != 0) {
            return nullptr;
        }
        const size_t source_block = static_cast<size_t>(input_base / 256);
        const size_t block_count = static_cast<size_t>(input_count / 256);
        const size_t source_row_bytes = static_cast<size_t>(weights->nb[1]);
        if (source_row_bytes !=
            static_cast<size_t>(weights->ne[0] / 256) *
                native_quant->source_block_bytes) {
            return nullptr;
        }
        auto tile = std::make_shared<native_persistent_dense_tile>(
            device,
            native_quant->kernel,
            static_cast<size_t>(kExpertBatch) * kExpertDimension *
                native_quant->row_bytes);
        std::memset(
            tile->weights_host,
            0,
            static_cast<size_t>(kExpertBatch) * kExpertDimension *
                native_quant->row_bytes);
        for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
            const int64_t row_base = output_base + slot * kExpertDimension;
            const int64_t row_count = std::max<int64_t>(
                0,
                std::min<int64_t>(
                    kExpertDimension, weights->ne[1] - row_base));
            for (int64_t row = 0; row < row_count; ++row) {
                const auto * source = static_cast<const char *>(weights->data) +
                                      (row_base + row) * weights->nb[1] +
                                      source_block *
                                          native_quant->source_block_bytes;
                auto * destination = tile->weights_host +
                                     (static_cast<size_t>(slot) *
                                          kExpertDimension + row) *
                                         native_quant->row_bytes;
                for (size_t block = 0; block < block_count; ++block) {
                    std::memcpy(
                        destination + block * native_quant->block_bytes,
                        source + block * native_quant->source_block_bytes,
                        native_quant->source_block_bytes);
                }
            }
        }
        tile->weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return tile;
    }

    std::shared_ptr<native_persistent_dense_tile> get_native_dense_tile(
        const ggml_tensor * weights,
        int64_t output_base,
        int64_t input_base) {
        const dense_cache_key key{
            weights,
            weights->data,
            weights->type,
            weights->ne[0],
            weights->ne[1],
            output_base,
            input_base};
        const auto found = native_dense_cache.find(key);
        if (found != native_dense_cache.end()) {
            touch_native_dense(key);
            return found->second;
        }
        const size_t tile_bytes = static_cast<size_t>(kExpertBatch) *
                                  kExpertDimension * native_quant->row_bytes;
        if (weight_cache_limit == 0 || tile_bytes > weight_cache_limit) {
            return nullptr;
        }
        evict_native_dense(tile_bytes);
        auto tile = pack_native_dense_tile(weights, output_base, input_base);
        if (!tile) {
            return nullptr;
        }
        native_dense_cache.emplace(key, tile);
        native_dense_cache_bytes += tile->bytes;
        native_dense_lru.push_back(key);
        return tile;
    }

    bool compute_dense_native(ggml_tensor * destination) {
        const ggml_tensor * weights = destination->src[0];
        const ggml_tensor * input = destination->src[1];
        std::lock_guard<std::mutex> lock(mutex);
        if (!supports_native_dense(weights)) {
            return false;
        }
        const int64_t input_dimension = weights->ne[0];
        const int64_t output_dimension = weights->ne[1];
        const auto * input_f32 = static_cast<const float *>(input->data);
        auto * input_bf16 = native_quant->input_host;
        const auto * output_bf16 = native_quant->output_host;
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
                std::shared_ptr<native_persistent_dense_tile> tile;
                try {
                    tile = get_native_dense_tile(
                        weights, output_base, input_base);
                } catch (const std::exception & exception) {
                    GGML_LOG_WARN(
                        "XDNA native dense cache unavailable (%s); using W8 path\n",
                        exception.what());
                    return compute_dense_streaming_locked(destination);
                }
                if (!tile) {
                    return compute_dense_streaming_locked(destination);
                }
                for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
                    auto * slot_input = input_bf16 + slot * kExpertDimension;
                    for (int64_t index = 0; index < input_count; ++index) {
                        slot_input[index] = float_to_bf16(
                            input_f32[input_base + index]);
                    }
                    std::fill(
                        slot_input + input_count,
                        slot_input + kExpertDimension,
                        static_cast<uint16_t>(0));
                }
                native_quant->input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                auto run = native_quant->kernel(
                    3,
                    native_quant->instruction_bo,
                    native_quant->instructions.size(),
                    tile->weights,
                    native_quant->input_bo,
                    native_quant->output_bo);
                if (run.wait() != ERT_CMD_STATE_COMPLETED) {
                    return false;
                }
                native_quant->output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                for (int64_t row = 0; row < output_group; ++row) {
                    const int64_t slot = row / kExpertDimension;
                    const int64_t slot_row = row % kExpertDimension;
                    accumulated[output_base + row] += bf16_to_float(
                        output_bf16[slot * kExpertDimension + slot_row]);
                }
            }
        }
        std::copy(
            accumulated.begin(),
            accumulated.end(),
            static_cast<float *>(destination->data));
        ++operations;
        ++dense_operations;
        ++native_operations;
        if (native_operations == 1 || native_operations % 100 == 0) {
            GGML_LOG_INFO(
                "XDNA native %s dense offload count: %" PRIu64
                " (cache=%zu MiB)\n",
                native_quant_format.c_str(),
                native_operations,
                native_dense_cache_bytes / (1024U * 1024U));
        }
        return true;
    }

    bool compute_experts_native(ggml_tensor * destination) {
        const ggml_tensor * weights = destination->src[0];
        const ggml_tensor * input = destination->src[1];
        const ggml_tensor * ids = destination->src[2];
        std::lock_guard<std::mutex> lock(mutex);

        if (!supports_native_experts(weights) || weights->ne[0] > kExpertDimension ||
            weights->ne[1] > kExpertDimension) {
            return false;
        }
        auto * quantized = native_quant->weights_host;
        auto * input_bf16 = native_quant->input_host;
        const auto * output_bf16 = native_quant->output_host;
        const auto * selected = static_cast<const int32_t *>(ids->data);
        const int64_t input_dimension = weights->ne[0];
        const int64_t output_dimension = weights->ne[1];
        const int64_t selected_count = ids->ne[0];
        std::array<bool, kExpertBatch> dirty_slots{};
        reset_uploaded_native_experts(weights);

        for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
            if (slot >= selected_count) {
                if (uploaded_native_experts[slot]) {
                    std::memset(
                        quantized +
                            static_cast<size_t>(slot) * kExpertDimension *
                                native_quant->row_bytes,
                        0,
                        static_cast<size_t>(kExpertDimension) *
                            native_quant->row_bytes);
                    uploaded_native_experts[slot].reset();
                    dirty_slots[slot] = true;
                }
                std::fill(
                    input_bf16 + slot * kExpertDimension,
                    input_bf16 + (slot + 1) * kExpertDimension,
                    static_cast<uint16_t>(0));
                continue;
            }
            const int32_t expert = selected[slot];
            if (expert < 0 || expert >= weights->ne[2]) {
                return false;
            }
            const auto packed = get_native_expert(weights, expert);
            if (!packed) {
                return false;
            }
            if (uploaded_native_experts[slot] != packed) {
                std::memcpy(
                    quantized +
                        static_cast<size_t>(slot) * kExpertDimension *
                            native_quant->row_bytes,
                    packed->weights.data(),
                    packed->weights.size());
                uploaded_native_experts[slot] = packed;
                dirty_slots[slot] = true;
                ++native_uploads;
                native_upload_bytes += packed->bytes;
            }
            const int64_t input_slot = input->ne[1] == 1 ? 0 : slot;
            const auto * input_f32 = reinterpret_cast<const float *>(
                static_cast<const char *>(input->data) +
                input_slot * input->nb[1]);
            auto * slot_input = input_bf16 + slot * kExpertDimension;
            for (int64_t index = 0; index < input_dimension; ++index) {
                slot_input[index] = float_to_bf16(input_f32[index]);
            }
            std::fill(
                slot_input + input_dimension,
                slot_input + kExpertDimension,
                static_cast<uint16_t>(0));
        }

        for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
            if (!dirty_slots[slot]) {
                continue;
            }
            const size_t offset = static_cast<size_t>(slot) *
                                  kExpertDimension * native_quant->row_bytes;
            native_quant->weights_bo.sync(
                XCL_BO_SYNC_BO_TO_DEVICE,
                static_cast<size_t>(kExpertDimension) * native_quant->row_bytes,
                offset);
        }
        native_quant->input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = native_quant->kernel(
            3,
            native_quant->instruction_bo,
            native_quant->instructions.size(),
            native_quant->weights_bo,
            native_quant->input_bo,
            native_quant->output_bo);
        if (run.wait() != ERT_CMD_STATE_COMPLETED) {
            return false;
        }
        native_quant->output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        for (int64_t slot = 0; slot < selected_count; ++slot) {
            auto * output_f32 = reinterpret_cast<float *>(
                static_cast<char *>(destination->data) +
                slot * destination->nb[1]);
            const auto * slot_output =
                output_bf16 + slot * kExpertDimension;
            for (int64_t index = 0; index < output_dimension; ++index) {
                output_f32[index] = bf16_to_float(slot_output[index]);
            }
        }
        ++operations;
        ++expert_operations;
        ++native_operations;
        if (native_operations == 1 || native_operations % 144 == 0) {
            GGML_LOG_INFO(
                "XDNA native %s expert offload count: %" PRIu64
                " (uploads=%" PRIu64 " bytes=%" PRIu64 ")\n",
                native_quant_format.c_str(),
                native_operations,
                native_uploads,
                native_upload_bytes);
        }
        return true;
    }

    bool compute_experts(ggml_tensor * destination) {
        const ggml_tensor * weights = destination->src[0];
        if (supports_native_experts(weights)) {
            return compute_experts_native(destination);
        }
        const ggml_tensor * input = destination->src[1];
        const ggml_tensor * ids = destination->src[2];
        std::lock_guard<std::mutex> lock(mutex);

        const int64_t input_dimension = weights->ne[0];
        const int64_t output_dimension = weights->ne[1];
        auto * quantized = weights_bo.map<int8_t *>();
        auto * scales = scales_bo.map<float *>();
        auto * input_bf16 = input_bo.map<uint16_t *>();
        const auto * output_bf16 = output_bo.map<const uint16_t *>();
        const auto * selected = static_cast<const int32_t *>(ids->data);

        const int64_t selected_count = ids->ne[0];
        reset_uploaded_experts(weights);
        std::array<bool, kExpertBatch> dirty_slots{};
        for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
            if (slot >= selected_count) {
                if (uploaded_experts[slot]) {
                    auto * slot_weights =
                        quantized + slot * expert_weight_slot_bytes();
                    auto * slot_scales = scales + slot * kExpertDimension;
                    std::fill(
                        slot_weights,
                        slot_weights + expert_weight_slot_bytes(),
                        static_cast<int8_t>(0));
                    std::fill(
                        slot_scales,
                        slot_scales + kExpertDimension,
                        0.0f);
                    uploaded_experts[slot].reset();
                    dirty_slots[slot] = true;
                }
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
            const auto packed = get_expert(weights, expert);
            auto * slot_weights =
                quantized + slot * expert_weight_slot_bytes();
            auto * slot_scales = scales + slot * kExpertDimension;
            if (uploaded_experts[slot] != packed) {
                std::memcpy(
                    slot_weights,
                    packed->weights.data(),
                    expert_weight_slot_bytes());
                std::memcpy(
                    slot_scales,
                    packed->scales.data(),
                    expert_scale_slot_bytes());
                uploaded_experts[slot] = packed;
                dirty_slots[slot] = true;
                ++weight_uploads;
                weight_upload_bytes += expert_weight_slot_bytes() +
                                       expert_scale_slot_bytes();
            }

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

        for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
            if (!dirty_slots[slot]) {
                continue;
            }
            const size_t offset =
                static_cast<size_t>(slot) * expert_weight_slot_bytes();
            weights_bo.sync(
                XCL_BO_SYNC_BO_TO_DEVICE,
                expert_weight_slot_bytes(),
                offset);
            scales_bo.sync(
                XCL_BO_SYNC_BO_TO_DEVICE,
                expert_scale_slot_bytes(),
                static_cast<size_t>(slot) * expert_scale_slot_bytes());
        }
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
            GGML_LOG_INFO(
                "XDNA weight cache: expert hits=%" PRIu64
                " misses=%" PRIu64 " uploads=%" PRIu64
                " upload_bytes=%" PRIu64 "\n",
                expert_cache_hits,
                expert_cache_misses,
                weight_uploads,
                weight_upload_bytes);
        }
        return true;
    }

    bool compute_dense(ggml_tensor * destination) {
        const ggml_tensor * weights = destination->src[0];
        if (supports_native_dense(weights) && weight_cache_limit > 0) {
            return compute_dense_native(destination);
        }
        return compute_dense_persistent(destination);
    }

    bool compute_dense_persistent(ggml_tensor * destination) {
        std::lock_guard<std::mutex> lock(mutex);
        if (weight_cache_limit == 0) {
            return compute_dense_streaming_locked(destination);
        }

        const ggml_tensor * weights = destination->src[0];
        const ggml_tensor * input = destination->src[1];
        const int64_t input_dimension = weights->ne[0];
        const int64_t output_dimension = weights->ne[1];
        const auto * input_f32 = static_cast<const float *>(input->data);
        auto * input_bf16 = input_bo.map<uint16_t *>();
        const auto * output_bf16 = output_bo.map<const uint16_t *>();
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
                std::shared_ptr<persistent_dense_tile> tile;
                try {
                    tile = get_dense_tile(weights, output_base, input_base);
                } catch (const std::exception & exception) {
                    GGML_LOG_WARN(
                        "XDNA dense weight cache unavailable (%s); using streaming path\n",
                        exception.what());
                    return compute_dense_streaming_locked(destination);
                }
                if (!tile) {
                    // A cache smaller than one tile or an explicitly disabled
                    // cache uses the original streaming implementation. The
                    // fallback preserves correctness on memory-constrained
                    // systems while the normal path keeps the tile resident.
                    return compute_dense_streaming_locked(destination);
                }

                for (int64_t slot = 0; slot < kExpertBatch; ++slot) {
                    auto * slot_input = input_bf16 + slot * kExpertDimension;
                    std::fill(
                        slot_input,
                        slot_input + kExpertDimension,
                        static_cast<uint16_t>(0));
                    for (int64_t index = 0; index < input_count; ++index) {
                        slot_input[index] = float_to_bf16(
                            input_f32[input_base + index]);
                    }
                }
                input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                auto run = kernel(
                    3,
                    instruction_bo,
                    instructions.size(),
                    tile->weights,
                    tile->scales,
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
                "XDNA dense Qwen offload count: %" PRIu64
                " (tile cache hits=%" PRIu64 " misses=%" PRIu64 ")\n",
                dense_operations,
                dense_cache_hits,
                dense_cache_misses);
        }
        return true;
    }

    bool compute_dense_streaming(ggml_tensor * destination) {
        std::lock_guard<std::mutex> lock(mutex);
        return compute_dense_streaming_locked(destination);
    }

    bool compute_dense_streaming_locked(ggml_tensor * destination) {
        const ggml_tensor * weights = destination->src[0];
        const ggml_tensor * input = destination->src[1];

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
