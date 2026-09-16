#include "backend_cuda.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

constexpr int kSlots = 8;
constexpr int kTile = 2048;

struct ColiCudaTensor {
    int fmt = 0, input = 0, output = 0, group = 0, device = 0;
    std::vector<int8_t> weights;
    std::vector<float> scales;
};

namespace {

uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float from_bf16(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<uint32_t> load_instructions(const char *path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open instruction file");
    const auto bytes = stream.tellg();
    if (bytes <= 0 || bytes % 4) throw std::runtime_error("invalid instruction file");
    stream.seekg(0);
    std::vector<uint32_t> result(static_cast<size_t>(bytes) / 4);
    stream.read(reinterpret_cast<char *>(result.data()), bytes);
    return result;
}

struct Runtime {
    xrt::device device{0};
    xrt::xclbin xclbin;
    xrt::hw_context context;
    xrt::kernel kernel;
    std::vector<uint32_t> instructions;
    xrt::bo instruction_bo, weights_bo, scales_bo, input_bo, output_bo;
    int8_t *weights;
    float *scales;
    uint16_t *input;
    const uint16_t *output;
    std::mutex lock;
    std::atomic<uint64_t> calls{0}, expert_calls{0};

    Runtime(const char *xclbin_path, const char *inst_path)
        : xclbin(std::string(xclbin_path)),
          instructions(load_instructions(inst_path)) {
        const auto kernels = xclbin.get_kernels();
        const auto found = std::find_if(kernels.begin(), kernels.end(), [](const auto &item) {
            return item.get_name().rfind("MLIR_AIE", 0) == 0;
        });
        if (found == kernels.end()) throw std::runtime_error("MLIR_AIE kernel is missing");
        device.register_xclbin(xclbin);
        context = xrt::hw_context(device, xclbin.get_uuid());
        kernel = xrt::kernel(context, found->get_name());
        instruction_bo = xrt::bo(device, instructions.size() * 4,
            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        weights_bo = xrt::bo(device, static_cast<size_t>(kSlots) * kTile * kTile,
            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        scales_bo = xrt::bo(device, static_cast<size_t>(kSlots) * kTile * sizeof(float),
            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        input_bo = xrt::bo(device, static_cast<size_t>(kSlots) * kTile * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        output_bo = xrt::bo(device, static_cast<size_t>(kSlots) * kTile * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        weights = weights_bo.map<int8_t *>();
        scales = scales_bo.map<float *>();
        input = input_bo.map<uint16_t *>();
        output = output_bo.map<const uint16_t *>();
        std::memcpy(instruction_bo.map<void *>(), instructions.data(), instructions.size() * 4);
        instruction_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    bool run() {
        weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        scales_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto command = kernel(3, instruction_bo, instructions.size(), weights_bo,
                              scales_bo, input_bo, output_bo);
        if (command.wait() != ERT_CMD_STATE_COMPLETED) return false;
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        ++calls;
        return true;
    }
};

std::unique_ptr<Runtime> runtime;
std::atomic<size_t> tensors{0}, tensor_bytes{0};

float decoded(const void *weights, const float *scales, int fmt,
              int input, int row, int column, int group) {
    const int64_t index = static_cast<int64_t>(row) * input + column;
    if (fmt == 0) return static_cast<const float *>(weights)[index];
    if (fmt == 1) return static_cast<const int8_t *>(weights)[index] * scales[row];
    const auto *packed = static_cast<const uint8_t *>(weights);
    const int64_t row_bytes = (input + 1) / 2;
    const uint8_t byte = packed[static_cast<int64_t>(row) * row_bytes + column / 2];
    const int value = ((column & 1) ? (byte >> 4) : (byte & 15)) - 8;
    if (fmt == 4) {
        const int groups = (input + group - 1) / group;
        return value * scales[static_cast<int64_t>(row) * groups + column / group];
    }
    return value * scales[row];
}

bool pack_tensor(ColiCudaTensor *tensor, const void *weights, const float *scales) {
    if (!tensor || !weights || tensor->input <= 0 || tensor->output <= 0 ||
        (tensor->fmt == 4 && tensor->group <= 0)) return false;
    if (tensor->fmt < 0 || tensor->fmt > 4 || tensor->fmt == 3) return false;
    tensor->weights.resize(static_cast<size_t>(tensor->input) * tensor->output);
    tensor->scales.resize(tensor->output);
    for (int row = 0; row < tensor->output; ++row) {
        float maximum = 0.0f;
        for (int col = 0; col < tensor->input; ++col)
            maximum = std::max(maximum, std::abs(decoded(
                weights, scales, tensor->fmt, tensor->input, row, col, tensor->group)));
        const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        tensor->scales[row] = scale;
        for (int col = 0; col < tensor->input; ++col) {
            const float value = decoded(
                weights, scales, tensor->fmt, tensor->input, row, col, tensor->group);
            tensor->weights[static_cast<size_t>(row) * tensor->input + col] =
                static_cast<int8_t>(std::clamp(
                    std::nearbyint(value / scale), -127.0f, 127.0f));
        }
    }
    return true;
}

bool project_segment(ColiCudaTensor *const *items, const float *const *inputs,
                     float *const *outputs, int count, int output_base) {
    if (!runtime || !items || !inputs || !outputs || count < 1 || count > kSlots ||
        !items[0]) return false;
    const int input_size = items[0]->input;
    const int output_size = items[0]->output;
    for (int slot = 0; slot < count; ++slot)
        if (!items[slot] || !inputs[slot] || !outputs[slot] ||
            items[slot]->input != input_size || items[slot]->output != output_size)
            return false;
    const int rows = std::min(kTile, output_size - output_base);
    if (rows <= 0) return false;
    for (int slot = 0; slot < count; ++slot)
        std::fill(outputs[slot] + output_base, outputs[slot] + output_base + rows, 0.0f);

    for (int input_base = 0; input_base < input_size; input_base += kTile) {
        const int columns = std::min(kTile, input_size - input_base);
        std::memset(runtime->weights, 0, static_cast<size_t>(kSlots) * kTile * kTile);
        std::fill(runtime->scales, runtime->scales + kSlots * kTile, 0.0f);
        std::fill(runtime->input, runtime->input + kSlots * kTile, uint16_t{0});
        for (int slot = 0; slot < count; ++slot) {
            for (int row = 0; row < rows; ++row) {
                const int source_row = output_base + row;
                std::memcpy(
                    runtime->weights + (static_cast<size_t>(slot) * kTile + row) * kTile,
                    items[slot]->weights.data() + static_cast<size_t>(source_row) * input_size + input_base,
                    columns);
                runtime->scales[slot * kTile + row] = items[slot]->scales[source_row];
            }
            for (int col = 0; col < columns; ++col)
                runtime->input[slot * kTile + col] = to_bf16(inputs[slot][input_base + col]);
        }
        if (!runtime->run()) return false;
        for (int slot = 0; slot < count; ++slot)
            for (int row = 0; row < rows; ++row)
                outputs[slot][output_base + row] += from_bf16(runtime->output[slot * kTile + row]);
    }
    return true;
}

bool project_batch(ColiCudaTensor *const *items, const float *const *inputs,
                   float *const *outputs, int count) {
    if (!runtime || !items || count < 1 || !items[0]) return false;
    std::lock_guard<std::mutex> guard(runtime->lock);
    for (int base = 0; base < items[0]->output; base += kTile)
        if (!project_segment(items, inputs, outputs, count, base)) return false;
    return true;
}

}  // namespace

extern "C" {

int coli_cuda_init(const int *devices, int count) {
    if (count != 1 || !devices || devices[0] != 0) return 0;
    const char *xclbin = std::getenv("COLI_XDNA_XCLBIN");
    const char *insts = std::getenv("COLI_XDNA_INSTS");
    if (!xclbin || !insts) return 0;
    try {
        runtime = std::make_unique<Runtime>(xclbin, insts);
        std::fprintf(stderr, "[XDNA] direct C backend initialized\n");
        return 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[XDNA] initialization failed: %s\n", error.what());
        return 0;
    }
}

void coli_cuda_shutdown(void) { runtime.reset(); }
int coli_cuda_available_device_count(void) { return 1; }
int coli_cuda_device_count(void) { return runtime ? 1 : 0; }
int coli_cuda_device_at(int index) { return index == 0 && runtime ? 0 : -1; }
int coli_cuda_device_integrated(int device) { return device == 0; }
int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    if (device != 0 || !free_bytes || !total_bytes) return 0;
    const char *value = std::getenv("COLI_XDNA_MEMORY_MB");
    char *end = nullptr;
    const unsigned long long parsed = value ? std::strtoull(value, &end, 10) : 512;
    if (value && (!*value || !end || *end || parsed == 0 ||
                  parsed > static_cast<unsigned long long>(SIZE_MAX / (1024U * 1024U)))) return 0;
    const size_t total = static_cast<size_t>(parsed) * 1024U * 1024U;
    *total_bytes = total;
    const size_t used = tensor_bytes.load();
    *free_bytes = total > used ? total - used : 0;
    return 1;
}

void coli_cuda_stats(int, size_t *count, size_t *bytes) {
    if (count) *count = tensors.load();
    if (bytes) *bytes = tensor_bytes.load();
}
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d, double *kernel, double *d2h) {
    if (calls) *calls = runtime ? runtime->expert_calls.load() : 0;
    if (experts) *experts = 0;
    if (rows) *rows = 0;
    if (h2d) *h2d = 0;
    if (kernel) *kernel = 0;
    if (d2h) *d2h = 0;
}
void coli_cuda_group_stats_device(int, uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                  double *h2d, double *kernel, double *d2h) {
    coli_cuda_group_stats(calls, experts, rows, h2d, kernel, d2h);
}
int coli_cuda_e8_set_grid(const void *) { return 0; }
int coli_cuda_fp8_set_lut(const float *) { return 0; }

int coli_cuda_tensor_upload_g(ColiCudaTensor **out, const void *weights, const float *scales,
                              int fmt, int input, int output, int device, int group) {
    if (!out || device != 0 || !runtime || !weights || (fmt != 0 && !scales)) return 0;
    auto tensor = std::make_unique<ColiCudaTensor>();
    tensor->fmt = fmt; tensor->input = input; tensor->output = output;
    tensor->device = device; tensor->group = group;
    if (!pack_tensor(tensor.get(), weights, scales)) return 0;
    tensor_bytes.fetch_add(tensor->weights.size() + tensor->scales.size() * sizeof(float));
    tensors.fetch_add(1);
    *out = tensor.release();
    return 1;
}
int coli_cuda_tensor_upload(ColiCudaTensor **out, const void *weights, const float *scales,
                            int fmt, int input, int output, int device) {
    return coli_cuda_tensor_upload_g(out, weights, scales, fmt, input, output, device, 0);
}
void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    tensor_bytes.fetch_sub(tensor->weights.size() + tensor->scales.size() * sizeof(float));
    tensors.fetch_sub(1);
    delete tensor;
}
size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weights.size() + tensor->scales.size() * sizeof(float) : 0;
}
size_t coli_cuda_tensor_vram(const ColiCudaTensor *tensor) { return coli_cuda_tensor_bytes(tensor); }
size_t coli_cuda_alloc_footprint(size_t bytes) { return bytes; }
int coli_cuda_tensor_device(const ColiCudaTensor *tensor) { return tensor ? tensor->device : -1; }
int coli_cuda_tensor_update(ColiCudaTensor *tensor, const void *weights, const float *scales) {
    return pack_tensor(tensor, weights, scales);
}

int coli_cuda_matmul(ColiCudaTensor **tensor, float *output, const float *input,
                     const void *weights, const float *scales, int fmt, int rows,
                     int input_size, int output_size, int device, int group) {
    if (!tensor || !output || !input || rows != 1) return 0;
    if (!*tensor && !coli_cuda_tensor_upload_g(tensor, weights, scales, fmt,
                                                input_size, output_size, device, group)) return 0;
    ColiCudaTensor *items[] = {*tensor};
    const float *inputs[] = {input}; float *outputs[] = {output};
    return project_batch(items, inputs, outputs, 1);
}

int coli_cuda_expert_group_pinned(ColiCudaTensor *const *gate, ColiCudaTensor *const *up,
                                  ColiCudaTensor *const *down, const int *rows, int count,
                                  float *output, const float *input, int) {
    if (!runtime || !gate || !up || !down || !rows || !output || !input ||
        count < 1 || count > kSlots) return 0;
    std::array<const float *, kSlots> inputs{};
    std::array<float *, kSlots> gates{}, ups{}, activated{}, outputs{};
    std::vector<float> gate_storage, up_storage, activated_storage;
    for (int i = 0; i < count; ++i) {
        if (rows[i] != 1 || !gate[i] || !up[i] || !down[i] ||
            gate[i]->input != up[i]->input || gate[i]->output != up[i]->output ||
            down[i]->input != gate[i]->output || down[i]->output != gate[i]->input) return 0;
    }
    gate_storage.resize(static_cast<size_t>(count) * gate[0]->output);
    up_storage.resize(gate_storage.size()); activated_storage.resize(gate_storage.size());
    int in_cursor = 0, out_cursor = 0;
    for (int i = 0; i < count; ++i) {
        inputs[i] = input + in_cursor; in_cursor += gate[i]->input;
        gates[i] = gate_storage.data() + static_cast<size_t>(i) * gate[0]->output;
        ups[i] = up_storage.data() + static_cast<size_t>(i) * gate[0]->output;
        activated[i] = activated_storage.data() + static_cast<size_t>(i) * gate[0]->output;
        outputs[i] = output + out_cursor; out_cursor += down[i]->output;
    }
    if (!project_batch(gate, inputs.data(), gates.data(), count) ||
        !project_batch(up, inputs.data(), ups.data(), count)) return 0;
    for (size_t i = 0; i < activated_storage.size(); ++i)
        activated_storage[i] = (gates[0][i] / (1.0f + std::exp(-gates[0][i]))) * ups[0][i];
    std::array<const float *, kSlots> activated_inputs{};
    for (int i = 0; i < count; ++i) activated_inputs[i] = activated[i];
    if (!project_batch(down, activated_inputs.data(), outputs.data(), count)) return 0;
    ++runtime->expert_calls;
    return 1;
}
int coli_cuda_expert_group(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                           ColiCudaTensor *const *d, const int *r, int n,
                           float *y, const float *x) {
    return coli_cuda_expert_group_pinned(g, u, d, r, n, y, x, 0);
}
int coli_cuda_expert_mlp(ColiCudaTensor *g, ColiCudaTensor *u, ColiCudaTensor *d,
                         float *y, const float *x, int rows) {
    ColiCudaTensor *ga[] = {g}, *ua[] = {u}, *da[] = {d}; int rs[] = {rows};
    return coli_cuda_expert_group(ga, ua, da, rs, 1, y, x);
}

#define STUB_INT(name, args) int name args { return 0; }
STUB_INT(coli_cuda_shared_mlp_w4a16, (ColiCudaTensor *, ColiCudaTensor *, ColiCudaTensor *, float *, const float *, int))
STUB_INT(coli_cuda_expert_group_issue, (ColiCudaTensor *const *, ColiCudaTensor *const *, ColiCudaTensor *const *, const int *, int, const float *))
const float *coli_cuda_expert_group_take(int) { return nullptr; }
STUB_INT(coli_cuda_attention_absorb, (ColiCudaTensor *, float *, const float *, const float *, const float *, int, int, int, int, int, int, float))
STUB_INT(coli_cuda_attention_absorb_batch, (ColiCudaTensor *, float *, const float *, const float *, const float *, int, int, int, int, int, int, int, float))
STUB_INT(coli_cuda_attention_project_batch, (ColiCudaTensor *, ColiCudaTensor *, float *, const float *, const float *, const float *, int, int, int, int, int, int, int, float))
STUB_INT(coli_cuda_attention_project_ragged, (ColiCudaTensor *, ColiCudaTensor *, float *, const float *, const void *const *, const float *const *, const float *const *, const int *, int, int, int, int, int, int, int, float))

float *coli_cuda_pipe_scratch(int, int, size_t) { return nullptr; }
void *coli_cuda_pipe_alloc(int, size_t) { return nullptr; }
void coli_cuda_pipe_free(int, void *) {}
STUB_INT(coli_cuda_pipe_upload, (int, void *, const void *, size_t))
STUB_INT(coli_cuda_pipe_download, (int, const void *, void *, size_t))
STUB_INT(coli_cuda_pipe_rmsnorm, (int, float *, const float *, const float *, int, int, float))
STUB_INT(coli_cuda_pipe_rope, (int, float *, const int *, int, int, int, int, int, float))
STUB_INT(coli_cuda_pipe_silu_mul, (int, float *, const float *, size_t))
STUB_INT(coli_cuda_pipe_add, (int, float *, const float *, size_t))
STUB_INT(coli_cuda_pipe_rows_add, (int, float *, const float *, const int *, int, int))
STUB_INT(coli_cuda_pipe_gemm, (ColiCudaTensor *, float *, const float *, int))
STUB_INT(coli_cuda_pipe_rmsnorm_s, (int, float *, const float *, const float *, int, int, float, int, int))
STUB_INT(coli_cuda_pipe_rope_base, (int, float *, int, int, int, int, int, int, float))
STUB_INT(coli_cuda_expert_group_resident_issue, (ColiCudaTensor *const *, ColiCudaTensor *const *, ColiCudaTensor *const *, const float *, int, int, const float *, float *))
STUB_INT(coli_cuda_expert_group_resident_take, (int, const int *, int, float *, float *, int))
STUB_INT(coli_cuda_pipe_router, (int, const float *, const void *, const void *, int, int, int, float, int, float, int *, float *, int *))
STUB_INT(coli_cuda_pipe_copy2d, (int, float *, int, const float *, int, int, int))
STUB_INT(coli_cuda_attention_project_batch_dev, (ColiCudaTensor *, ColiCudaTensor *, float *, const float *, const float *, const float *, int, int, int, int, int, int, int, float))
STUB_INT(coli_cuda_attention_absorb_batch_dev, (ColiCudaTensor *, float *, const float *, const float *, const float *, int, int, int, int, int, int, int, float))
STUB_INT(coli_cuda_attention_absorb_kvdev, (ColiCudaTensor *, float *, const float *, const float *, const float *, int, int, int, int, int, int, float))
STUB_INT(coli_cuda_pipe_peer_copy, (int, float *, int, const float *, size_t))
STUB_INT(coli_cuda_attention_project_batch_dev_out, (ColiCudaTensor *, ColiCudaTensor *, float *, const float *, const float *, const float *, int, int, int, int, int, int, int, float))
STUB_INT(coli_cuda_pipe_sync, (int))

}  // extern "C"
