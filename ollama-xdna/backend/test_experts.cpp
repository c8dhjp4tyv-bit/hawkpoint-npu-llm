#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace {

constexpr int kExperts = 8;
constexpr int kRows = 2048;
constexpr int kColumns = 2048;

std::vector<uint32_t> load_instructions(const std::string & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open instructions: " + path);
    }
    const auto bytes = stream.tellg();
    if (bytes < 0 || bytes % sizeof(uint32_t) != 0) {
        throw std::runtime_error("invalid instruction file");
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

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_experts EXPERT_XCLBIN INSTS_BIN\n";
        return 2;
    }

    try {
        const size_t weight_count =
            static_cast<size_t>(kExperts) * kRows * kColumns;
        std::vector<int8_t> weights(weight_count);
        std::vector<float> scales(kExperts * kRows);
        std::vector<uint16_t> input(kExperts * kColumns);
        for (int expert = 0; expert < kExperts; ++expert) {
            for (int row = 0; row < kRows; ++row) {
                scales[expert * kRows + row] =
                    0.0005f * static_cast<float>(1 + (row % 5));
                auto * row_data =
                    weights.data() +
                    (static_cast<size_t>(expert) * kRows + row) * kColumns;
                for (int column = 0; column < kColumns; ++column) {
                    row_data[column] = static_cast<int8_t>(
                        ((expert * 7 + row * 3 + column * 5) % 31) - 15);
                }
            }
            for (int column = 0; column < kColumns; ++column) {
                input[expert * kColumns + column] = float_to_bf16(
                    0.002f * static_cast<float>(((expert + column) % 17) - 8));
            }
        }

        const auto instructions = load_instructions(argv[2]);
        xrt::device device(0);
        xrt::xclbin xclbin{std::string(argv[1])};
        const auto kernels = xclbin.get_kernels();
        const auto found = std::find_if(
            kernels.begin(), kernels.end(), [](const xrt::xclbin::kernel & kernel) {
                return kernel.get_name().rfind("MLIR_AIE", 0) == 0;
            });
        if (found == kernels.end()) {
            throw std::runtime_error("MLIR_AIE kernel is missing");
        }

        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        xrt::kernel kernel(context, found->get_name());
        xrt::bo instruction_bo(
            device,
            instructions.size() * sizeof(uint32_t),
            XCL_BO_FLAGS_CACHEABLE,
            kernel.group_id(1));
        xrt::bo weights_bo(
            device, weights.size(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        xrt::bo scales_bo(
            device,
            scales.size() * sizeof(float),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(4));
        xrt::bo input_bo(
            device,
            input.size() * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(5));
        xrt::bo output_bo(
            device,
            static_cast<size_t>(kExperts) * kRows * sizeof(uint16_t),
            XRT_BO_FLAGS_HOST_ONLY,
            kernel.group_id(6));

        std::memcpy(
            instruction_bo.map<void *>(),
            instructions.data(),
            instructions.size() * sizeof(uint32_t));
        std::memcpy(weights_bo.map<void *>(), weights.data(), weights.size());
        std::memcpy(
            scales_bo.map<void *>(), scales.data(), scales.size() * sizeof(float));
        std::memcpy(
            input_bo.map<void *>(), input.data(), input.size() * sizeof(uint16_t));

        instruction_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
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
            throw std::runtime_error("NPU execution failed");
        }
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const auto * output = output_bo.map<const uint16_t *>();

        float maximum_error = 0.0f;
        double mean_error = 0.0;
        size_t checked = 0;
        for (int expert = 0; expert < kExperts; ++expert) {
            for (int row = 0; row < kRows; row += 17) {
                float expected = 0.0f;
                const auto * row_data =
                    weights.data() +
                    (static_cast<size_t>(expert) * kRows + row) * kColumns;
                for (int column = 0; column < kColumns; ++column) {
                    expected +=
                        static_cast<float>(row_data[column]) *
                        bf16_to_float(input[expert * kColumns + column]);
                }
                expected *= scales[expert * kRows + row];
                const float actual =
                    bf16_to_float(output[expert * kRows + row]);
                const float error = std::abs(actual - expected);
                maximum_error = std::max(maximum_error, error);
                mean_error += error;
                ++checked;
            }
        }
        mean_error /= static_cast<double>(checked);
        std::cout << "PASS checked=" << checked
                  << " max_error=" << maximum_error
                  << " mean_error=" << mean_error << '\n';
        return maximum_error < 0.03f ? 0 : 1;
    } catch (const std::exception & exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
