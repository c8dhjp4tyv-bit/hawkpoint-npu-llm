#include "backend_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    require(argc == 3, "usage: test_backend_xdna XCLBIN INSTS");
    setenv("COLI_XDNA_XCLBIN", argv[1], 1);
    setenv("COLI_XDNA_INSTS", argv[2], 1);
    int device = 0;
    require(coli_cuda_init(&device, 1), "XDNA initialization");

    constexpr int input_size = 32;
    constexpr int output_size = 16;
    std::vector<float> weights(input_size * output_size);
    std::vector<float> input(input_size);
    std::vector<float> expected(output_size, 0.0f), actual(output_size, 0.0f);
    for (int col = 0; col < input_size; ++col)
        input[col] = std::sin(static_cast<float>(col + 1) * 0.17f);
    for (int row = 0; row < output_size; ++row)
        for (int col = 0; col < input_size; ++col) {
            const float value = std::cos(static_cast<float>((row + 3) * (col + 1)) * 0.013f);
            weights[row * input_size + col] = value;
            expected[row] += value * input[col];
        }

    ColiCudaTensor *tensor = nullptr;
    require(coli_cuda_matmul(&tensor, actual.data(), input.data(), weights.data(),
                             nullptr, 0, 1, input_size, output_size, 0, 0),
            "direct C matmul");
    float maximum_error = 0.0f;
    for (int row = 0; row < output_size; ++row)
        maximum_error = std::max(maximum_error, std::abs(expected[row] - actual[row]));
    require(maximum_error < 0.15f, "W8/BF16 numerical tolerance");
    coli_cuda_tensor_free(tensor);

    constexpr int grouped_input = 33;
    constexpr int grouped_output = 3;
    constexpr int group = 16;
    constexpr int row_bytes = (grouped_input + 1) / 2;
    constexpr int groups = (grouped_input + group - 1) / group;
    std::vector<uint8_t> packed(grouped_output * row_bytes, 0);
    std::vector<float> grouped_scales(grouped_output * groups);
    std::vector<float> grouped_x(grouped_input), grouped_y(grouped_output), grouped_ref(grouped_output);
    for (int col = 0; col < grouped_input; ++col) grouped_x[col] = (col % 7 - 3) * 0.125f;
    for (int row = 0; row < grouped_output; ++row) {
        for (int block = 0; block < groups; ++block)
            grouped_scales[row * groups + block] = 0.02f * (row + block + 1);
        for (int col = 0; col < grouped_input; ++col) {
            const int value = ((row * 5 + col * 3) % 16) - 8;
            uint8_t &byte = packed[row * row_bytes + col / 2];
            if (col & 1) byte |= static_cast<uint8_t>((value + 8) << 4);
            else byte |= static_cast<uint8_t>(value + 8);
            grouped_ref[row] += value * grouped_scales[row * groups + col / group] * grouped_x[col];
        }
    }
    tensor = nullptr;
    require(coli_cuda_matmul(&tensor, grouped_y.data(), grouped_x.data(), packed.data(),
                             grouped_scales.data(), 4, 1, grouped_input,
                             grouped_output, 0, group),
            "grouped-int4 direct C matmul");
    maximum_error = 0.0f;
    for (int row = 0; row < grouped_output; ++row)
        maximum_error = std::max(maximum_error, std::abs(grouped_ref[row] - grouped_y[row]));
    require(maximum_error < 0.08f, "grouped-int4 W8/BF16 numerical tolerance");
    coli_cuda_tensor_free(tensor);

    constexpr int experts = 2;
    constexpr int hidden = 24;
    std::vector<std::vector<float>> gate_weights(
        experts, std::vector<float>(hidden * input_size));
    std::vector<std::vector<float>> up_weights(
        experts, std::vector<float>(hidden * input_size));
    std::vector<std::vector<float>> down_weights(
        experts, std::vector<float>(input_size * hidden));
    std::vector<float> expert_x(experts * input_size);
    std::vector<float> expert_y(experts * input_size, 0.0f);
    std::vector<float> expert_ref(experts * input_size, 0.0f);
    ColiCudaTensor *gate[experts]{}, *up[experts]{}, *down[experts]{};
    int expert_rows[experts] = {1, 1};
    for (int expert = 0; expert < experts; ++expert) {
        for (int col = 0; col < input_size; ++col)
            expert_x[expert * input_size + col] =
                std::sin(static_cast<float>((expert + 1) * (col + 2)) * 0.09f);
        for (int row = 0; row < hidden; ++row)
            for (int col = 0; col < input_size; ++col) {
                const int index = row * input_size + col;
                gate_weights[expert][index] =
                    std::cos(static_cast<float>((expert + 2) * (row + 1) + col) * 0.031f) * 0.2f;
                up_weights[expert][index] =
                    std::sin(static_cast<float>((expert + 1) * (col + 1) + row) * 0.027f) * 0.2f;
            }
        for (int row = 0; row < input_size; ++row)
            for (int col = 0; col < hidden; ++col)
                down_weights[expert][row * hidden + col] =
                    std::cos(static_cast<float>((row + 1) * (col + 3) + expert) * 0.019f) * 0.1f;

        std::vector<float> gate_ref(hidden, 0.0f), up_ref(hidden, 0.0f);
        for (int row = 0; row < hidden; ++row)
            for (int col = 0; col < input_size; ++col) {
                const float x = expert_x[expert * input_size + col];
                gate_ref[row] += gate_weights[expert][row * input_size + col] * x;
                up_ref[row] += up_weights[expert][row * input_size + col] * x;
            }
        for (int row = 0; row < input_size; ++row)
            for (int col = 0; col < hidden; ++col) {
                const float silu = gate_ref[col] / (1.0f + std::exp(-gate_ref[col]));
                expert_ref[expert * input_size + row] +=
                    down_weights[expert][row * hidden + col] * silu * up_ref[col];
            }
        require(coli_cuda_tensor_upload(&gate[expert], gate_weights[expert].data(),
                                        nullptr, 0, input_size, hidden, 0),
                "expert gate upload");
        require(coli_cuda_tensor_upload(&up[expert], up_weights[expert].data(),
                                        nullptr, 0, input_size, hidden, 0),
                "expert up upload");
        require(coli_cuda_tensor_upload(&down[expert], down_weights[expert].data(),
                                        nullptr, 0, hidden, input_size, 0),
                "expert down upload");
    }
    require(coli_cuda_expert_group_pinned(gate, up, down, expert_rows, experts,
                                          expert_y.data(), expert_x.data(), 0),
            "routed expert direct C pipeline");
    maximum_error = 0.0f;
    for (size_t i = 0; i < expert_y.size(); ++i)
        maximum_error = std::max(maximum_error, std::abs(expert_ref[i] - expert_y[i]));
    require(maximum_error < 0.15f, "routed expert W8/BF16 numerical tolerance");
    for (int expert = 0; expert < experts; ++expert) {
        coli_cuda_tensor_free(gate[expert]);
        coli_cuda_tensor_free(up[expert]);
        coli_cuda_tensor_free(down[expert]);
    }
    coli_cuda_shutdown();
    std::printf("PASS Colibri XDNA direct C backend max_error=%.6f\n", maximum_error);
    return 0;
}
