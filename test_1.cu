#include "cuda_tensor.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

bool are_close(const std::vector<float> &a, const std::vector<float> &b, float tol = 1e-5f)
{
    for (size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > tol)
            return false;
    return true;
}

void test_elementwise_add()
{
    std::cout << "Running test: elementwise_add...\n";
    const size_t shape[2] = {2, 3};
    float host_A[] = {1, 2, 3, 4, 5, 6};
    float host_B[] = {6, 5, 4, 3, 2, 1};
    float expected[] = {7, 7, 7, 7, 7, 7};

    CudaTensor<2> A(shape), B(shape), C(shape);
    A.copy_from_host(host_A);
    B.copy_from_host(host_B);
    C.elementwise_add(A, B);

    std::vector<float> host_result(6);
    C.copy_to_host(host_result.data());

    assert(are_close(host_result, std::vector<float>(expected, expected + 6)));
    std::cout << "✅ Passed elementwise_add\n";
}

void test_scalar_multiply()
{
    std::cout << "Running test: scalar_multiply...\n";
    const size_t shape[2] = {2, 3};
    float host_A[] = {1, 2, 3, 4, 5, 6};
    float expected[] = {2, 4, 6, 8, 10, 12};
    float scalar = 2.0f;

    CudaTensor<2> A(shape), C(shape);
    A.copy_from_host(host_A);
    C.scalar_multiply(A, scalar);

    std::vector<float> host_result(6);
    C.copy_to_host(host_result.data());

    assert(are_close(host_result, std::vector<float>(expected, expected + 6)));
    std::cout << "✅ Passed scalar_multiply\n";
}

void test_matmul_device()
{
    std::cout << "Running test: matmul_device...\n";
    const size_t shapeA[2] = {2, 3};
    const size_t shapeB[2] = {3, 2};

    float host_A[] = {1, 2, 3, 4, 5, 6}; // 2x3
    float host_B[] = {1, 2, 3, 4, 5, 6}; // 3x2
    float expected[] = {
        1 * 1 + 2 * 3 + 3 * 5, 1 * 2 + 2 * 4 + 3 * 6, // Row 0
        4 * 1 + 5 * 3 + 6 * 5, 4 * 2 + 5 * 4 + 6 * 6  // Row 1
    };

    CudaTensor<2> A(shapeA), B(shapeB), C({2, 2});
    A.copy_from_host(host_A);
    B.copy_from_host(host_B);

    CudaTensor<2> result({2, 2});
    CudaTensor<2>::matmul_device(A, B, result);

    std::vector<float> host_result(4);
    result.copy_to_host(host_result.data());

    assert(are_close(host_result, std::vector<float>(expected, expected + 4)));

    std::cout << "✅ Passed matmul_device\n";
}

int main()
{
    test_elementwise_add();
    test_scalar_multiply();
    test_matmul_device();
    std::cout << "\n🎉 All tests passed!\n";
    return 0;
}
