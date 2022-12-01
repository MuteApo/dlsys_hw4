#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

#include "lambda.h"

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
#define STRIDE 8
    typedef float scalar_t;
    __device__ const scalar_t inf = std::numeric_limits<scalar_t>::infinity();
    const size_t ELEM_SIZE = sizeof(scalar_t);
    typedef Py_ssize_t ptrdiff_t;

    struct CudaArray {
        CudaArray(const size_t size) {
            cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
            if (err != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(err));
            this->size = size;
        }
        ~CudaArray() { cudaFree(ptr); }
        size_t ptr_as_int() { return (size_t)ptr; }

        scalar_t *ptr;
        size_t size;
    };

    struct CudaDims {
        dim3 block, grid;
    };

    CudaDims CudaOneDim(size_t size) {
        /**
         * Utility function to get cuda dimensions for 1D call
         */
        CudaDims dim;
        size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
        dim.block = dim3(BASE_THREAD_NUM, 1, 1);
        dim.grid = dim3(num_blocks, 1, 1);
        return dim;
    }

    CudaDims CudaTwoDim(size_t row, size_t col) {
        /**
         * Utility function to get cuda dimensions for 2D call
         */
        CudaDims dim;
        size_t num_rows = (row + STRIDE * TILE - 1) / (STRIDE * TILE);
        size_t num_cols = (col + STRIDE * TILE - 1) / (STRIDE * TILE);
        dim.block = dim3(STRIDE, STRIDE, 1);
        dim.grid = dim3(num_rows, num_cols, 1);
        return dim;
    }

#define MAX_VEC_SIZE 8
    struct CudaVec {
        uint32_t size;
        int32_t data[MAX_VEC_SIZE];
    };

    CudaVec VecToCuda(const std::vector<int32_t> &x) {
        CudaVec shape;
        if (x.size() > MAX_VEC_SIZE)
            throw std::runtime_error("Exceeded CUDA supported max dimesions");
        shape.size = x.size();
        for (size_t i = 0; i < x.size(); i++) {
            shape.data[i] = x[i];
        }
        return shape;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Fill call
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void FillKernel(scalar_t *out, scalar_t val, size_t size) {
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < size) out[gid] = val;
    }

    void Fill(CudaArray *out, scalar_t val) {
        CudaDims dim = CudaOneDim(out->size);
        FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Compact and setitem cals
    ////////////////////////////////////////////////////////////////////////////////

    // Untility function to convert contiguous index i to memory location from
    // strides

    __global__ void CompactKernel(const scalar_t *a,
                                  scalar_t *out,
                                  size_t size,
                                  CudaVec shape,
                                  CudaVec strides,
                                  size_t offset) {
        /**
         * The CUDA kernel for the compact opeation.  This should effectively
         * map a single entry in the non-compact input a, to the corresponding
         * item (at location gid) in the compact array out.
         *
         * Args:
         *   a: CUDA pointer to a array
         *   out: CUDA point to out array
         *   size: size of out array
         *   shape: vector of shapes of a and out arrays (of type CudaVec, for
         * past passing to CUDA kernel) strides: vector of strides of out array
         *   offset: offset of out array
         */
        Py_ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < size) {
            Py_ssize_t src = offset, rem = gid;
            for (int j = shape.size - 1; j >= 0; j--) {
                src += rem % shape.data[j] * strides.data[j];
                rem /= shape.data[j];
            }
            out[gid] = a[src];
        }
    }

    void Compact(const CudaArray &a,
                 CudaArray *out,
                 std::vector<int32_t> shape,
                 std::vector<int32_t> strides,
                 size_t offset) {
        /**
         * Compact an array in memory.  Unlike the C++ version, in CUDA this
         * will primarily call the relevant CUDA kernel.  In this case, we
         * illustrate how you should set this up (i.e., we give you the code for
         * this fuction, and also the prototype for the CompactKernel()
         * function).  For the functions after this, however, you'll need to
         * define these kernels as you see fit to execute the underlying
         * function.
         *
         * Args:
         *   a: non-compact represntation of the array, given as input
         *   out: compact version of the array to be written
         *   shape: shapes of each dimension for a and out
         *   strides: strides of the *a* array (not out, which has compact
         * strides) offset: offset of the *a* array (not out, which has zero
         * offset, being compact)
         */

        // Nothing needs to be added here
        CudaDims dim = CudaOneDim(out->size);
        CompactKernel<<<dim.grid, dim.block>>>(a.ptr,
                                               out->ptr,
                                               out->size,
                                               VecToCuda(shape),
                                               VecToCuda(strides),
                                               offset);
    }

    __global__ void EwiseSetitemKernel(const scalar_t *a,
                                       scalar_t *out,
                                       size_t size,
                                       CudaVec shape,
                                       CudaVec strides,
                                       size_t offset) {
        Py_ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < size) {
            Py_ssize_t dst = offset, rem = gid;
            for (int j = shape.size - 1; j >= 0; j--) {
                dst += rem % shape.data[j] * strides.data[j];
                rem /= shape.data[j];
            }
            out[dst] = a[gid];
        }
    }

    void EwiseSetitem(const CudaArray &a,
                      CudaArray *out,
                      std::vector<int32_t> shape,
                      std::vector<int32_t> strides,
                      size_t offset) {
        /**
         * Set items in a (non-compact) array using CUDA.  Yyou will most likely
         * want to implement a EwiseSetitemKernel() function, similar to those
         * above, that will do the actual work.
         *
         * Args:
         *   a: _compact_ array whose items will be written to out
         *   out: non-compact array whose items are to be written
         *   shape: shapes of each dimension for a and out
         *   strides: strides of the *out* array (not a, which has compact
         * strides) offset: offset of the *out* array (not a, which has zero
         * offset, being compact)
         */

        CudaDims dim = CudaOneDim(a.size);
        EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr,
                                                    out->ptr,
                                                    a.size,
                                                    VecToCuda(shape),
                                                    VecToCuda(strides),
                                                    offset);
    }

    __global__ void ScalarSetitemKernel(scalar_t val,
                                        scalar_t *out,
                                        size_t size,
                                        CudaVec shape,
                                        CudaVec strides,
                                        size_t offset) {
        Py_ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < size) {
            Py_ssize_t dst = offset, rem = gid;
            for (int j = shape.size - 1; j >= 0; j--) {
                dst += rem % shape.data[j] * strides.data[j];
                rem /= shape.data[j];
            }
            out[dst] = val;
        }
    }

    void ScalarSetitem(size_t size,
                       scalar_t val,
                       CudaArray *out,
                       std::vector<int32_t> shape,
                       std::vector<int32_t> strides,
                       size_t offset) {
        /**
         * Set items is a (non-compact) array
         *
         * Args:
         *   size: number of elements to write in out array (note that this will
         * note be the same as out.size, because out is a non-compact subset
         * array);  it _will_ be the same as the product of items in shape, but
         * covenient to just pass it here. val: scalar value to write to out:
         * non-compact array whose items are to be written shape: shapes of each
         * dimension of out strides: strides of the out array offset: offset of
         * the out array
         */

        CudaDims dim = CudaOneDim(size);
        ScalarSetitemKernel<<<dim.grid, dim.block>>>(
            val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar operations
    ////////////////////////////////////////////////////////////////////////////////

#define KERN_EWISE_UNARY(func, a, out, size, fn)                               \
    __global__ void func(const scalar_t *a, scalar_t *out, size_t size) {      \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                    \
        if (gid < size) out[gid] = fn(a[gid]);                                 \
    }
#define FUNC_EWISE_UNARY(func, a, out, kern)                                   \
    void func(const CudaArray &a, CudaArray *out) {                            \
        CudaDims dim = CudaOneDim(out->size);                                  \
        kern<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);             \
    }

#define KERN_EWISE_BINARY(func, a, b, out, size, fn)                           \
    __global__ void func(                                                      \
        const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size) {    \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                    \
        if (gid < size) out[gid] = fn(a[gid], b[gid]);                         \
    }
#define FUNC_EWISE_BINARY(func, a, b, out, kern)                               \
    void func(const CudaArray &a, const CudaArray &b, CudaArray *out) {        \
        CudaDims dim = CudaOneDim(out->size);                                  \
        kern<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);      \
    }

#define KERN_SCALAR(func, a, val, out, size, fn)                               \
    __global__ void func(                                                      \
        const scalar_t *a, scalar_t val, scalar_t *out, size_t size) {         \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                    \
        if (gid < size) out[gid] = fn(a[gid], val);                            \
    }

#define FUNC_SCALAR(func, a, val, out, kern)                                   \
    void func(const CudaArray &a, scalar_t val, CudaArray *out) {              \
        CudaDims dim = CudaOneDim(out->size);                                  \
        kern<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);        \
    }

    KERN_EWISE_BINARY(EwiseAddKernel, a, b, out, size, LAMBDA_ADD)
    FUNC_EWISE_BINARY(EwiseAdd, a, b, out, EwiseAddKernel)

    KERN_SCALAR(ScalarAddKernel, a, val, out, size, LAMBDA_ADD)
    FUNC_SCALAR(ScalarAdd, a, val, out, ScalarAddKernel)

    KERN_EWISE_BINARY(EwiseMulKernel, a, b, out, size, LAMBDA_MUL)
    FUNC_EWISE_BINARY(EwiseMul, a, b, out, EwiseMulKernel)

    KERN_SCALAR(ScalarMulKernel, a, val, out, size, LAMBDA_MUL)
    FUNC_SCALAR(ScalarMul, a, val, out, ScalarMulKernel)

    KERN_EWISE_BINARY(EwiseDivKernel, a, b, out, size, LAMBDA_DIV)
    FUNC_EWISE_BINARY(EwiseDiv, a, b, out, EwiseDivKernel)

    KERN_SCALAR(ScalarDivKernel, a, val, out, size, LAMBDA_DIV)
    FUNC_SCALAR(ScalarDiv, a, val, out, ScalarDivKernel)

    KERN_SCALAR(ScalarPowerKernel, a, val, out, size, LAMBDA_POW)
    FUNC_SCALAR(ScalarPower, a, val, out, ScalarPowerKernel)

    KERN_EWISE_BINARY(EwiseMaximumKernel, a, b, out, size, LAMBDA_MAX)
    FUNC_EWISE_BINARY(EwiseMaximum, a, b, out, EwiseMaximumKernel)

    KERN_SCALAR(ScalarMaximumKernel, a, val, out, size, LAMBDA_MAX)
    FUNC_SCALAR(ScalarMaximum, a, val, out, ScalarMaximumKernel)

    KERN_EWISE_BINARY(EwiseEqKernel, a, b, out, size, LAMBDA_EQ)
    FUNC_EWISE_BINARY(EwiseEq, a, b, out, EwiseEqKernel)

    KERN_SCALAR(ScalarEqKernel, a, val, out, size, LAMBDA_EQ)
    FUNC_SCALAR(ScalarEq, a, val, out, ScalarEqKernel)

    KERN_EWISE_BINARY(EwiseGeKernel, a, b, out, size, LAMBDA_GE)
    FUNC_EWISE_BINARY(EwiseGe, a, b, out, EwiseGeKernel)

    KERN_SCALAR(ScalarGeKernel, a, val, out, size, LAMBDA_GE)
    FUNC_SCALAR(ScalarGe, a, val, out, ScalarGeKernel)

    KERN_EWISE_UNARY(EwiseLogKernel, a, out, size, LAMBDA_LOG)
    FUNC_EWISE_UNARY(EwiseLog, a, out, EwiseLogKernel)

    KERN_EWISE_UNARY(EwiseExpKernel, a, out, size, LAMBDA_EXP)
    FUNC_EWISE_UNARY(EwiseExp, a, out, EwiseExpKernel)

    KERN_EWISE_UNARY(EwiseTanhKernel, a, out, size, LAMBDA_TANH)
    FUNC_EWISE_UNARY(EwiseTanh, a, out, EwiseTanhKernel)

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar operations
    ////////////////////////////////////////////////////////////////////////////////

#define ADDR_1D(cols, idx_row, idx_col) ((idx_row) * (cols) + (idx_col))
    __global__ void MatmulKernel(const scalar_t *a,
                                 const scalar_t *b,
                                 scalar_t *out,
                                 uint32_t M,
                                 uint32_t N,
                                 uint32_t P) {
        __shared__ scalar_t sa[STRIDE * TILE][STRIDE * TILE];
        __shared__ scalar_t sb[STRIDE * TILE][STRIDE * TILE];
        scalar_t sum[TILE][TILE] = {0};

        size_t row = ADDR_1D(blockDim.x, blockIdx.x, threadIdx.x);
        size_t col = ADDR_1D(blockDim.y, blockIdx.y, threadIdx.y);
        for (size_t i = 0; i < (N + STRIDE * TILE - 1) / (STRIDE * TILE); i++) {
            size_t col_a = ADDR_1D(STRIDE, i, threadIdx.y);
            size_t row_b = ADDR_1D(STRIDE, i, threadIdx.x);
            for (int j = 0; j < TILE; j++)
                for (int k = 0; k < TILE; k++) {
                    size_t dst_x = ADDR_1D(TILE, threadIdx.x, j);
                    size_t dst_y = ADDR_1D(TILE, threadIdx.y, k);

                    size_t a_x = ADDR_1D(TILE, row, j);
                    size_t a_y = ADDR_1D(TILE, col_a, k);
                    sa[dst_x][dst_y] =
                        a_x < M && a_y < N ? a[ADDR_1D(N, a_x, a_y)] : 0.;

                    size_t b_x = ADDR_1D(TILE, row_b, j);
                    size_t b_y = ADDR_1D(TILE, col, k);
                    sb[dst_y][dst_x] =
                        b_x < N && b_y < P ? b[ADDR_1D(P, b_x, b_y)] : 0.;
                }

            __syncthreads();

            for (int j = 0; j < TILE; j++)
                for (int k = 0; k < TILE; k++) {
                    scalar_t *aa = sa[ADDR_1D(TILE, threadIdx.x, j)];
                    scalar_t *bb = sb[ADDR_1D(TILE, threadIdx.y, k)];
                    for (int l = 0; l < STRIDE * TILE; l++)
                        sum[j][k] += aa[l] * bb[l];
                }
            __syncthreads();
        }

        for (int i = 0; i < TILE; i++)
            for (int j = 0; j < TILE; j++) {
                size_t x = ADDR_1D(TILE, row, i);
                size_t y = ADDR_1D(TILE, col, j);
                if (x < M && y < P) out[ADDR_1D(P, x, y)] = sum[i][j];
            }
    }

    void Matmul(const CudaArray &a,
                const CudaArray &b,
                CudaArray *out,
                uint32_t M,
                uint32_t N,
                uint32_t P) {
        /**
         * Multiply two (compact) matrices into an output (also comapct) matrix.
         * You will want to look at the lecture and notes on GPU-based linear
         * algebra to see how to do this.  Since ultimately mugrade is just
         * evaluating correctness, you _can_ implement a version that simply
         * parallelizes over (i,j) entries in the output array.  However, to
         * really get the full benefit of this problem, we would encourage you
         * to use cooperative fetching, shared memory register tiling, and other
         * ideas covered in the class notes.  Note that unlike the tiled matmul
         * function in the CPU backend, here you should implement a single
         * function that works across all size matrices, whether or not they are
         * a multiple of a tile size.  As with previous CUDA implementations,
         * this function here will largely just set up the kernel call, and you
         * should implement the logic in a separate MatmulKernel() call.
         *
         *
         * Args:
         *   a: compact 2D array of size m x n
         *   b: comapct 2D array of size n x p
         *   out: compact 2D array of size m x p to write the output to
         *   M: rows of a / out
         *   N: columns of a / rows of b
         *   P: columns of b / out
         */

        CudaDims dim = CudaTwoDim(M, P);
        MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Max and sum reductions
    ////////////////////////////////////////////////////////////////////////////////

#define KERN_REDUCE(func, a, out, size, reduce_size, init, fn)                 \
    __global__ void func(                                                      \
        const scalar_t *a, scalar_t *out, size_t size, size_t reduce_size) {   \
        size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                    \
        if (gid < size) {                                                      \
            out[gid] = thrust::reduce(thrust::device,                          \
                                      a + gid * reduce_size,                   \
                                      a + (gid + 1) * reduce_size,             \
                                      (scalar_t)init,                          \
                                      fn);                                     \
        }                                                                      \
    }

    KERN_REDUCE(ReduceMaxKernel, a, out, size, reduce_size, -inf, LAMBDA_MAX)

    KERN_REDUCE(ReduceSumKernel, a, out, size, reduce_size, 0, LAMBDA_ADD)

    void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size) {
        /**
         * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even
         * though it is inefficient, for simplicity you can perform each
         * reduction in a single CUDA thread.
         *
         * Args:
         *   a: compact array of size a.size = out.size * reduce_size to reduce
         * over out: compact array to write into redice_size: size of the
         * dimension to reduce over
         */

        CudaDims dim = CudaOneDim(out->size);
        ReduceMaxKernel<<<dim.grid, dim.block>>>(
            a.ptr, out->ptr, out->size, reduce_size);
    }

    void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size) {
        /**
         * Reduce by taking summation over `reduce_size` contiguous blocks.
         * Again, for simplicity you can perform each reduction in a single CUDA
         * thread.
         *
         * Args:
         *   a: compact array of size a.size = out.size * reduce_size to reduce
         * over out: compact array to write into redice_size: size of the
         * dimension to reduce over
         */

        CudaDims dim = CudaOneDim(out->size);
        ReduceSumKernel<<<dim.grid, dim.block>>>(
            a.ptr, out->ptr, out->size, reduce_size);
    }

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
    namespace py = pybind11;
    using namespace needle;
    using namespace cuda;

    m.attr("__device_name__") = "cuda";
    m.attr("__tile_size__") = TILE;

    py::class_<CudaArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def_readonly("size", &CudaArray::size)
        .def("ptr", &CudaArray::ptr_as_int);

    // return numpy array, copying from CPU
    m.def("to_numpy",
          [](const CudaArray &a,
             std::vector<size_t> shape,
             std::vector<size_t> strides,
             size_t offset) {
              std::vector<size_t> numpy_strides = strides;
              std::transform(numpy_strides.begin(),
                             numpy_strides.end(),
                             numpy_strides.begin(),
                             [](size_t &c) { return c * ELEM_SIZE; });

              // copy memory to host
              scalar_t *host_ptr = (scalar_t *)std::malloc(a.size * ELEM_SIZE);
              if (host_ptr == 0) throw std::bad_alloc();
              cudaError_t err = cudaMemcpy(
                  host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
              if (err != cudaSuccess)
                  throw std::runtime_error(cudaGetErrorString(err));

              // return numpy array
              py::capsule deallocate_buffer(host_ptr, [](void *p) { free(p); });
              return py::array_t<scalar_t>(
                  shape, numpy_strides, host_ptr + offset, deallocate_buffer);
          });

    // copy numpy array to GPU
    m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out) {
        cudaError_t err = cudaMemcpy(out->ptr,
                                     a.request().ptr,
                                     out->size * ELEM_SIZE,
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
    });

    m.def("fill", Fill);
    m.def("compact", Compact);
    m.def("ewise_setitem", EwiseSetitem);
    m.def("scalar_setitem", ScalarSetitem);
    m.def("ewise_add", EwiseAdd);
    m.def("scalar_add", ScalarAdd);

    m.def("ewise_mul", EwiseMul);
    m.def("scalar_mul", ScalarMul);
    m.def("ewise_div", EwiseDiv);
    m.def("scalar_div", ScalarDiv);
    m.def("scalar_power", ScalarPower);

    m.def("ewise_maximum", EwiseMaximum);
    m.def("scalar_maximum", ScalarMaximum);
    m.def("ewise_eq", EwiseEq);
    m.def("scalar_eq", ScalarEq);
    m.def("ewise_ge", EwiseGe);
    m.def("scalar_ge", ScalarGe);

    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);

    m.def("matmul", Matmul);

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}
