#include "rank_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <mutex>
#include <sstream>

namespace
{
constexpr int kThreadsPerBlock = 256;
constexpr int kKernelRadius = 2;
constexpr float kNeighborTotal = 25.0f;
constexpr float kRadiansToDegrees = 57.29577951308232f;

std::mutex gpu_mutex;
RankCudaPoint *device_points = nullptr;
float *device_ranges = nullptr;
float *device_ranks = nullptr;
int *device_rings = nullptr;
int *device_columns = nullptr;
std::uint8_t *device_valid = nullptr;
unsigned int *device_cell_range_bits = nullptr;
int *device_range_image = nullptr;
int *device_valid_count = nullptr;
float *device_rank_sum = nullptr;
int point_capacity = 0;
int image_capacity = 0;

std::string cuda_error(const char *operation, cudaError_t status)
{
    std::ostringstream oss;
    oss << operation << ": " << cudaGetErrorString(status);
    return oss.str();
}

void release_unlocked()
{
    cudaFree(device_points);
    cudaFree(device_ranges);
    cudaFree(device_ranks);
    cudaFree(device_rings);
    cudaFree(device_columns);
    cudaFree(device_valid);
    cudaFree(device_cell_range_bits);
    cudaFree(device_range_image);
    cudaFree(device_valid_count);
    cudaFree(device_rank_sum);
    device_points = nullptr;
    device_ranges = nullptr;
    device_ranks = nullptr;
    device_rings = nullptr;
    device_columns = nullptr;
    device_valid = nullptr;
    device_cell_range_bits = nullptr;
    device_range_image = nullptr;
    device_valid_count = nullptr;
    device_rank_sum = nullptr;
    point_capacity = 0;
    image_capacity = 0;
}

bool allocate_point_buffers(int capacity, std::string *error)
{
    cudaError_t status = cudaMalloc(&device_points, sizeof(RankCudaPoint) * capacity);
    if (status == cudaSuccess) status = cudaMalloc(&device_ranges, sizeof(float) * capacity);
    if (status == cudaSuccess) status = cudaMalloc(&device_ranks, sizeof(float) * capacity);
    if (status == cudaSuccess) status = cudaMalloc(&device_rings, sizeof(int) * capacity);
    if (status == cudaSuccess) status = cudaMalloc(&device_columns, sizeof(int) * capacity);
    if (status == cudaSuccess) status = cudaMalloc(&device_valid, sizeof(std::uint8_t) * capacity);
    if (status == cudaSuccess && !device_valid_count) status = cudaMalloc(&device_valid_count, sizeof(int));
    if (status == cudaSuccess && !device_rank_sum) status = cudaMalloc(&device_rank_sum, sizeof(float));
    if (status != cudaSuccess)
    {
        if (error) *error = cuda_error("allocating point buffers", status);
        return false;
    }
    return true;
}

bool ensure_capacity(int point_count, int image_size, std::string *error)
{
    if (point_count > point_capacity)
    {
        const int new_capacity = std::max(point_count, point_capacity * 2);
        cudaFree(device_points);
        cudaFree(device_ranges);
        cudaFree(device_ranks);
        cudaFree(device_rings);
        cudaFree(device_columns);
        cudaFree(device_valid);
        device_points = nullptr;
        device_ranges = nullptr;
        device_ranks = nullptr;
        device_rings = nullptr;
        device_columns = nullptr;
        device_valid = nullptr;
        if (!allocate_point_buffers(new_capacity, error))
        {
            release_unlocked();
            return false;
        }
        point_capacity = new_capacity;
    }

    if (image_size > image_capacity)
    {
        const int new_capacity = std::max(image_size, image_capacity * 2);
        cudaFree(device_cell_range_bits);
        cudaFree(device_range_image);
        device_cell_range_bits = nullptr;
        device_range_image = nullptr;
        cudaError_t status = cudaMalloc(&device_cell_range_bits, sizeof(unsigned int) * new_capacity);
        if (status == cudaSuccess) status = cudaMalloc(&device_range_image, sizeof(int) * new_capacity);
        if (status != cudaSuccess)
        {
            if (error) *error = cuda_error("allocating range image", status);
            release_unlocked();
            return false;
        }
        image_capacity = new_capacity;
    }
    return true;
}

__global__ void initialize_buffers_kernel(int point_count,
                                          int image_size,
                                          float *ranks,
                                          int *rings,
                                          int *columns,
                                          std::uint8_t *valid,
                                          unsigned int *cell_range_bits,
                                          int *range_image)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < point_count)
    {
        ranks[index] = 0.0f;
        rings[index] = -1;
        columns[index] = -1;
        valid[index] = 0;
    }
    if (index < image_size)
    {
        cell_range_bits[index] = UINT_MAX;
        range_image[index] = INT_MAX;
    }
}

__global__ void preprocess_points_kernel(const RankCudaPoint *points,
                                         int point_count,
                                         int scan_lines,
                                         int columns,
                                         float horizontal_resolution_deg,
                                         float blind_range,
                                         float max_range,
                                         float fov_up_deg,
                                         float fov_down_deg,
                                         int ring_source_mode,
                                         bool pointcloud_ring_available,
                                         float *ranges,
                                         int *rings,
                                         int *column_indices,
                                         std::uint8_t *valid,
                                         unsigned int *cell_range_bits,
                                         int *valid_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= point_count)
        return;

    const RankCudaPoint point = points[index];
    const float range = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
    ranges[index] = range;
    if (!isfinite(range) || range <= blind_range || range > max_range)
        return;

    int ring = -1;
    if (ring_source_mode == 2)
    {
        const float fov_total = fov_up_deg - fov_down_deg;
        if (fov_total <= 1e-6f || scan_lines <= 1 || range <= 1e-6f)
            return;
        const float elevation = asinf(point.z / range) * kRadiansToDegrees;
        ring = __float2int_rn((fov_up_deg - elevation) / fov_total * static_cast<float>(scan_lines - 1));
        if (ring < 0 || ring >= scan_lines)
            return;
    }
    else
    {
        if (ring_source_mode == 1 && !pointcloud_ring_available)
            return;
        ring = __float2int_rn(point.preserved_ring);
        if (!isfinite(point.preserved_ring) || ring < 0 || ring >= scan_lines)
            return;
    }

    float azimuth = atan2f(point.y, point.x) * kRadiansToDegrees;
    if (azimuth < 0.0f) azimuth += 360.0f;
    int column = __float2int_rn(azimuth / horizontal_resolution_deg) % columns;
    if (column < 0) column += columns;

    rings[index] = ring;
    column_indices[index] = column;
    valid[index] = 1;
    atomicAdd(valid_count, 1);
    atomicMin(&cell_range_bits[ring * columns + column], __float_as_uint(range));
}

__global__ void select_range_image_kernel(const float *ranges,
                                          const int *rings,
                                          const int *column_indices,
                                          const std::uint8_t *valid,
                                          const unsigned int *cell_range_bits,
                                          int point_count,
                                          int columns,
                                          int *range_image)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= point_count || !valid[index])
        return;

    const int cell = rings[index] * columns + column_indices[index];
    if (__float_as_uint(ranges[index]) == cell_range_bits[cell])
        atomicMin(&range_image[cell], index);
}

__global__ void compute_rank_kernel(const float *ranges,
                                    const int *rings,
                                    const int *column_indices,
                                    const std::uint8_t *valid,
                                    const int *range_image,
                                    int point_count,
                                    int scan_lines,
                                    int columns,
                                    float inverse_two_sigma_squared,
                                    float inverse_max_range,
                                    int range_factor_mode,
                                    float *ranks,
                                    float *rank_sum)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= point_count || !valid[index])
        return;

    const int ring = rings[index];
    const int column = column_indices[index];
    float consistency_sum = 0.0f;
    for (int du = -kKernelRadius; du <= kKernelRadius; ++du)
    {
        const int neighbor_ring = ring + du;
        if (neighbor_ring < 0 || neighbor_ring >= scan_lines)
            continue;

        for (int dv = -kKernelRadius; dv <= kKernelRadius; ++dv)
        {
            int neighbor_column = column + dv;
            if (neighbor_column < 0) neighbor_column += columns;
            if (neighbor_column >= columns) neighbor_column -= columns;

            const int neighbor_index = range_image[neighbor_ring * columns + neighbor_column];
            if (neighbor_index >= point_count)
                continue;

            const float delta = ranges[index] - ranges[neighbor_index];
            consistency_sum += expf(-(delta * delta) * inverse_two_sigma_squared);
        }
    }

    const float consistency = consistency_sum / kNeighborTotal;
    const float normalized_range = ranges[index] * inverse_max_range;
    float range_factor = 1.0f;
    if (range_factor_mode == 0)
        range_factor += normalized_range;
    else if (range_factor_mode == 2)
        range_factor = 1.0f / (1.0f + normalized_range);
    const float rank = (1.0f + consistency) * range_factor;
    ranks[index] = rank;
    atomicAdd(rank_sum, fmaxf(rank, 1e-6f));
}
}  // namespace

bool rank_cuda_is_available(std::string *reason)
{
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess)
    {
        if (reason) *reason = cuda_error("cudaGetDeviceCount", status);
        return false;
    }
    if (device_count <= 0)
    {
        if (reason) *reason = "no CUDA device found";
        return false;
    }

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, 0);
    if (status != cudaSuccess)
    {
        if (reason) *reason = cuda_error("cudaGetDeviceProperties", status);
        return false;
    }
    if (reason) *reason = properties.name;
    return true;
}

bool rank_cuda_compute(const RankCudaPoint *points,
                       int point_count,
                       int scan_lines,
                       int columns,
                       float horizontal_resolution_deg,
                       float sigma_range,
                       float blind_range,
                       float max_range,
                       float fov_up_deg,
                       float fov_down_deg,
                       int ring_source_mode,
                       bool pointcloud_ring_available,
                       int range_factor_mode,
                       float *ranks,
                       std::uint8_t *valid,
                       int *valid_count,
                       float *rank_sum,
                       std::string *error)
{
    if (!points || !ranks || !valid || !valid_count || !rank_sum || point_count <= 0 ||
        scan_lines <= 0 || columns <= 0 || horizontal_resolution_deg <= 0.0f ||
        sigma_range <= 0.0f || max_range <= 0.0f)
    {
        if (error) *error = "invalid CUDA rank input";
        return false;
    }

    std::lock_guard<std::mutex> lock(gpu_mutex);
    const int image_size = scan_lines * columns;
    if (!ensure_capacity(point_count, image_size, error))
        return false;

    cudaError_t status = cudaMemcpy(device_points,
                                    points,
                                    sizeof(RankCudaPoint) * point_count,
                                    cudaMemcpyHostToDevice);
    if (status == cudaSuccess) status = cudaMemset(device_valid_count, 0, sizeof(int));
    if (status == cudaSuccess) status = cudaMemset(device_rank_sum, 0, sizeof(float));
    if (status != cudaSuccess)
    {
        if (error) *error = cuda_error("copying rank points", status);
        return false;
    }

    const int work_size = std::max(point_count, image_size);
    const int work_blocks = (work_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int point_blocks = (point_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    initialize_buffers_kernel<<<work_blocks, kThreadsPerBlock>>>(point_count,
                                                                 image_size,
                                                                 device_ranks,
                                                                 device_rings,
                                                                 device_columns,
                                                                 device_valid,
                                                                 device_cell_range_bits,
                                                                 device_range_image);
    preprocess_points_kernel<<<point_blocks, kThreadsPerBlock>>>(device_points,
                                                                 point_count,
                                                                 scan_lines,
                                                                 columns,
                                                                 horizontal_resolution_deg,
                                                                 blind_range,
                                                                 max_range,
                                                                 fov_up_deg,
                                                                 fov_down_deg,
                                                                 ring_source_mode,
                                                                 pointcloud_ring_available,
                                                                 device_ranges,
                                                                 device_rings,
                                                                 device_columns,
                                                                 device_valid,
                                                                 device_cell_range_bits,
                                                                 device_valid_count);
    select_range_image_kernel<<<point_blocks, kThreadsPerBlock>>>(device_ranges,
                                                                  device_rings,
                                                                  device_columns,
                                                                  device_valid,
                                                                  device_cell_range_bits,
                                                                  point_count,
                                                                  columns,
                                                                  device_range_image);
    compute_rank_kernel<<<point_blocks, kThreadsPerBlock>>>(device_ranges,
                                                            device_rings,
                                                            device_columns,
                                                            device_valid,
                                                            device_range_image,
                                                            point_count,
                                                            scan_lines,
                                                            columns,
                                                            1.0f / (2.0f * sigma_range * sigma_range),
                                                            1.0f / max_range,
                                                            range_factor_mode,
                                                            device_ranks,
                                                            device_rank_sum);
    status = cudaGetLastError();
    if (status == cudaSuccess)
        status = cudaMemcpy(ranks, device_ranks, sizeof(float) * point_count, cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
        status = cudaMemcpy(valid, device_valid, sizeof(std::uint8_t) * point_count, cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
        status = cudaMemcpy(valid_count, device_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
        status = cudaMemcpy(rank_sum, device_rank_sum, sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        if (error) *error = cuda_error("running full rank pipeline", status);
        return false;
    }

    if (error) error->clear();
    return true;
}

void rank_cuda_release()
{
    std::lock_guard<std::mutex> lock(gpu_mutex);
    release_unlocked();
}
