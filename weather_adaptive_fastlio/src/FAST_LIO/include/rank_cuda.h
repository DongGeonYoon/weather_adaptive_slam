#pragma once

#include <cstdint>
#include <string>

struct RankCudaPoint
{
    float x;
    float y;
    float z;
    float preserved_ring;
};

bool rank_cuda_is_available(std::string *reason = nullptr);

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
                       std::string *error = nullptr);

void rank_cuda_release();
