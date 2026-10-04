// vggt_geom: the geometry core, in plain C++17. Header-only, no dependencies.
//
// SCOPE (fixed by the project plan): projection, unprojection, rigid transform
// composition, quaternion to rotation, and LiDAR-to-image projection with the
// occlusion check. Numbers in, numbers out. This file never opens a file,
// never parses a point cloud, never touches a GPU.
//
// It mirrors the Python reference function for function:
//   src/vggt_aura/geometry.py      quat_xyzw_to_matrix, invert_se3, composition
//   src/vggt_aura/ground_truth.py  project_pinhole, unproject_pinhole, pixel_indices,
//                                  min_depth_buffer, directional_minima,
//                                  occlusion_reason, nearest_per_pixel, build_ground_truth
// Python stays the reference. If the two ever disagree, this file is wrong.
//
// CONVENTIONS (identical to the Python side):
//   - A 4x4 transform is stored ROW-MAJOR and named destination_from_source:
//       p_destination = T * p_source
//   - Camera frame: X right, Y down, Z forward. Depth means Z.
//   - Quaternions are (x, y, z, w).
//   - A pixel's centre sits on its integer coordinate; pixel = floor(u + 0.5).
//   - Images are indexed [v][u], v down, stored row-major as v * width + u.
//
// To stay bit-compatible with NumPy, build with -ffp-contract=off so the
// compiler does not fuse a*b+c into one rounded operation.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vggt_geom {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------- rotations and rigid transforms

// out: 3x3 row-major. The quaternion is normalised first.
inline void quat_xyzw_to_matrix(const double* q, double* out) {
    const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(norm > 0.0)) throw std::invalid_argument("zero quaternion");
    const double x = q[0] / norm, y = q[1] / norm, z = q[2] / norm, w = q[3] / norm;
    out[0] = 1 - 2 * (y * y + z * z);  out[1] = 2 * (x * y - z * w);      out[2] = 2 * (x * z + y * w);
    out[3] = 2 * (x * y + z * w);      out[4] = 1 - 2 * (x * x + z * z);  out[5] = 2 * (y * z - x * w);
    out[6] = 2 * (x * z - y * w);      out[7] = 2 * (y * z + x * w);      out[8] = 1 - 2 * (x * x + y * y);
}

// out = a * b. With a = c_from_b and b = b_from_a this gives c_from_a: read right to left.
inline void compose(const double* a, const double* b, double* out) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) sum += a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = sum;
        }
}

// Closed-form inverse of a rigid transform: [R t] -> [R^T  -R^T t].
inline void invert_se3(const double* t, double* out) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out[r * 4 + c] = t[c * 4 + r];
    for (int r = 0; r < 3; ++r) {
        double sum = 0.0;
        for (int k = 0; k < 3; ++k) sum += out[r * 4 + k] * t[k * 4 + 3];
        out[r * 4 + 3] = -sum;
    }
    out[12] = 0.0; out[13] = 0.0; out[14] = 0.0; out[15] = 1.0;
}

// points, out: n x 3 row-major. out = R * p + t for every point.
inline void transform_points(const double* t, const double* points, std::size_t n, double* out) {
    for (std::size_t i = 0; i < n; ++i) {
        const double x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
        out[i * 3]     = t[0] * x + t[1] * y + t[2] * z + t[3];
        out[i * 3 + 1] = t[4] * x + t[5] * y + t[6] * z + t[7];
        out[i * 3 + 2] = t[8] * x + t[9] * y + t[10] * z + t[11];
    }
}

// ---------------------------------------------------------------- projection

// Camera-frame points to pixel coordinates and Z-depth. u and v are NaN where z <= 1e-9.
inline void project_pinhole(const double* points, std::size_t n, double fx, double fy, double cx, double cy,
                            double* u, double* v, double* z) {
    for (std::size_t i = 0; i < n; ++i) {
        const double depth = points[i * 3 + 2];
        z[i] = depth;
        if (depth > 1e-9) {
            u[i] = fx * points[i * 3] / depth + cx;
            v[i] = fy * points[i * 3 + 1] / depth + cy;
        } else {
            u[i] = kNaN;
            v[i] = kNaN;
        }
    }
}

// The inverse: pixels plus Z-depth back to camera-frame points (n x 3).
inline void unproject_pinhole(const double* u, const double* v, const double* depth, std::size_t n,
                              double fx, double fy, double cx, double cy, double* out) {
    for (std::size_t i = 0; i < n; ++i) {
        out[i * 3]     = (u[i] - cx) / fx * depth[i];
        out[i * 3 + 1] = (v[i] - cy) / fy * depth[i];
        out[i * 3 + 2] = depth[i];
    }
}

// Nearest pixel per point, and whether it falls inside the image. Outside points get index 0.
inline void pixel_indices(const double* u, const double* v, std::size_t n, int width, int height,
                          std::int64_t* ui, std::int64_t* vi, std::uint8_t* inside) {
    for (std::size_t i = 0; i < n; ++i) {
        const double fu = std::floor(u[i] + 0.5), fv = std::floor(v[i] + 0.5);
        const bool ok = std::isfinite(fu) && std::isfinite(fv) && fu >= 0 && fu < width && fv >= 0 && fv < height;
        inside[i] = ok ? 1 : 0;
        ui[i] = ok ? static_cast<std::int64_t>(fu) : 0;
        vi[i] = ok ? static_cast<std::int64_t>(fv) : 0;
    }
}

// ---------------------------------------------------------------- occlusion

enum class OcclusionMode : int { None = 0, Window = 1, TwoSided = 2 };
constexpr std::uint8_t kVisible = 0, kHiddenSamePixel = 1, kHiddenByNeighbours = 2;

struct OcclusionParams {
    OcclusionMode mode = OcclusionMode::TwoSided;
    int radius = 6;
    int half_width = 0;      // keep 0: wider search lines delete roads (see ground_truth.py)
    double rel_tol = 0.10;
    double abs_tol = 0.5;
    double min_depth = 1.0;
};

// Per-pixel nearest depth; +inf where empty. Indices must already be inside the image.
inline std::vector<double> min_depth_buffer(const std::int64_t* ui, const std::int64_t* vi, const double* z,
                                            std::size_t n, int width, int height) {
    std::vector<double> buffer(static_cast<std::size_t>(width) * height, kInf);
    for (std::size_t i = 0; i < n; ++i) {
        double& cell = buffer[static_cast<std::size_t>(vi[i]) * width + ui[i]];
        if (z[i] < cell) cell = z[i];
    }
    return buffer;
}

struct Sides { std::vector<double> above, below, left, right; };

// Nearest depth found in a strip on each side of every pixel; the pixel itself is never included.
// "above" covers rows v-radius .. v-1 and columns u-half_width .. u+half_width; the others are that strip rotated.
inline Sides directional_minima(const std::vector<double>& buffer, int width, int height, int radius, int half_width) {
    const std::size_t size = buffer.size();
    std::vector<double> across_u(buffer), across_v(buffer);
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            double mu = buffer[static_cast<std::size_t>(row) * width + col], mv = mu;
            for (int k = 1; k <= half_width; ++k) {
                if (col - k >= 0)     mu = std::fmin(mu, buffer[static_cast<std::size_t>(row) * width + (col - k)]);
                if (col + k < width)  mu = std::fmin(mu, buffer[static_cast<std::size_t>(row) * width + (col + k)]);
                if (row - k >= 0)     mv = std::fmin(mv, buffer[static_cast<std::size_t>(row - k) * width + col]);
                if (row + k < height) mv = std::fmin(mv, buffer[static_cast<std::size_t>(row + k) * width + col]);
            }
            across_u[static_cast<std::size_t>(row) * width + col] = mu;
            across_v[static_cast<std::size_t>(row) * width + col] = mv;
        }
    Sides sides{std::vector<double>(size, kInf), std::vector<double>(size, kInf),
                std::vector<double>(size, kInf), std::vector<double>(size, kInf)};
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col) {
            const std::size_t here = static_cast<std::size_t>(row) * width + col;
            double above = kInf, below = kInf, left = kInf, right = kInf;
            for (int k = 1; k <= radius; ++k) {
                if (row - k >= 0)     above = std::fmin(above, across_u[static_cast<std::size_t>(row - k) * width + col]);
                if (row + k < height) below = std::fmin(below, across_u[static_cast<std::size_t>(row + k) * width + col]);
                if (col - k >= 0)     left  = std::fmin(left,  across_v[here - static_cast<std::size_t>(k)]);
                if (col + k < width)  right = std::fmin(right, across_v[here + static_cast<std::size_t>(k)]);
            }
            sides.above[here] = above; sides.below[here] = below; sides.left[here] = left; sides.right[here] = right;
        }
    return sides;
}

// Per point: kVisible, kHiddenSamePixel or kHiddenByNeighbours. "Much nearer" = nearer by more than
// max(abs_tol, rel_tol * depth). Same-pixel takes precedence, as in the Python reference.
inline void occlusion_reason(const std::int64_t* ui, const std::int64_t* vi, const double* z, std::size_t n,
                             int width, int height, const OcclusionParams& params, std::uint8_t* reason) {
    for (std::size_t i = 0; i < n; ++i) reason[i] = kVisible;
    if (params.mode == OcclusionMode::None || n == 0) return;
    const std::vector<double> buffer = min_depth_buffer(ui, vi, z, n, width, height);

    if (params.half_width == 0) {
        // FAST PATH, the default setting. With search lines one pixel wide, "the nearest depth on a side is
        // below the threshold" is the same statement as "SOME pixel on that side is below the threshold".
        // So each point scans its own column and row directly and stops at the first hit. This touches only
        // pixels near actual points, and builds none of the four full-image side arrays. It was added after
        // the general path measured SLOWER than NumPy on real frames (51 ms against 31 ms).
        const std::int64_t w = width, h = height, r = params.radius;
        for (std::size_t i = 0; i < n; ++i) {
            const std::int64_t u = ui[i], v = vi[i];
            const std::size_t here = static_cast<std::size_t>(v * w + u);
            const double scaled = params.rel_tol * z[i];
            const double threshold = z[i] - (params.abs_tol > scaled ? params.abs_tol : scaled);
            bool above = false, below = false, left = false, right = false;
            for (std::int64_t k = 1; k <= r && !above; ++k) if (v - k >= 0) above = buffer[static_cast<std::size_t>((v - k) * w + u)] < threshold;
            for (std::int64_t k = 1; k <= r && !below; ++k) if (v + k < h)  below = buffer[static_cast<std::size_t>((v + k) * w + u)] < threshold;
            for (std::int64_t k = 1; k <= r && !left; ++k)  if (u - k >= 0) left  = buffer[here - static_cast<std::size_t>(k)] < threshold;
            for (std::int64_t k = 1; k <= r && !right; ++k) if (u + k < w)  right = buffer[here + static_cast<std::size_t>(k)] < threshold;
            const bool by_neighbours = params.mode == OcclusionMode::Window ? (above || below || left || right)
                                                                           : ((above && below) || (left && right));
            if (by_neighbours) reason[i] = kHiddenByNeighbours;
            if (buffer[here] < threshold) reason[i] = kHiddenSamePixel;
        }
        return;
    }

    // General path, for search strips wider than one pixel.
    const Sides sides = directional_minima(buffer, width, height, params.radius, params.half_width);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t here = static_cast<std::size_t>(vi[i]) * width + ui[i];
        const double scaled = params.rel_tol * z[i];
        const double threshold = z[i] - (params.abs_tol > scaled ? params.abs_tol : scaled);
        const bool above = sides.above[here] < threshold, below = sides.below[here] < threshold;
        const bool left = sides.left[here] < threshold, right = sides.right[here] < threshold;
        const bool by_neighbours = params.mode == OcclusionMode::Window ? (above || below || left || right)
                                                                       : ((above && below) || (left && right));
        if (by_neighbours) reason[i] = kHiddenByNeighbours;
        if (buffer[here] < threshold) reason[i] = kHiddenSamePixel;
    }
}

// Aggregation rule: the nearest point of each occupied pixel. Returns point indices in ascending order.
// On an exact depth tie the lower index wins, as NumPy's stable lexsort does.
inline std::vector<std::int64_t> nearest_per_pixel(const std::int64_t* ui, const std::int64_t* vi, const double* z,
                                                   std::size_t n, int width, int height) {
    std::vector<std::int64_t> best(static_cast<std::size_t>(width) * height, -1);
    for (std::size_t i = 0; i < n; ++i) {
        std::int64_t& slot = best[static_cast<std::size_t>(vi[i]) * width + ui[i]];
        if (slot < 0 || z[i] < z[static_cast<std::size_t>(slot)]) slot = static_cast<std::int64_t>(i);
    }
    std::vector<std::uint8_t> chosen(n, 0);
    for (std::int64_t index : best) if (index >= 0) chosen[static_cast<std::size_t>(index)] = 1;
    std::vector<std::int64_t> out;
    for (std::size_t i = 0; i < n; ++i) if (chosen[i]) out.push_back(static_cast<std::int64_t>(i));
    return out;
}

// ---------------------------------------------------------------- the whole LiDAR-to-image step

struct GroundTruth {
    std::vector<std::int64_t> chosen;      // indices into the input points: one per pixel with ground truth
    std::vector<std::int64_t> u, v;        // pixel of each chosen point
    std::vector<double> depth;             // its Z-depth in metres
    std::vector<std::int64_t> candidate;   // indices of every point that was tested for occlusion
    std::vector<std::uint8_t> reason;      // occlusion verdict for each candidate
    std::int64_t in_image_in_front = 0;    // points inside the image and beyond min_depth, BEFORE the keep filter
};

// Mirrors ground_truth.build_ground_truth, with the transform included:
//   points (any frame) -> camera frame -> pixels -> drop (outside, too near, keep == 0) -> occlusion -> nearest per pixel.
inline GroundTruth lidar_to_image(const double* points, std::size_t n, const double* camera_from_source,
                                  double fx, double fy, double cx, double cy, int width, int height,
                                  const OcclusionParams& params, const std::uint8_t* keep /* may be null */) {
    std::vector<double> cam(n * 3), u(n), v(n), z(n);
    transform_points(camera_from_source, points, n, cam.data());
    project_pinhole(cam.data(), n, fx, fy, cx, cy, u.data(), v.data(), z.data());
    std::vector<std::int64_t> ui(n), vi(n);
    std::vector<std::uint8_t> inside(n);
    pixel_indices(u.data(), v.data(), n, width, height, ui.data(), vi.data(), inside.data());

    GroundTruth out;
    std::vector<std::int64_t> cu, cv;
    std::vector<double> cz;
    for (std::size_t i = 0; i < n; ++i) {
        if (!(inside[i] && z[i] >= params.min_depth)) continue;
        ++out.in_image_in_front;
        if (keep == nullptr || keep[i]) {
            out.candidate.push_back(static_cast<std::int64_t>(i));
            cu.push_back(ui[i]); cv.push_back(vi[i]); cz.push_back(z[i]);
        }
    }
    const std::size_t m = out.candidate.size();
    out.reason.resize(m);
    occlusion_reason(cu.data(), cv.data(), cz.data(), m, width, height, params, out.reason.data());

    std::vector<std::int64_t> su, sv, seen;
    std::vector<double> sz;
    for (std::size_t j = 0; j < m; ++j)
        if (out.reason[j] == kVisible) {
            seen.push_back(out.candidate[j]); su.push_back(cu[j]); sv.push_back(cv[j]); sz.push_back(cz[j]);
        }
    for (std::int64_t j : nearest_per_pixel(su.data(), sv.data(), sz.data(), seen.size(), width, height)) {
        const std::size_t k = static_cast<std::size_t>(j);
        out.chosen.push_back(seen[k]); out.u.push_back(su[k]); out.v.push_back(sv[k]); out.depth.push_back(sz[k]);
    }
    return out;
}

}  // namespace vggt_geom
