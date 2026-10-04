// Python bindings for the geometry core, via pybind11.
//
// This is the ONLY file that knows about Python. It checks array shapes, hands
// raw pointers to the header-only core in include/vggt_geom/geometry.hpp, and
// wraps the results as NumPy arrays. No geometry lives here.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "vggt_geom/geometry.hpp"

namespace py = pybind11;
namespace vg = vggt_geom;

// c_style: row-major, contiguous. forcecast: convert other dtypes or layouts instead of rejecting them.
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Int64Array = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using Uint8Array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument(message);
}

std::size_t points_count(const DoubleArray& points) {
    require(points.ndim() == 2 && points.shape(1) == 3, "points must have shape (N, 3)");
    return static_cast<std::size_t>(points.shape(0));
}

void require_4x4(const DoubleArray& t) { require(t.ndim() == 2 && t.shape(0) == 4 && t.shape(1) == 4, "transform must be 4x4"); }

template <typename T>
py::array_t<T> to_numpy(const std::vector<T>& values) {
    py::array_t<T> out(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

vg::OcclusionParams make_params(const std::string& mode, int radius, int half_width, double rel_tol, double abs_tol,
                                double min_depth) {
    vg::OcclusionParams params;
    if (mode == "none") params.mode = vg::OcclusionMode::None;
    else if (mode == "window") params.mode = vg::OcclusionMode::Window;
    else if (mode == "two_sided") params.mode = vg::OcclusionMode::TwoSided;
    else throw std::invalid_argument("mode must be 'none', 'window' or 'two_sided'");
    require(radius >= 0 && half_width >= 0, "radius and half_width must not be negative");
    params.radius = radius; params.half_width = half_width;
    params.rel_tol = rel_tol; params.abs_tol = abs_tol; params.min_depth = min_depth;
    return params;
}

void require_pixels(const Int64Array& ui, const Int64Array& vi, const DoubleArray& z, int width, int height) {
    require(ui.ndim() == 1 && vi.ndim() == 1 && z.ndim() == 1 && ui.size() == vi.size() && ui.size() == z.size(),
            "ui, vi and z must be 1-D and of equal length");
    require(width > 0 && height > 0, "width and height must be positive");
    for (py::ssize_t i = 0; i < ui.size(); ++i)
        require(ui.data()[i] >= 0 && ui.data()[i] < width && vi.data()[i] >= 0 && vi.data()[i] < height,
                "pixel index outside the image");
}

}  // namespace

PYBIND11_MODULE(vggt_geom_cpp, m) {
    m.doc() = "C++ geometry core. Mirrors vggt_aura.geometry and vggt_aura.ground_truth; Python stays the reference.";

    m.def("quat_xyzw_to_matrix", [](const DoubleArray& q) {
        require(q.ndim() == 1 && q.size() == 4, "quaternion must have 4 values (x, y, z, w)");
        py::array_t<double> out({3, 3});
        vg::quat_xyzw_to_matrix(q.data(), out.mutable_data());
        return out;
    });

    m.def("compose", [](const DoubleArray& a, const DoubleArray& b) {
        require_4x4(a); require_4x4(b);
        py::array_t<double> out({4, 4});
        vg::compose(a.data(), b.data(), out.mutable_data());
        return out;
    });

    m.def("invert_se3", [](const DoubleArray& t) {
        require_4x4(t);
        py::array_t<double> out({4, 4});
        vg::invert_se3(t.data(), out.mutable_data());
        return out;
    });

    m.def("transform_points", [](const DoubleArray& t, const DoubleArray& points) {
        require_4x4(t);
        const std::size_t n = points_count(points);
        py::array_t<double> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
        vg::transform_points(t.data(), points.data(), n, out.mutable_data());
        return out;
    });

    m.def("project_pinhole", [](const DoubleArray& points, double fx, double fy, double cx, double cy) {
        const std::size_t n = points_count(points);
        py::array_t<double> u(static_cast<py::ssize_t>(n)), v(static_cast<py::ssize_t>(n)), z(static_cast<py::ssize_t>(n));
        vg::project_pinhole(points.data(), n, fx, fy, cx, cy, u.mutable_data(), v.mutable_data(), z.mutable_data());
        return py::make_tuple(u, v, z);
    });

    m.def("unproject_pinhole", [](const DoubleArray& u, const DoubleArray& v, const DoubleArray& depth,
                                  double fx, double fy, double cx, double cy) {
        require(u.ndim() == 1 && v.ndim() == 1 && depth.ndim() == 1 && u.size() == v.size() && u.size() == depth.size(),
                "u, v and depth must be 1-D and of equal length");
        py::array_t<double> out({u.size(), static_cast<py::ssize_t>(3)});
        vg::unproject_pinhole(u.data(), v.data(), depth.data(), static_cast<std::size_t>(u.size()), fx, fy, cx, cy,
                              out.mutable_data());
        return out;
    });

    m.def("pixel_indices", [](const DoubleArray& u, const DoubleArray& v, int width, int height) {
        require(u.ndim() == 1 && v.ndim() == 1 && u.size() == v.size(), "u and v must be 1-D and of equal length");
        const py::ssize_t n = u.size();
        py::array_t<std::int64_t> ui(n), vi(n);
        py::array_t<std::uint8_t> inside(n);
        vg::pixel_indices(u.data(), v.data(), static_cast<std::size_t>(n), width, height, ui.mutable_data(),
                          vi.mutable_data(), inside.mutable_data());
        return py::make_tuple(ui, vi, inside);
    });

    m.def("occlusion_reason",
          [](const Int64Array& ui, const Int64Array& vi, const DoubleArray& z, int width, int height,
             const std::string& mode, int radius, int half_width, double rel_tol, double abs_tol) {
              require_pixels(ui, vi, z, width, height);
              py::array_t<std::uint8_t> reason(ui.size());
              vg::occlusion_reason(ui.data(), vi.data(), z.data(), static_cast<std::size_t>(ui.size()), width, height,
                                   make_params(mode, radius, half_width, rel_tol, abs_tol, 0.0), reason.mutable_data());
              return reason;
          },
          py::arg("ui"), py::arg("vi"), py::arg("z"), py::arg("width"), py::arg("height"), py::arg("mode") = "two_sided",
          py::arg("radius") = 6, py::arg("half_width") = 0, py::arg("rel_tol") = 0.10, py::arg("abs_tol") = 0.5);

    m.def("nearest_per_pixel", [](const Int64Array& ui, const Int64Array& vi, const DoubleArray& z, int width, int height) {
        require_pixels(ui, vi, z, width, height);
        return to_numpy(vg::nearest_per_pixel(ui.data(), vi.data(), z.data(), static_cast<std::size_t>(ui.size()), width, height));
    });

    m.def("lidar_to_image",
          [](const DoubleArray& points, const DoubleArray& camera_from_source, double fx, double fy, double cx, double cy,
             int width, int height, const std::string& mode, int radius, int half_width, double rel_tol, double abs_tol,
             double min_depth, const py::object& keep) {
              require_4x4(camera_from_source);
              require(width > 0 && height > 0, "width and height must be positive");
              const std::size_t n = points_count(points);
              const std::uint8_t* keep_data = nullptr;
              Uint8Array keep_array;
              if (!keep.is_none()) {
                  keep_array = Uint8Array::ensure(keep);
                  require(keep_array && keep_array.ndim() == 1 && static_cast<std::size_t>(keep_array.size()) == n,
                          "keep must be a 1-D array with one value per point");
                  keep_data = keep_array.data();
              }
              const vg::GroundTruth result = vg::lidar_to_image(points.data(), n, camera_from_source.data(), fx, fy, cx, cy, width,
                                                                height, make_params(mode, radius, half_width, rel_tol, abs_tol, min_depth),
                                                                keep_data);
              py::dict out;
              out["chosen"] = to_numpy(result.chosen);
              out["u"] = to_numpy(result.u);
              out["v"] = to_numpy(result.v);
              out["depth_m"] = to_numpy(result.depth);
              out["candidate_index"] = to_numpy(result.candidate);
              out["reason"] = to_numpy(result.reason);
              out["in_image_in_front"] = result.in_image_in_front;
              return out;
          },
          py::arg("points"), py::arg("camera_from_source"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("width"), py::arg("height"), py::arg("mode") = "two_sided", py::arg("radius") = 6,
          py::arg("half_width") = 0, py::arg("rel_tol") = 0.10, py::arg("abs_tol") = 0.5, py::arg("min_depth") = 1.0,
          py::arg("keep") = py::none());
}
