#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if IMAGE_TYPE == 4
using Real = double;
constexpr Real EPS = 0x1p-52;
#else
using Real = float;
constexpr Real EPS = 0x1p-23f;
#endif
#if IMAGE_TYPE == 0
using Input = unsigned char;
#elif IMAGE_TYPE == 1
using Input = __half;
#elif IMAGE_TYPE == 2
using Input = __nv_bfloat16;
#elif IMAGE_TYPE == 3
using Input = float;
#else
using Input = double;
#endif
#if IMAGE_TYPE == 0 && BILINEAR
using Output = float;
#else
using Output = Input;
#endif

struct Camera {
    Real fx, fy, cx, cy;
    Real k1, k2, p1, p2, k3, k4, k5, k6;
    Real max_angle, circle_x, circle_y, circle_radius;
    int height, width;
};

#define CAMERA_ARGS Real fx, Real fy, Real cx, Real cy, \
    Real k1, Real k2, Real p1, Real p2, Real k3, Real k4, Real k5, Real k6, \
    Real max_angle, Real circle_x, Real circle_y, Real circle_radius, int height, int width
#define CAMERA_INIT Camera camera{fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, \
    max_angle, circle_x, circle_y, circle_radius, height, width}

__device__ bool pixel_valid(Real x, Real y, const Camera& c) {
    bool valid = isfinite(x) && isfinite(y) && x >= Real(-0.5) && y >= Real(-0.5)
        && x < c.width - Real(0.5) && y < c.height - Real(0.5);
    if (c.circle_radius > 0) {
        Real dx = x - c.circle_x, dy = y - c.circle_y;
        valid = valid && dx * dx + dy * dy <= c.circle_radius * c.circle_radius;
    }
    return valid;
}

__device__ void distort(Real x, Real y, const Camera& c, Real& u, Real& v) {
#if DISTORTED
    Real r2 = x * x + y * y;
    Real radial = (1 + r2 * (c.k1 + r2 * (c.k2 + r2 * c.k3))) /
                  (1 + r2 * (c.k4 + r2 * (c.k5 + r2 * c.k6)));
    u = x * radial + 2 * c.p1 * x * y + c.p2 * (r2 + 2 * x * x);
    v = y * radial + c.p1 * (r2 + 2 * y * y) + 2 * c.p2 * x * y;
#else
    u = x; v = y;
#endif
}

__device__ Real fisheye_radius(Real theta, const Camera& c) {
    Real t2 = theta * theta;
    // Fisheye's four angular coefficients occupy k1, k2, p1, p2.
    return theta * (1 + t2 * (c.k1 + t2 * (c.k2 + t2 * (c.p1 + t2 * c.p2))));
}

__device__ bool project(Real x, Real y, Real z, const Camera& c, Real& u, Real& v) {
    Real lateral = sqrt(x * x + y * y);
    Real theta = atan2(lateral, z);
#if FISHEYE
    Real scale = lateral > EPS ? fisheye_radius(theta, c) / fmax(lateral, EPS) : 1 / fmax(z, EPS);
    u = x * scale; v = y * scale;
#else
    distort(x / fmax(z, EPS), y / fmax(z, EPS), c, u, v);
#endif
    u = u * c.fx + c.cx; v = v * c.fy + c.cy;
    return pixel_valid(u, v, c) && theta <= c.max_angle && z > 0;
}

extern "C" __global__ void prepare_gaze(
    const Real* fixation, Real* gaze, long long stride_b, long long stride_xy,
    int batch, CAMERA_ARGS) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    CAMERA_INIT;
    Real pixel_x = fixation[b * stride_b + stride_xy] * width - Real(0.5);
    Real pixel_y = fixation[b * stride_b] * height - Real(0.5);
    // ATen's tensor/scalar division multiplies by the scalar reciprocal.
    Real tx = (pixel_x - cx) * (Real(1) / fx), ty = (pixel_y - cy) * (Real(1) / fy);
    Real x = tx, y = ty, z = 1;
#if FISHEYE
    Real radius = sqrt(tx * tx + ty * ty), theta = radius;
#if DISTORTED
    for (int i = 0; i < 12; ++i) {
        Real t2 = theta * theta;
        Real derivative = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * p1 + t2 * 9 * p2)));
        theta -= (fisheye_radius(theta, camera) - radius) / fmax(derivative, EPS);
    }
#endif
    Real scale = radius > EPS ? sin(theta) / fmax(radius, EPS) : 1;
    x = tx * scale; y = ty * scale; z = cos(theta);
#else
#if DISTORTED
    for (int i = 0; i < 12; ++i) {
        Real r2 = x * x + y * y;
        Real numerator = 1 + r2 * (k1 + r2 * (k2 + r2 * k3));
        Real denominator = 1 + r2 * (k4 + r2 * (k5 + r2 * k6));
        Real radial = numerator / denominator;
        Real derivative = ((k1 + r2 * (2 * k2 + r2 * 3 * k3)) * denominator
            - numerator * (k4 + r2 * (2 * k5 + r2 * 3 * k6))) / (denominator * denominator);
        Real jxx = radial + 2 * x * x * derivative + 2 * p1 * y + 6 * p2 * x;
        Real jyy = radial + 2 * y * y * derivative + 6 * p1 * y + 2 * p2 * x;
        Real jxy = 2 * x * y * derivative + 2 * p1 * x + 2 * p2 * y;
        Real determinant = jxx * jyy - jxy * jxy;
        if (!(fabs(determinant) > EPS)) determinant = EPS;
        Real dx, dy;
        distort(x, y, camera, dx, dy);
        dx -= tx; dy -= ty;
        x -= (jyy * dx - jxy * dy) / determinant;
        y -= (jxx * dy - jxy * dx) / determinant;
    }
#endif
    Real norm = sqrt(x * x + y * y + 1);
    x /= norm; y /= norm; z /= norm;
#endif
    Real recovered_x, recovered_y;
    bool valid = project(x, y, z, camera, recovered_x, recovered_y)
        && pixel_valid(pixel_x, pixel_y, camera)
        && fmax(fabs(recovered_x - pixel_x), fabs(recovered_y - pixel_y)) < Real(0.001);
    Real pitch, roll;
#if PAN_TILT
    pitch = atan2(x, z); roll = atan2(-y, sqrt(x * x + z * z));
#else
    pitch = atan2(x, sqrt(y * y + z * z)); roll = atan2(-y, z);
#endif
    Real cp = cos(pitch), sp = sin(pitch), cr = cos(roll), sr = sin(roll);
    Real* r = gaze + 10 * b;
#if PAN_TILT
    r[0] = cp; r[1] = sp * sr; r[2] = sp * cr;
    r[3] = 0;  r[4] = cr;      r[5] = -sr;
    r[6] = -sp; r[7] = cp * sr; r[8] = cp * cr;
#else
    r[0] = cp;      r[1] = 0;  r[2] = sp;
    r[3] = sr * sp; r[4] = cr; r[5] = -sr * cp;
    r[6] = -cr * sp; r[7] = sr; r[8] = cr * cp;
#endif
    r[9] = valid ? 1 : 0;
}

__device__ Real read_pixel(const Input* image, int x, int y, int h, int w,
                          long long stride_y, long long stride_x) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0;
    return Real(image[y * stride_y + x * stride_x]);
}

extern "C" __global__ void calibrated_sample(
    const Input* image, const float* rays, const Real* rotation,
    Output* output, Real* pixels, long long stride_b, long long stride_c,
    long long stride_y, long long stride_x, long long rotation_b,
    long long rotation_row, long long rotation_col,
    int batch, int channels, int points, CAMERA_ARGS) {
    CAMERA_INIT;
    for (long long linear = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         linear < (long long)batch * points; linear += (long long)blockDim.x * gridDim.x) {
        int n = linear % points, b = linear / points;
        const Real* r = rotation + b * rotation_b;
        Real x = rays[3 * n], y = rays[3 * n + 1], z = rays[3 * n + 2];
        // Match the accumulation used by batched matrix multiplication.
        Real rx = fma(r[2 * rotation_col], z, fma(r[rotation_col], y, r[0] * x));
        Real ry = fma(r[rotation_row + 2 * rotation_col], z, fma(r[rotation_row + rotation_col], y, r[rotation_row] * x));
        Real rz = fma(r[2 * rotation_row + 2 * rotation_col], z, fma(r[2 * rotation_row + rotation_col], y, r[2 * rotation_row] * x));
        Real u, v;
        bool valid = project(rx, ry, rz, camera, u, v);
#if !EXPLICIT_ROTATION
        valid = valid && r[9] != 0;
#endif
        if (pixels) { pixels[2 * linear] = u; pixels[2 * linear + 1] = v; }
        if (!valid) { u = 0; v = 0; }
        int x0 = int(floor(u)), y0 = int(floor(v));
        Real wx = u - x0, wy = v - y0;
        for (int c = 0; c < channels; ++c) {
            const Input* source = image + b * stride_b + c * stride_c;
            Real value = 0;
            if (valid) {
#if BILINEAR
                value = read_pixel(source, x0, y0, height, width, stride_y, stride_x) * (1 - wx) * (1 - wy)
                      + read_pixel(source, x0 + 1, y0, height, width, stride_y, stride_x) * wx * (1 - wy)
                      + read_pixel(source, x0, y0 + 1, height, width, stride_y, stride_x) * (1 - wx) * wy
                      + read_pixel(source, x0 + 1, y0 + 1, height, width, stride_y, stride_x) * wx * wy;
#else
                value = read_pixel(source, int(floor(u + Real(0.5))), int(floor(v + Real(0.5))),
                                   height, width, stride_y, stride_x);
#endif
            }
            output[((long long)b * channels + c) * points + n] = Output(value);
        }
    }
}
