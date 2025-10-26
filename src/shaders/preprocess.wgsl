struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32, 2>, // Combined together because each value is 16 bits
    rot: array<u32, 2>,
    scale: array<u32, 2>
};

@group(0) @binding(0) var<uniform> numPoints: u32;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(3) var<uniform> gaussianScaling: f32;

@group(1) @binding(0) var<storage, read_write> splats: array<vec3<f32>>;

@group(2) @binding(0) var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1) var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2) var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3) var<storage, read_write> sort_dispatch: DispatchIndirect;

const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32, 5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32, 7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    return vec3<f32>(0.0);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize, 1, 1)
fn preprocess(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) wgs: vec3<u32>
) {
    let index = gid.x;

    if (index >= numPoints) {
        return;
    }

    let gaussian = gaussians[index];
    let posXY: vec2<f32> = unpack2x16float(gaussian.pos_opacity[0]);
    let posZopacity: vec2<f32> = unpack2x16float(gaussian.pos_opacity[1]);
    
    // TODO(aczw): can also construct quads here, and store the radius in `splats`. This would
    // allow the quad size to also change based on depth
    let worldPos = vec4<f32>(posXY.x, posXY.y, posZopacity.x, 1.0);
    let t = camera.view * worldPos;
    let clipPos = camera.proj * t;
    let ndcPos: vec3<f32> = clipPos.xyz / clipPos.w;

    // Frustum culling. Use a slightly bigger bounding box so we still draw splats on the edges
    if (ndcPos.x < -1.2 || ndcPos.x > 1.2 || ndcPos.y < -1.2 || ndcPos.y > 1.2) {
        return;
    }

    // Unpack rotation and scale components
    let rotRX: vec2<f32> = unpack2x16float(gaussian.rot[0]);
    let rotYZ: vec2<f32> = unpack2x16float(gaussian.rot[1]);
    let scaleXY: vec2<f32> = unpack2x16float(gaussian.scale[0]);
    let scaleZW: vec2<f32> = unpack2x16float(gaussian.scale[1]);

    // Construct quaternion and normalize just in case
    let quat = normalize(vec4<f32>(rotRX.x, rotRX.y, rotYZ.x, rotYZ.y));
    let r = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;

    let rot = mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
    );
    
    var scale = mat3x3<f32>();
    scale[0][0] = gaussianScaling * scaleXY.x;
    scale[1][1] = gaussianScaling * scaleXY.y;
    scale[2][2] = gaussianScaling * scaleZW.x;

    // Compute 3D covariance
    let m = scale * rot;
    let cov3d: mat3x3<f32> = transpose(m) * m;

    let focal: vec2<f32> = camera.focal;
    let jacobian = mat3x3<f32>(
        focal.x / t.z, 0.0, -(focal.x * t.x) / (t.z * t.z),
		0.0, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
		0.0, 0.0, 0.0
    );

    let view: mat4x4<f32> = camera.view;
    let w = mat3x3<f32>(
        view[0][0], view[1][0], view[2][0],
        view[0][1], view[1][1], view[2][1],
        view[0][2], view[1][2], view[2][2],
    );

    // Compute 2D covariance
    let wj: mat3x3<f32> = w * jacobian;
    var cov2d: mat3x3<f32> = transpose(wj) * transpose(cov3d) * wj;

    cov2d[0][0] += 0.3;
    cov2d[1][1] += 0.3;

    // Covariance is symmetrial, so we can just store diagonal once
    let cov = vec3<f32>(cov2d[0][0], cov2d[0][1], cov2d[1][1]);

    // Determinant
    let det = cov.x * cov.z - cov.y * cov.y;

    if (det == 0.0) {
        return;
    }

    // Compute conic (inverse 2D covariance)
    let detInv = 1.0 / det;
    let conic = vec3<f32>(cov.z * detInv, -cov.y * detInv, cov.x * detInv);

    let prevSize: u32 = atomicAdd(&sort_infos.keys_size, 1u);
    splats[prevSize] = ndcPos;

    let keysPerDispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
}
