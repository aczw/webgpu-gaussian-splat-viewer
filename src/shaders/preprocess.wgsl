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
    scaling: f32,
    shDeg: u32,
}

struct Gaussian {
    pos_opacity: array<u32, 2>,
    rot: array<u32, 2>,
    scale: array<u32, 2>
};

struct Splat {
    center: vec2<f32>, /* In NDC coordinates */
    radius: vec2<f32>, /* In NDC coordinates */
    color: vec3<f32>,
};

@group(0) @binding(0) var<uniform> numPoints: u32;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(3) var<uniform> settings: RenderSettings;
@group(0) @binding(4) var<storage, read> shCoefficients: array<u32>;

@group(1) @binding(0) var<storage, read_write> splats: array<Splat>;

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

// Reads the nth SH coefficient from the storage buffer.
//
// Each vertex has up to 16 coefficients, each with 3 channels (RGB). Each channel
// is 16 bits = 2 bytes. This means each vertex takes up 16 * 3 * 2 = 96 bytes.
// We're reading the storage buffer as f32s, which takes up 4 bytes each.
//
// Therefore each vertex is strided by 96 / 4 = 24 elements.
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let offsetIdx: u32 = splat_idx * 24u;

    // Load all the coefficients
    // TODO(aczw): is this faster than finding the right offset (will need modulo?)
    // TODO(aczw): precompute the `coefficients` array once for the vertex and reuse it in here
    var coefficients: array<vec3<f32>, 16>;
    var coeffIdx = 0u;

    for (var currElt = offsetIdx; currElt < offsetIdx + 24u; currElt += 3u) {
        let eltA = shCoefficients[currElt];
        let eltB = shCoefficients[currElt + 1u];
        let eltC = shCoefficients[currElt + 2u];
        
        let eltASplit: vec2<f32> = unpack2x16float(eltA);
        let eltBSplit: vec2<f32> = unpack2x16float(eltB);
        let eltCSplit: vec2<f32> = unpack2x16float(eltC);

        let coeffA = vec3<f32>(eltASplit.x, eltASplit.y, eltBSplit.x);
        let coeffB = vec3<f32>(eltBSplit.y, eltCSplit.x, eltCSplit.y);

        coefficients[coeffIdx] = coeffA;
        coefficients[coeffIdx + 1u] = coeffB;

        coeffIdx += 2u;
    }

    return coefficients[c_idx];
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
    
    let worldPos = vec4<f32>(posXY.x, posXY.y, posZopacity.x, 1.0);
    let viewPos = camera.view * worldPos;
    let clipPos = camera.proj * viewPos;
    let ndcPos: vec2<f32> = clipPos.xy / clipPos.w;

    // Frustum culling. Use a slightly bigger bounding box so we still draw splats on the edges
    if (ndcPos.x < -1.2 || ndcPos.x > 1.2 || ndcPos.y < -1.2 || ndcPos.y > 1.2) {
        return;
    }

    // Construct quaternion and normalize
    let rotationA: vec2<f32> = unpack2x16float(gaussian.rot[0]);
    let rotationB: vec2<f32> = unpack2x16float(gaussian.rot[1]);
    let quat = vec4<f32>(rotationA.x, rotationA.y, rotationB.x, rotationB.y);
    
    // Construct rotation matrix
    let r = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;
    let R = mat3x3<f32>(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
        2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
        2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
    );
    
    let scaleA: vec2<f32> = unpack2x16float(gaussian.scale[0]);
    let scaleB: vec2<f32> = unpack2x16float(gaussian.scale[1]);
    let scale = exp(vec3<f32>(scaleA.x, scaleA.y, scaleB.x));
    
    // Construct scale matrix
    let scaling = settings.scaling;
    var S = mat3x3<f32>();
    S[0][0] = scaling * scale.x;
    S[1][1] = scaling * scale.y;
    S[2][2] = scaling * scale.z;

    // Compute 3D covariance
    let M = S * R;
    let sigma: mat3x3<f32> = transpose(M) * M;

    // Covariance is symmetric, so we only store unique values
    let cov3d = array<f32, 6>(
        sigma[0][0], sigma[0][1], sigma[0][2],
        sigma[1][1], sigma[1][2], sigma[2][2]
    );

    let focal: vec2<f32> = camera.focal;
    let t: vec3<f32> = viewPos.xyz;
    let J = mat3x3<f32>(
        focal.x / t.z, 0.0, -(focal.x * t.x) / (t.z * t.z),
		0.0, focal.y / t.z, -(focal.y * t.y) / (t.z * t.z),
		0.0, 0.0, 0.0
    );

    let W = transpose(
        mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz)
    );

    let vrk = mat3x3<f32>(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );

    // Compute 2D covariance
    let T: mat3x3<f32> = W * J;
    var cov2d: mat3x3<f32> = transpose(T) * transpose(vrk) * T;
    cov2d[0][0] += 0.3;
    cov2d[1][1] += 0.3;

    // Covariance is symmetrical, so we can just store diagonal once
    let cov = vec3<f32>(cov2d[0][0], cov2d[0][1], cov2d[1][1]);

    // Determinant
    let det = cov.x * cov.z - cov.y * cov.y;

    if (det == 0.0) {
        return;
    }

    // Compute conic (inverse 2D covariance)
    let detInv = 1.0 / det;
    let conic = vec3<f32>(cov.z * detInv, -cov.y * detInv, cov.x * detInv);

    // Compute eigenvalues of covariance
    let mid = 0.5 * (cov.x + cov.z);
    let lambda1: f32 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2: f32 = mid - sqrt(max(0.1, mid * mid - det));

    // Compute radius of gaussian. Round to the nearest pixel because we're in pixel space
    let pixelRadius = ceil(3.0 * sqrt(max(lambda1, lambda2)));
    let radius = vec2<f32>(pixelRadius) / camera.viewport;

    // TODO(aczw): flip direction?
    let direction = normalize(viewPos.xyz);
    let color: vec3<f32> = computeColorFromSH(direction, index, settings.shDeg);

    var splat: Splat;
    splat.center = ndcPos.xy;
    splat.radius = radius;
    splat.color = color;

    let prevSize: u32 = atomicAdd(&sort_infos.keys_size, 1u);
    splats[prevSize] = splat;

    // TODO(aczw): remove modulo usage?
    let keysPerDispatch = workgroupSize * sortKeyPerThread;
    if ((prevSize + 1u) % keysPerDispatch == 0) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    sort_indices[prevSize] = prevSize;

    // The original paper interprets the depth this way for the key:
    // key |= *((uint32_t*)&depths[idx])
    // This performs a direct bitcast on the depth value, so we do that here as well
    // TODO(aczw): which way should we store the depth? Invert by subtracting by far plane?
    sort_depths[prevSize] = bitcast<u32>(viewPos.z);
}
