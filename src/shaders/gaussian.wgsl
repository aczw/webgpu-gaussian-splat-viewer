struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Splat {
    center: u32, /* In NDC coordinates */
    conicOpacity: array<u32, 2>,
    colorRadius: array<u32, 2>, /* Radius is in pixel coordinates */
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat, either) center: u32,
    @location(1) @interpolate(flat, either) conicOpacity: vec2<u32>,
    @location(2) @interpolate(flat, either) colorRadius: vec2<u32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<uniform> scaling: f32;
@group(0) @binding(3) var<storage, read> sortIndices: array<u32>;

const MIN_OPACITY: f32 = 1.0 / 255.0;

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32 /* Easy access to the number of splats to draw */
) -> VertexOutput {
    let index = sortIndices[instIdx];
    let splat: Splat = splats[index];
    
    // Create the six possible vertex positions in NDC space
    // TODO(aczw): surely there has to be a better way to do this
    let colorBRadius: vec2<f32> = unpack2x16float(splat.colorRadius[1]);
    let size: vec2<f32> = colorBRadius.yy / camera.viewport;
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(-size.x, -size.y), vec2<f32>(size.x, -size.y), vec2<f32>(-size.x, size.y),
        vec2<f32>(-size.x, size.y), vec2<f32>(size.x, -size.y), vec2<f32>(size.x, size.y)
    );

    // Translate to "world space" NDC position from "local space" (centered at origin)
    let localNdcPos: vec2<f32> = positions[vertIdx] * scaling;
    let worldNdcPos = vec2<f32>(localNdcPos + unpack2x16float(splat.center));

    var out: VertexOutput;
    out.position = vec4<f32>(worldNdcPos, 0.0, 1.0);
    out.center = splat.center;
    out.conicOpacity = vec2<u32>(splat.conicOpacity[0], splat.conicOpacity[1]);
    out.colorRadius = vec2<u32>(splat.colorRadius[0], splat.colorRadius[1]);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var ndcPosition = (in.position.xy / camera.viewport) * 2.0 - 1.0;
    ndcPosition.y *= -1.0; // @builtin(position) flips y-coord of framebuffer

    // Find offset from splat center, flip X, and convert to pixel space
    var d = ndcPosition - unpack2x16float(in.center);
    d.x *= -1.0;
    d *= camera.viewport * 0.5;

    let conicXY: vec2<f32> = unpack2x16float(in.conicOpacity[0]);
    let conicZOpacity: vec2<f32> = unpack2x16float(in.conicOpacity[1]);

    let con = vec3<f32>(conicXY.xy, conicZOpacity.x);
    let power = -0.5 * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;

    if (power > 0.0) {
        return vec4<f32>();
    }
    
    let alpha = min(0.99, conicZOpacity.y * exp(power));

    if (alpha < MIN_OPACITY) {
        return vec4<f32>();
    }

    let colorRG: vec2<f32> = unpack2x16float(in.colorRadius[0]);
    let colorBRadius: vec2<f32> = unpack2x16float(in.colorRadius[1]);

    return vec4<f32>(colorRG.xy, colorBRadius.x, 1.0) * alpha;
}
