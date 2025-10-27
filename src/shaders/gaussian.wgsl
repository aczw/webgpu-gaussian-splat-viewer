struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Splat {
    center: vec2<f32>, /* In NDC coordinates */
    radius: vec2<f32>, /* In NDC coordinates */
    conicOpacity: vec4<f32>,
    color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) center: vec2<f32>,
    @location(2) conicOpacity: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<uniform> scaling: f32;

const MIN_OPACITY: f32 = 1.0 / 255.0;

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32 /* Easy access to the number of splats to draw */
) -> VertexOutput {
    let splat: Splat = splats[instIdx];

    // Create the six possible vertex positions in NDC space
    // TODO(aczw): surely there has to be a better way to do this
    let size: vec2<f32> = splat.radius;
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(-size.x, -size.y), vec2<f32>(size.x, -size.y), vec2<f32>(-size.x, size.y),
        vec2<f32>(-size.x, size.y), vec2<f32>(size.x, -size.y), vec2<f32>(size.x, size.y)
    );

    // Translate to "world space" NDC position from "local space" (centered at origin)
    let localPos: vec2<f32> = positions[vertIdx] * scaling;
    let worldPos = vec3<f32>(localPos + splat.center, 0.0);

    var out: VertexOutput;
    out.position = vec4<f32>(worldPos, 1.0);
    out.color = splat.color;
    out.center = splat.center;
    out.conicOpacity = splat.conicOpacity;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var ndcPosition = (in.position.xy / camera.viewport) * 2.0 - 1.0;
    ndcPosition.y *= -1.0; // @builtin(position) flips y-coord of framebuffer

    // Find offset from splat center, flip X, and convert to pixel space
    var d = ndcPosition - in.center;
    d.x *= -1.0;
    d *= camera.viewport * 0.5;

    let con: vec3<f32> = in.conicOpacity.xyz;
    let power = -0.5 * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;

    if (power > 0.0) {
        return vec4<f32>();
    }
    
    let alpha = min(0.99, in.conicOpacity.w * exp(power));

    if (alpha < MIN_OPACITY) {
        return vec4<f32>();
    }

    return vec4<f32>(in.color, 1.0) * alpha;
}
