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
    color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(1) color: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<uniform> scaling: f32;

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

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
