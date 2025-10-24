struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> splats: array<vec3<f32>>;
@group(0) @binding(2) var<uniform> scaling: f32;

const positions = array<vec2<f32>, 6>(
    vec2<f32>(-0.01, -0.01), vec2<f32>(0.01, -0.01), vec2<f32>(-0.01, 0.01),
    vec2<f32>(-0.01, 0.01), vec2<f32>(0.01, -0.01), vec2<f32>(0.01, 0.01)
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertIdx: u32,
    @builtin(instance_index) instIdx: u32 /* Easy access to the number of splats to draw */
) -> VertexOutput {
    var localPos: vec2<f32> = positions[vertIdx] * scaling;
    localPos = vec2<f32>(localPos.x, localPos.y * (camera.viewport.x / camera.viewport.y));

    // Translate to "world space" NDC position from "local space" (centered at origin)
    let offset: vec3<f32> = splats[instIdx];
    let worldPos = vec3<f32>(localPos.xy + offset.xy, offset.z);

    var out: VertexOutput;
    out.position = vec4<f32>(worldPos, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
