struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

//TODO: information defined in preprocess compute shader
// struct Splat {};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

const positions = array<vec2<f32>, 6>(
    vec2<f32>(-0.01, -0.01), vec2<f32>(0.01, -0.01), vec2<f32>(-0.01, 0.01),
    vec2<f32>(-0.01, 0.01), vec2<f32>(0.01, -0.01), vec2<f32>(0.01, 0.01)
);

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    // TODO: reconstruct 2D quad based on information from splat, pass
    let aspect = camera.viewport.x / camera.viewport.y;
    let position = positions[index];

    var out: VertexOutput;
    out.position = vec4<f32>(position.x, position.y * aspect, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
