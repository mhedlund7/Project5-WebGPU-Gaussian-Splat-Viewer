struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color : vec4<f32>,
    @location(1) conic : vec3<f32>,
    @location(2) opacity : f32,
    @location(3) center : vec2<f32>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    ndc_center: vec2<f32>,
    radius: vec2<f32>,
    ndc_depth: f32,
    pad: f32,
    color: vec4<f32>,
    conic_opacity: vec4<f32>,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sorted_indices : array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

fn quad_vertex_offset(vertex_index: u32) -> vec2<f32> {
    var offsets = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );
    return offsets[vertex_index];
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    let splat_idx = sorted_indices[instance_index];

    // splat info
    let splat = splats[splat_idx];
    let offset = quad_vertex_offset(vertex_index % 6u);
    let ndc_offset = vec2<f32>(
    offset.x * splat.radius.x,
    offset.y * splat.radius.y
    );
    let ndc_out = splat.ndc_center + ndc_offset;

    // out info
    out.position = vec4<f32>(ndc_out.x, ndc_out.y, splat.ndc_depth, 1.0);
    out.color = splat.color;
    out.conic = splat.conic_opacity.xyz;
    out.opacity = splat.conic_opacity.w;
    // convert center into pixel space
    out.center = (0.5 + splat.ndc_center * vec2<f32>(0.5, -0.5)) * camera.viewport;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var dist = in.position.xy - in.center.xy;
    dist.x = -dist.x;
    let power = -0.5 * (in.conic.x * dist.x * dist.x + in.conic.z * dist.y * dist.y) - in.conic.y * dist.x * dist.y;
    if (power > 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let alpha = min(0.99, in.opacity * exp(power));
    return in.color * alpha;
}