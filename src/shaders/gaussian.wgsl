struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color : vec3<f32>,
};

struct Splat {
    //TODO: store information for 2D splat rendering
    ndc_center: vec2<f32>,
    radius: f32,
    ndc_depth: f32,
    color: vec4<f32>,
};

struct SplatsBuf {
  data: array<Splat>,
};

struct SortedBuf {
  data: array<u32>,
};

@group(1) @binding(0)
var<storage, read> splats: SplatsBuf;
@group(1) @binding(1)
var<storage, read> sorted_indices : SortedBuf;

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
    if (instance_index >= arrayLength(&sorted_indices.data)) {
        out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0); // Clip out
        return out;
    }
    var splat_idx = sorted_indices.data[instance_index];

    splat_idx = instance_index;
    // let splat_idx = sorted_indices[in_instance_index];
    if (splat_idx >= arrayLength(&splats.data)) {
        out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0); // Clip out
        return out;
    }
    // splat info
    let splat = splats.data[splat_idx];
    let offset = quad_vertex_offset(vertex_index % 6u) * splat.radius;
    let ndc_out = splat.ndc_center + offset;

    // out info
    out.position = vec4<f32>(ndc_out.x, ndc_out.y, splat.ndc_depth, 1.0);
    out.color = splat.color.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}