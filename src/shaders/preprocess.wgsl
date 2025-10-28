const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
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
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    ndc_center: vec2<f32>,
    radius: f32,
    ndc_depth: f32,
    color: vec4<f32>,
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<uniform> render_settings: RenderSettings;


@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

// see if ndc point is inside bounding box for 1.2x frustum
fn in_ndc_bounding_box(ndc_pos: vec3<f32>) -> bool {
    let s = 1.2;
    return ndc_pos.x >= -s && ndc_pos.x <= s &&
           ndc_pos.y >= -s && ndc_pos.y <= s &&
           ndc_pos.z >= -0.0 && ndc_pos.z <= 1.0;
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    return vec3<f32>(0.);
}

fn mat3_from_quat(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let r = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * r), 2.0 * (x * z + y * r)),
        vec3<f32>(2.0 * (x * y + z * r), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * r)),
        vec3<f32>(2.0 * (x * z - y * r), 2.0 * (y * z + x * r), 1.0 - 2.0 * (x * x + y * y))
    );
}

fn top3x3(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(m[0][0], m[0][1], m[0][2]),
        vec3<f32>(m[1][0], m[1][1], m[1][2]),
        vec3<f32>(m[2][0], m[2][1], m[2][2])
    );
}

fn compute3DCovarianceFromGaussian(g: Gaussian, modifier: f32) -> mat3x3<f32> {
    let scale_xy = unpack2x16float(g.scale[0]);
    let scale_zw = unpack2x16float(g.scale[1]);
    let scales = vec3<f32>(
        exp(scale_xy.x), 
        exp(scale_xy.y), 
        exp(scale_zw.x)
    );

    // Debug: clamp scales to reasonable values
    let rot_xy = unpack2x16float(g.rot[0]);
    let rot_zr = unpack2x16float(g.rot[1]);
    let quat = vec4<f32>(rot_xy.x, rot_xy.y, rot_zr.x, rot_zr.y);
    let rot_mat = mat3_from_quat(quat);

    let scale_mat = mat3x3<f32>(
        vec3<f32>(modifier * scales.x, 0.0, 0.0),
        vec3<f32>(0.0, modifier * scales.y, 0.0),
        vec3<f32>(0.0, 0.0, modifier * scales.z)
    );

    let covariance = rot_mat * scale_mat * transpose(scale_mat) * transpose(rot_mat);
    return covariance;
}

fn compute2DCovarianceFromGaussian(g: Gaussian, view_pos3: vec3<f32>, modifier: f32, cam: CameraUniforms) -> mat3x3<f32> {
    let cov3D = compute3DCovarianceFromGaussian(g, modifier);


    let fx = cam.focal.x;
    let fy = cam.focal.y;

    let jacobian = mat3x3<f32>(
        vec3<f32>(fx / view_pos3.z, 0.0, -fx * view_pos3.x / (view_pos3.z * view_pos3.z)),
        vec3<f32>(0.0, fy / view_pos3.z, -fy * view_pos3.y / (view_pos3.z * view_pos3.z)),
        vec3<f32>(0.0, 0.0, 0.0)
    );

    let W = transpose(mat3x3<f32>(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz));

    let T = W * jacobian;

    let Vrk = mat3x3f(
        cov3D[0][0], cov3D[0][1], cov3D[0][2],
        cov3D[0][1], cov3D[1][1], cov3D[1][2],
        cov3D[0][2], cov3D[1][2], cov3D[2][2]
    );

    var cov2D = transpose(T) * Vrk * T;

    cov2D[0][0] += 0.3;
    cov2D[1][1] += 0.3;

    return cov2D;
}

fn computeRadiusFromCovariance(cov2D: mat3x3<f32>) -> f32 {
    let det = cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[1][0];
    let mid = 0.5 * (cov2D[0][0] + cov2D[1][1]);
    let lambdaOffset = sqrt(max(0.1, mid * mid - det));

    let lambda1 = mid + lambdaOffset;
    let lambda2 = mid - lambdaOffset;

    let max_lambda = max(lambda1, lambda2);

    let radius = ceil(3.0 * sqrt(max_lambda));
    return radius;
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

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    let n = arrayLength(&gaussians);
    if (idx >= n) {
        return;
    }

    let g = gaussians[idx];
    // stored as x, y, z, opacity
    let a = unpack2x16float(g.pos_opacity[0]);
    let b = unpack2x16float(g.pos_opacity[1]);
    let world_pos = vec4<f32>(a.x, a.y, b.x, 1.);

    // convert from world to ndc space
    let view_pos = camera.view * world_pos;
    let clip_pos = camera.proj * view_pos;
    let ndc_pos = clip_pos.xyz / clip_pos.w;

    // discard splats outside of ndc bounding box
    if (!in_ndc_bounding_box(ndc_pos)) {
        return;
    }

    // reserve output
    let splat_idx = atomicAdd(&sort_infos.keys_size, 1);
    if (splat_idx >= arrayLength(&splats)) {
        return;
    }

    // fill in splat info
    var splat: Splat;
    splat.ndc_center = ndc_pos.xy;

    // use quad to represent splat for now

    // evaluate radius
    let view_dir = normalize(-view_pos.xyz);
    let view_pos3 = (camera.view * world_pos).xyz;
    let cov2D = compute2DCovarianceFromGaussian(g, view_pos3, render_settings.gaussian_scaling, camera);

    let radius = computeRadiusFromCovariance(cov2D);
    let radius2D = radius * vec2<f32>(1.0, 1.0) / camera.viewport;
    splat.radius = radius2D.x; // assuming square viewport
    splat.ndc_depth = ndc_pos.z;
    
    // evaluate color
    // white for now
    splat.color = vec4<f32>(radius2D.x, radius2D.y, 0.0, 1.0);
    // let sh_deg = u32(render_settings.sh_deg);
    // let color = computeColorFromSH(view_dir, idx, sh_deg);
    // splat.color = vec4<f32>(color, 1.0);

    splats[splat_idx] = splat;

    // compute sort key (depth)
    let depth = view_pos3.z;
    // sort in reverse order (farther splats have smaller sort key)
    let sort_key = bitcast<u32>(100 - depth);
    sort_depths[splat_idx] = sort_key;
    sort_indices[splat_idx] = splat_idx;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if ((splat_idx) % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}