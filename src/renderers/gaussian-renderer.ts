import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setScalingMultiplier: (multiplier: number) => void;
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);


  // settings for gaussian rendering (scaling, sh_deg)
  const gaussian_rendering_settings_array = new Float32Array(4);
  gaussian_rendering_settings_array[0] = 1.0; // scaling multiplier
  gaussian_rendering_settings_array[1] = pc.sh_deg; // padding

  const gaussian_rendering_settings_buffer = createBuffer(
    device,
    'gaussian uniforms',
    gaussian_rendering_settings_array.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    gaussian_rendering_settings_array
  );

  // splat buffer
  const splatSize = 32;
  const splatBuffer = createBuffer(
    device,
    'splat buffer',
    pc.num_points * splatSize,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  );

  // indirect draw
  const drawArgsArray = new Uint32Array([6, pc.num_points, 0, 0]); // vertexCount, instanceCount, firstVertex, firstInstance
  const indirectDrawBuffer = createBuffer(
    device,
    'indirect draw buffer',
    drawArgsArray.byteLength,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    drawArgsArray,
  );

  // nulling buffer
  const nulling_buffer = createBuffer(
    device,
    'nulling buffer',
    nulling_data.byteLength,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    nulling_data,
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const preprocess_camera_bind_group = device.createBindGroup({
    label: 'camera bind group',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
    ],
  });

  const preprocess_gaussian_bind_group = device.createBindGroup({
    label: 'gaussian bind group',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splatBuffer } },
      { binding: 2, resource: { buffer: gaussian_rendering_settings_buffer } },
    ],
  });

  const preprocess_sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: {
      module: device.createShaderModule(
        { 
          code: renderWGSL 
        }
      ),
      entryPoint: 'vs_main',
    },
    fragment: {
      module: device.createShaderModule(
        {
          code: renderWGSL
        }
      ),
      entryPoint: 'fs_main',
      targets: [
        { 
          format: presentation_format 
        }
      ],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const render_gaussian_bind_group = device.createBindGroup({
    label: 'render gaussian bind group',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: splatBuffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================


  const preprocess = (encoder: GPUCommandEncoder) => {
    device.queue.writeBuffer(
      sorter.sort_info_buffer, 0, new Uint32Array([0])
    );
    const pass = encoder.beginComputePass({
      label: 'preprocess pass',
    });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_camera_bind_group);
    pass.setBindGroup(1, preprocess_gaussian_bind_group);
    pass.setBindGroup(2, preprocess_sort_bind_group);
    const numWorkgroups = Math.ceil(pc.num_points / C.histogram_wg_size);

    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render pass',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(1, render_gaussian_bind_group);
    pass.drawIndirect(indirectDrawBuffer, 0);
    pass.end();
  };

  // ===============================================
  //    Scaling Multiplier Function
  // ===============================================

  // update scaling multiplier
  function setScalingMultiplier(multiplier: number) {
    gaussian_rendering_settings_array[0] = multiplier;
    device.queue.writeBuffer(
      gaussian_rendering_settings_buffer, 0,
      gaussian_rendering_settings_array
    );
  }

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {

      encoder.copyBufferToBuffer(
        nulling_buffer, 0,
        sorter.sort_info_buffer, 0, 4
      );

      encoder.copyBufferToBuffer(
        nulling_buffer, 0,
        indirectDrawBuffer, 4, 4
      );

      preprocess(encoder);
      sorter.sort(encoder);

      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer, 0,
        indirectDrawBuffer, 4, 4
      );

      render(encoder, texture_view);
    },
    camera_buffer,
    setScalingMultiplier,
  };
}
