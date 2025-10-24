import { PointCloud } from "../utils/load";
import { get_sorter, c_histogram_block_rows, C } from "../sort/sort";
import { Renderer } from "./renderer";

import preprocessWgsl from "../shaders/preprocess.wgsl";
import gaussianWgsl from "../shaders/gaussian.wgsl";

export interface GaussianRenderer extends Renderer {}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView<ArrayBuffer>
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer
): GaussianRenderer {
  const sorter = get_sorter(pc.num_points, device);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================
  const indirectBuffer = createBuffer(
    device,
    "Gaussian indirect draw params buffer",
    4 * Uint32Array.BYTES_PER_ELEMENT /* 16 */,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    new Uint32Array([6, pc.num_points, 0, 0])
  );
  const nulling_data = new Uint32Array([0]);

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: "preprocess",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: preprocessWgsl }),
      entryPoint: "preprocess",
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const sort_bind_group = device.createBindGroup({
    label: "sort",
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      {
        binding: 1,
        resource: { buffer: sorter.ping_pong[0].sort_depths_buffer },
      },
      {
        binding: 2,
        resource: { buffer: sorter.ping_pong[0].sort_indices_buffer },
      },
      {
        binding: 3,
        resource: { buffer: sorter.sort_dispatch_indirect_buffer },
      },
    ],
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const gaussianRenderShader = device.createShaderModule({
    label: "Gaussian indirect render shader (vert/frag)",
    code: gaussianWgsl,
  });

  // TODO(aczw): primitive should use triangle fan/strip instead of individual triangles?
  const uniformsBgl = device.createBindGroupLayout({
    label: "Uniforms bind group layout",
    entries: [
      {
        // Camera uniforms
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });
  const uniformsBg = device.createBindGroup({
    label: "Uniforms bind group",
    layout: uniformsBgl,
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  const renderPipeline = device.createRenderPipeline({
    label: "Gaussian indirect render pipeline",
    layout: device.createPipelineLayout({
      label: "Gaussian indirect render pipeline layout",
      bindGroupLayouts: [uniformsBgl],
    }),
    vertex: {
      module: gaussianRenderShader,
      entryPoint: "vs_main",
    },
    fragment: {
      module: gaussianRenderShader,
      entryPoint: "fs_main",
      targets: [{ format: presentation_format }],
    },
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      sorter.sort(encoder);

      const renderPass = encoder.beginRenderPass({
        label: "Gaussian indirect render pass",
        colorAttachments: [
          {
            view: texture_view,
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });

      renderPass.setPipeline(renderPipeline);
      renderPass.setBindGroup(0, uniformsBg);
      renderPass.drawIndirect(indirectBuffer, 0);

      renderPass.end();
    },
    camera_buffer,
  };
}
