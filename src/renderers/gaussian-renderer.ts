import { PointCloud } from "../utils/load";
import { get_sorter, c_histogram_block_rows, C } from "../sort/sort";
import { Renderer } from "./renderer";

import preprocessWgsl from "../shaders/preprocess.wgsl";
import gaussianWgsl from "../shaders/gaussian.wgsl";

export interface GaussianRenderer extends Renderer {
  updateScaling: (scaling: number) => void;
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
  // @ts-ignore
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

  const numPointsUniformBuffer = createBuffer(
    device,
    "Gaussian preprocess num points uniform buffer",
    Uint32Array.BYTES_PER_ELEMENT /* 4 */,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    new Uint32Array([pc.num_points])
  );

  const splatsStorageBuffer = createBuffer(
    device,
    "Gaussian splats storage buffer",
    pc.num_points * (2 + 2 + 3) * Float32Array.BYTES_PER_ELEMENT /* num_points * 28 */,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
  );

  const renderSettingsUniformBuffer = createBuffer(
    device,
    "Gaussian render settings uniform buffer",
    2 * Float32Array.BYTES_PER_ELEMENT /* 8 */,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    new Float32Array([1, pc.sh_deg])
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocessConstantsBgl = device.createBindGroupLayout({
    label: "Gaussian preprocess constants bind group layout",
    entries: [
      {
        // Num points uniform
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      {
        // Camera uniforms
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      {
        // Gaussians storage buffer
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        // Gaussian scaling uniform buffer
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      {
        // SH buffer
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
    ],
  });
  const preprocessConstantsBg = device.createBindGroup({
    label: "Gaussian preprocess constants bind group",
    layout: preprocessConstantsBgl,
    entries: [
      { binding: 0, resource: { buffer: numPointsUniformBuffer } },
      { binding: 1, resource: { buffer: camera_buffer } },
      { binding: 2, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 3, resource: { buffer: renderSettingsUniformBuffer } },
      { binding: 4, resource: { buffer: pc.sh_buffer } },
    ],
  });

  const preprocessSplatsBgl = device.createBindGroupLayout({
    label: "Gaussian preprocess splats bind group layout",
    entries: [
      // Splats storage buffer
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const preprocessSplatsBg = device.createBindGroup({
    label: "Gaussian preprocess splats bind group",
    layout: preprocessSplatsBgl,
    entries: [{ binding: 0, resource: { buffer: splatsStorageBuffer } }],
  });

  const preprocessSortBgl = device.createBindGroupLayout({
    label: "Gaussian preprocess sort bind group layout",
    entries: [
      {
        // Sort info structs
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        // Sort depths
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        // Sort indices
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        // Sort dispatch
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  const preprocessSortBg = device.createBindGroup({
    label: "Gaussian preprocess sort bind group",
    layout: preprocessSortBgl,
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

  const preprocessPipeline = device.createComputePipeline({
    label: "Gaussian preprocess compute pipeline",
    layout: device.createPipelineLayout({
      label: "Gaussian preprocess compute pipeline layout",
      bindGroupLayouts: [preprocessConstantsBgl, preprocessSplatsBgl, preprocessSortBgl],
    }),
    compute: {
      module: device.createShaderModule({
        label: "Gaussian preprocess compute shader",
        code: preprocessWgsl,
      }),
      entryPoint: "preprocess",
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const gaussianRenderShader = device.createShaderModule({
    label: "Gaussian indirect render shader (vert/frag)",
    code: gaussianWgsl,
  });

  // TODO(aczw): primitive should use triangle fan/strip instead of individual triangles?
  const constantsBgl = device.createBindGroupLayout({
    label: "Gaussian indirect render constants bind group layout",
    entries: [
      {
        // Camera uniforms
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
      {
        // Splats storage buffer
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
      {
        // Scaling uniform buffer
        binding: 2,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });
  const constantsBg = device.createBindGroup({
    label: "Gaussian indirect render constants bind group",
    layout: constantsBgl,
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: splatsStorageBuffer } },
      { binding: 2, resource: { buffer: renderSettingsUniformBuffer } },
    ],
  });

  const renderPipeline = device.createRenderPipeline({
    label: "Gaussian indirect render pipeline",
    layout: device.createPipelineLayout({
      label: "Gaussian indirect render pipeline layout",
      bindGroupLayouts: [constantsBgl],
    }),
    vertex: {
      module: gaussianRenderShader,
      entryPoint: "vs_main",
    },
    fragment: {
      module: gaussianRenderShader,
      entryPoint: "fs_main",
      targets: [
        {
          format: presentation_format,
          blend: {
            color: {
              operation: "add",
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
            },
            alpha: {
              operation: "add",
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
            },
          },
        },
      ],
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
      encoder.clearBuffer(sorter.sort_dispatch_indirect_buffer, 0, 4);
      encoder.clearBuffer(sorter.sort_info_buffer, 0, 4);
      encoder.clearBuffer(splatsStorageBuffer);

      {
        const preprocessPass = encoder.beginComputePass({
          label: "Gaussian preprocess compute pass",
        });

        preprocessPass.setPipeline(preprocessPipeline);
        preprocessPass.setBindGroup(0, preprocessConstantsBg);
        preprocessPass.setBindGroup(1, preprocessSplatsBg);
        preprocessPass.setBindGroup(2, preprocessSortBg);
        preprocessPass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size));

        preprocessPass.end();
      }

      // Use updated splat count for instance count in indirect drawing
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirectBuffer, 4, 4);

      sorter.sort(encoder);

      {
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
        renderPass.setBindGroup(0, constantsBg);
        renderPass.drawIndirect(indirectBuffer, 0);

        renderPass.end();
      }
    },
    camera_buffer,
    updateScaling: (scaling) =>
      device.queue.writeBuffer(
        renderSettingsUniformBuffer,
        0,
        new Float32Array([scaling, pc.sh_deg])
      ),
  };
}
