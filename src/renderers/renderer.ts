import { Pane } from "tweakpane";
import * as TweakpaneFileImportPlugin from "tweakpane-plugin-file-import";

import { load } from "../utils/load";
import { default as get_renderer_gaussian, GaussianRenderer } from "./gaussian-renderer";
import { default as get_renderer_pointcloud } from "./point-cloud-renderer";
import { Camera, type CameraPreset, load_camera_presets } from "../camera/camera";
import { CameraControl } from "../camera/camera-control";
import { time, timeReturn } from "../utils/simple-console";

export type PerfTimer = {
  querySet: GPUQuerySet;
  resolveBuffer: GPUBuffer;
  resultBuffer: GPUBuffer;
};

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView, perf?: PerfTimer) => void;
  camera_buffer: GPUBuffer;
}

export default async function init(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice,
  canTimestamp: boolean
) {
  let ply_file_loaded = false;
  let cam_file_loaded = false;
  let renderers: { pointcloud?: Renderer; gaussian?: Renderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined;
  let pointcloud_renderer: Renderer | undefined;
  let renderer: Renderer | undefined;
  let cameras: CameraPreset[];

  const camera = new Camera(canvas, device);
  new CameraControl(camera);

  const observer = new ResizeObserver(() => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    camera.on_update_canvas();
  });
  observer.observe(canvas);

  const presentation_format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentation_format,
    alphaMode: "opaque",
  });

  const params = {
    fps: 0.0,
    renderTime: 0.0,
    splatSize: 1,
    renderer: "gaussian",
    plyFile: "",
    camFile: "",
  };

  const pane = new Pane({
    expanded: true,
  });

  pane.registerPlugin(TweakpaneFileImportPlugin);

  {
    const stats = pane.addFolder({
      title: "Stats",
      expanded: true,
    });

    stats.addMonitor(params, "fps", { label: "FPS", interval: 50 });

    if (canTimestamp) {
      stats.addMonitor(params, "renderTime", {
        label: "Render time",
        interval: 50,
        format: (time) => `${time} µs`,
      });
    }
  }

  {
    const scene = pane.addFolder({
      title: "Scene",
      expanded: true,
    });

    scene
      .addInput(params, "plyFile", {
        label: "PLY file",
        view: "file-input",
        lineCount: 2,
        filetypes: [".ply"],
        invalidFiletypeMessage: "We can't accept those file types!",
      })
      .on("change", async (file) => {
        const uploadedFile = file.value;
        if (uploadedFile) {
          const pc = await load(uploadedFile, device);
          pointcloud_renderer = get_renderer_pointcloud(
            pc,
            device,
            presentation_format,
            camera.uniform_buffer
          );
          gaussian_renderer = get_renderer_gaussian(
            pc,
            device,
            presentation_format,
            camera.uniform_buffer
          );
          renderers = {
            pointcloud: pointcloud_renderer,
            gaussian: gaussian_renderer,
          };
          renderer = renderers[params["renderer"]];
          ply_file_loaded = true;
        } else {
          ply_file_loaded = false;
        }
      });

    scene
      .addInput(params, "camFile", {
        label: "Camera JSON file",
        view: "file-input",
        lineCount: 2,
        filetypes: [".json"],
        invalidFiletypeMessage: "We can't accept those filetypes!",
      })
      .on("change", async (file) => {
        const uploadedFile = file.value;
        if (uploadedFile) {
          cameras = await load_camera_presets(file.value);
          camera.set_preset(cameras[0]);
          cam_file_loaded = true;
        } else {
          cam_file_loaded = false;
        }
      });
  }

  {
    const render = pane.addFolder({
      title: "Render settings",
      expanded: true,
    });

    render
      .addInput(params, "renderer", {
        label: "Type",
        options: {
          "Point Cloud": "pointcloud",
          Gaussian: "gaussian",
        },
      })
      .on("change", (e) => {
        renderer = renderers[e.value];
      });

    render
      .addInput(params, "splatSize", { label: "Splat size", min: 0, max: 1.5 })
      .on("change", (e) => {
        gaussian_renderer?.updateScaling(e.value);
      });
  }

  document.addEventListener("keydown", (event) => {
    switch (event.key) {
      case "0":
      case "1":
      case "2":
      case "3":
      case "4":
      case "5":
      case "6":
      case "7":
      case "8":
      case "9":
        const i = parseInt(event.key);
        console.log(`set to camera preset ${i}`);
        camera.set_preset(cameras[i]);
        break;
    }
  });

  const perf: PerfTimer = (() => {
    if (!canTimestamp) return;

    const querySet = device.createQuerySet({
      label: "Performance query set",
      type: "timestamp",
      count: 2,
    });

    const resolveBuffer = device.createBuffer({
      label: "Performance resolve buffer",
      size: querySet.count * 8 /* 64 bits */,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });

    const resultBuffer = device.createBuffer({
      label: "Performance result buffer",
      size: resolveBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    return { canTimestamp, querySet, resolveBuffer, resultBuffer };
  })();

  function frame() {
    if (ply_file_loaded && cam_file_loaded) {
      params.fps = (1.0 / timeReturn()) * 1000.0;
      time();

      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      renderer.frame(encoder, texture_view, canTimestamp ? perf : null);

      device.queue.submit([encoder.finish()]);

      if (canTimestamp && perf.resultBuffer.mapState === "unmapped") {
        perf.resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
          const times = new BigInt64Array(perf.resultBuffer.getMappedRange());
          params["Render time"] = Number(times[1] - times[0]) / 1000;
          perf.resultBuffer.unmap();
        });
      }
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
