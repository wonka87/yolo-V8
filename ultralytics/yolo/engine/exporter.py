# Ultralytics YOLO 🚀, GPL-3.0 license
"""
Export a YOLOv8 PyTorch model to TorchScript or ONNX (detect-only fork).

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
ONNX                    | `onnx`                    | yolov8n.onnx

Python:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolov8n.pt format=onnx
"""
import json
import subprocess
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import hydra
import torch

from ultralytics.nn.modules import Detect, Segment
from ultralytics.nn.tasks import DetectionModel, SegmentationModel
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, callbacks, colorstr, get_default_args
from ultralytics.yolo.utils.checks import check_imgsz, check_requirements
from ultralytics.yolo.utils.files import file_size
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import guess_task_from_head, select_device, smart_inference_mode


def export_formats():
    # YOLOv8 export formats. The full table is kept because AutoBackend uses the suffixes (and their order) to detect
    # model types; only TorchScript and ONNX can be exported by this fork.
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],]
    return dict(zip(['Format', 'Argument', 'Suffix', 'CPU', 'GPU'], map(list, zip(*x))))


SUPPORTED_FORMATS = 'torchscript', 'onnx'


def try_export(inner_func):
    # YOLOv8 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args['prefix']
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f'{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)')
            return f, model
        except Exception as e:
            LOGGER.info(f'{prefix} export failure ❌ {dt.t:.1f}s: {e}')
            return None, None

    return outer_func


class Exporter:
    """
    Exporter

    A class for exporting a model.

    Attributes:
        args (OmegaConf): Configuration for the exporter.
        save_dir (Path): Directory to save results.
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        """
        Initializes the Exporter class.

        Args:
            config (str, optional): Path to a configuration file. Defaults to DEFAULT_CONFIG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        if overrides is None:
            overrides = {}
        self.args = get_config(config, overrides)
        self.callbacks = defaultdict(list, {k: [v] for k, v in callbacks.default_callbacks.items()})  # add callbacks
        callbacks.add_integration_callbacks(self)

    @smart_inference_mode()
    def __call__(self, model=None):
        self.run_callbacks("on_export_start")
        t = time.time()
        format = self.args.format.lower()  # to lowercase
        assert format in SUPPORTED_FORMATS, f'ERROR: Invalid format={format}, valid formats are {SUPPORTED_FORMATS}'
        jit, onnx = format == 'torchscript', format == 'onnx'

        # Load PyTorch model
        self.device = select_device('cpu' if self.args.device is None else self.args.device)
        if self.args.half:
            if self.device.type == 'cpu':
                LOGGER.info('half=True only compatible with GPU export, i.e. use device=0')
                self.args.half = False
            assert not self.args.dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic'

        # Checks
        # if self.args.batch == model.args['batch_size']:  # user has not modified training batch_size
        self.args.batch = 1
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.optimize:
            assert self.device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'

        # Input
        im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)
        file = Path(getattr(model, 'pt_path', None) or getattr(model, 'yaml_file', None) or model.yaml['yaml_file'])
        if file.suffix == '.yaml':
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()
        for k, m in model.named_modules():
            if isinstance(m, (Detect, Segment)):
                m.dynamic = self.args.dynamic
                m.export = True

        y = None
        for _ in range(2):
            y = model(im)  # dry runs
        if self.args.half:
            im, model = im.half(), model.half()  # to FP16
        shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

        # Warnings
        warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings('ignore', category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings('ignore', category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = tuple(y.shape) if isinstance(y, torch.Tensor) else (x.shape for x in y)
        self.metadata = {'stride': int(max(model.stride)), 'names': model.names}  # model metadata
        self.pretty_name = self.file.stem.replace('yolo', 'YOLO')

        # Exports
        f = ['', '']  # exported filenames
        if jit:  # TorchScript
            f[0], _ = self._export_torchscript()
        if onnx:  # ONNX
            f[1], _ = self._export_onnx()

        # Finish
        f = [str(x) for x in f if x]  # filter out '' and None
        if any(f):
            task = guess_task_from_head(model.yaml["head"][-1][-2])
            s = "-WARNING ⚠️ not yet supported for YOLOv8 exported models"
            LOGGER.info(f'\nExport complete ({time.time() - t:.1f}s)'
                        f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                        f"\nPredict:         yolo task={task} mode=predict model={f[-1]} {s}"
                        f"\nValidate:        yolo task={task} mode=val model={f[-1]} {s}"
                        f"\nVisualize:       https://netron.app")

        self.run_callbacks("on_export_end")
        return f  # return list of exported files/dirs

    @try_export
    def _export_torchscript(self, prefix=colorstr('TorchScript:')):
        # YOLOv8 TorchScript model export
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = self.file.with_suffix('.torchscript')

        ts = torch.jit.trace(self.model, self.im, strict=False)
        d = {"shape": self.im.shape, "stride": int(max(self.model.stride)), "names": self.model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            LOGGER.info(f'{prefix} optimizing for mobile...')
            from torch.utils.mobile_optimizer import optimize_for_mobile
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        return f, None

    @try_export
    def _export_onnx(self, prefix=colorstr('ONNX:')):
        # YOLOv8 ONNX export
        check_requirements('onnx>=1.12.0')
        import onnx  # noqa

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = str(self.file.with_suffix('.onnx'))

        output_names = ['output0', 'output1'] if isinstance(self.model, SegmentationModel) else ['output0']
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
                dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionModel):
                dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

        torch.onnx.export(
            self.model.cpu() if dynamic else self.model,  # --dynamic only compatible with cpu
            self.im.cpu() if dynamic else self.im,
            f,
            verbose=False,
            opset_version=self.args.opset,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic or None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(self.model.stride)), 'names': self.model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, f)

        # Simplify
        if self.args.simplify:
            try:
                check_requirements('onnxsim')
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                subprocess.run(f'onnxsim {f} {f}', shell=True)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        return f, model_onnx

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def export(cfg):
    cfg.model = cfg.model or "yolov8n.yaml"
    cfg.format = cfg.format or "torchscript"

    # exporter = Exporter(cfg)
    #
    # model = None
    # if isinstance(cfg.model, (str, Path)):
    #     if Path(cfg.model).suffix == '.yaml':
    #         model = DetectionModel(cfg.model)
    #     elif Path(cfg.model).suffix == '.pt':
    #         model = attempt_load_weights(cfg.model, fuse=True)
    #     else:
    #         TypeError(f'Unsupported model type {cfg.model}')
    # exporter(model=model)

    from ultralytics import YOLO
    model = YOLO(cfg.model)
    model.export(**cfg)


if __name__ == "__main__":
    """
    CLI:
    yolo mode=export model=yolov8n.yaml format=onnx
    """
    export()
