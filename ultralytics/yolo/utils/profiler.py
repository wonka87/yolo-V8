# Ultralytics YOLO 🚀, GPL-3.0 license
"""Optional training-loop profiler (YOLO_PROFILE=1). Measures where training time goes without changing training:

- data_wait_s: time the loop waited for the next batch from the dataloader (CPU augmentation bound if large)
- step_s: forward + loss + backward + optimizer + logging per batch (wall time; the per-batch loss logging
  already synchronizes CUDA, so this includes the GPU work)
- epoch_end_s: per-epoch work after the batch loop (scheduler, EMA attrs, final val, checkpoint save)
- gpu_util_mean / gpu_util_samples: nvidia-smi utilization sampled every second during training
YOLO_PROFILE=2 additionally times the phases of each step (forward, loss, assigner, backward, optimizer, log) with a
CUDA synchronize at every phase boundary (adds a little overhead, does not change results) and records the padded
number of gts per batch (the assigner works on batch x max_gts x anchors).
Writes <save_dir>/profile.json at the end of training.
"""
import contextlib
import json
import shutil
import subprocess
import threading
import time


class TrainProfiler:
    current = None  # the running profiler, for phase timing inside the loss (TrainProfiler.phase)

    def __init__(self, enabled=False, sample_s=1.0, level=1):
        self.enabled = enabled
        self.level = level if enabled else 0
        self.phases, self.counters = {}, {}
        TrainProfiler.current = self if enabled else None
        self.data_wait = self.step = self.epoch_end = 0.0
        self.n_batches = self.n_epochs = 0
        self.epochs = []
        self._util = []
        self._t = self._t_loop_end = None
        self._stop = threading.Event()
        if enabled and shutil.which('nvidia-smi'):
            threading.Thread(target=self._sample_gpu, args=(sample_s,), daemon=True).start()
        self.t0 = time.time()

    def _sample_gpu(self, sample_s):
        while not self._stop.is_set():
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=5).stdout.split()
                if out:
                    self._util.append(float(out[0]))
            except Exception:
                pass
            self._stop.wait(sample_s)

    @staticmethod
    @contextlib.contextmanager
    def phase(name):
        """Time a phase of the training step (YOLO_PROFILE=2 only; no-op otherwise)."""
        p = TrainProfiler.current
        if p is None or p.level < 2:
            yield
            return
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
        t = time.time()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            p.phases[name] = p.phases.get(name, 0.0) + time.time() - t

    @staticmethod
    def count(name, value):
        p = TrainProfiler.current
        if p is not None and p.level >= 2:
            s, n = p.counters.get(name, (0.0, 0))
            p.counters[name] = (s + float(value), n + 1)

    def epoch_start(self):
        if self.enabled:
            self._t = time.time()
            self._ep = {'data_wait_s': 0.0, 'step_s': 0.0, 'batches': 0}

    def batch_ready(self):
        if self.enabled:
            now = time.time()
            self._ep['data_wait_s'] += now - self._t
            self._t = now

    def batch_done(self):
        if self.enabled:
            now = time.time()
            self._ep['step_s'] += now - self._t
            self._ep['batches'] += 1
            self._t = now

    def epoch_loop_done(self):
        if self.enabled:
            self._t_loop_end = time.time()

    def epoch_done(self):
        if self.enabled:
            self._ep['epoch_end_s'] = time.time() - self._t_loop_end
            self.data_wait += self._ep['data_wait_s']
            self.step += self._ep['step_s']
            self.epoch_end += self._ep['epoch_end_s']
            self.n_batches += self._ep['batches']
            self.n_epochs += 1
            self.epochs.append({k: round(v, 2) if isinstance(v, float) else v for k, v in self._ep.items()})

    def write(self, path):
        if not self.enabled:
            return
        self._stop.set()
        total = time.time() - self.t0
        d = {
            'total_s': round(total, 1), 'epochs': self.n_epochs, 'batches': self.n_batches,
            'data_wait_s': round(self.data_wait, 1), 'step_s': round(self.step, 1), 'epoch_end_s': round(self.epoch_end, 1),
            'data_wait_frac': round(self.data_wait / total, 3) if total else None,
            'gpu_util_mean': round(sum(self._util) / len(self._util), 1) if self._util else None,
            'gpu_util_samples': len(self._util),
            'phases_s': {k: round(v, 1) for k, v in self.phases.items()},
            'counters_mean': {k: round(s / n, 1) for k, (s, n) in self.counters.items() if n},
            'per_epoch': self.epochs}
        TrainProfiler.current = None
        with open(path, 'w') as f:
            json.dump(d, f)
