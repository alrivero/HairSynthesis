from __future__ import annotations

import atexit
import faulthandler
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch


def _round_mb(value_bytes: Optional[float]) -> Optional[float]:
    if value_bytes is None:
        return None
    return round(float(value_bytes) / (1024.0 * 1024.0), 2)


def _read_proc_status() -> Dict[str, int]:
    status: Dict[str, int] = {}
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as handle:
            for line in handle:
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                parts = value.strip().split()
                if not parts:
                    continue
                try:
                    status[key] = int(parts[0])
                except ValueError:
                    continue
    except OSError:
        return {}
    return status


def _read_meminfo() -> Dict[str, int]:
    info: Dict[str, int] = {}
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as handle:
            for line in handle:
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                parts = value.strip().split()
                if not parts:
                    continue
                try:
                    info[key] = int(parts[0])
                except ValueError:
                    continue
    except OSError:
        return {}
    return info


def _fd_count() -> Optional[int]:
    try:
        return len(os.listdir('/proc/self/fd'))
    except OSError:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            'shape': list(value.shape),
            'dtype': str(value.dtype),
            'device': str(value.device),
        }
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class RuntimeDiagnostics:
    """Writes structured runtime breadcrumbs to disk for post-mortem debugging."""

    def __init__(
        self,
        *,
        log_dir: str,
        device: Optional[str] = None,
        enabled: bool = True,
        heartbeat_interval_sec: float = 10.0,
        batch_log_every: int = 25,
        trace_batches_every: int = 0,
        enable_faulthandler: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.device = device
        self.enabled = bool(enabled)
        self.heartbeat_interval_sec = max(1.0, float(heartbeat_interval_sec))
        self.batch_log_every = max(0, int(batch_log_every))
        self.trace_batches_every = max(0, int(trace_batches_every))
        self.enable_faulthandler = bool(enable_faulthandler)

        self.events_path = os.path.join(log_dir, 'runtime_events.jsonl')
        self.heartbeat_path = os.path.join(log_dir, 'runtime_heartbeat.json')
        self.traceback_path = os.path.join(log_dir, 'runtime_tracebacks.log')
        self.cuda_memory_path = os.path.join(log_dir, 'runtime_cuda_memory.log')

        self._host = socket.gethostname()
        self._pid = os.getpid()
        self._start_monotonic = time.monotonic()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._event_handle = None
        self._traceback_handle = None
        self._closed = False
        self._excepthook_installed = False
        self._previous_excepthook = None
        self._previous_signal_handlers: Dict[int, Any] = {}
        self._state: Dict[str, Any] = {
            'status': 'initializing',
            'stage': None,
            'epoch': None,
            'phase': None,
            'batch_idx': None,
            'train_batch_step': None,
            'should_visualize': None,
            'last_event': None,
        }

        if not self.enabled:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        self._event_handle = open(self.events_path, 'a', encoding='utf-8', buffering=1)
        self._traceback_handle = open(self.traceback_path, 'a', encoding='utf-8', buffering=1)

        if self.enable_faulthandler:
            try:
                faulthandler.enable(file=self._traceback_handle, all_threads=True)
            except Exception:
                pass

        self._install_signal_handlers()
        self._install_excepthook()
        self._write_traceback_header('diagnostics initialized')
        self.record_event(
            'diagnostics_initialized',
            heartbeat_interval_sec=self.heartbeat_interval_sec,
            batch_log_every=self.batch_log_every,
            trace_batches_every=self.trace_batches_every,
            device=self.device,
        )
        self._write_heartbeat()
        self._start_heartbeat_thread()
        atexit.register(self.close)

    @classmethod
    def from_config(cls, config) -> 'RuntimeDiagnostics':
        train_cfg = getattr(config, 'train', None)
        diagnostics_cfg = getattr(train_cfg, 'diagnostics', None) if train_cfg is not None else None

        enabled = True
        heartbeat_interval_sec = 10.0
        batch_log_every = 25
        trace_batches_every = 0
        enable_faulthandler = True

        if diagnostics_cfg is not None:
            enabled = bool(getattr(diagnostics_cfg, 'enabled', enabled))
            heartbeat_interval_sec = float(
                getattr(diagnostics_cfg, 'heartbeat_interval_sec', heartbeat_interval_sec)
            )
            batch_log_every = int(getattr(diagnostics_cfg, 'batch_log_every', batch_log_every))
            trace_batches_every = int(
                getattr(diagnostics_cfg, 'trace_batches_every', trace_batches_every)
            )
            enable_faulthandler = bool(
                getattr(diagnostics_cfg, 'enable_faulthandler', enable_faulthandler)
            )

        return cls(
            log_dir=config.train.log_path,
            device=getattr(config, 'device', None),
            enabled=enabled,
            heartbeat_interval_sec=heartbeat_interval_sec,
            batch_log_every=batch_log_every,
            trace_batches_every=trace_batches_every,
            enable_faulthandler=enable_faulthandler,
        )

    def should_log_batch(self, batch_idx: int, *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        if force:
            return True
        if self.batch_log_every <= 0:
            return False
        return (int(batch_idx) % self.batch_log_every) == 0

    def should_trace_batch(self, batch_idx: int, *, force: bool = False) -> bool:
        if not self.enabled:
            return False
        if force:
            return True
        if self.trace_batches_every <= 0:
            return False
        return (int(batch_idx) % self.trace_batches_every) == 0

    def set_batch_context(
        self,
        *,
        epoch_idx: Optional[int] = None,
        phase: Optional[str] = None,
        batch_idx: Optional[int] = None,
        train_batch_step: Optional[int] = None,
        should_visualize: Optional[bool] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        with self._lock:
            if epoch_idx is not None:
                self._state['epoch'] = int(epoch_idx)
            if phase is not None:
                self._state['phase'] = str(phase)
            if batch_idx is not None:
                self._state['batch_idx'] = int(batch_idx)
            if train_batch_step is not None:
                self._state['train_batch_step'] = int(train_batch_step)
            if should_visualize is not None:
                self._state['should_visualize'] = bool(should_visualize)
            if stage is not None:
                self._state['stage'] = str(stage)
            if status is not None:
                self._state['status'] = str(status)
            self._write_heartbeat_locked()

    def record_event(self, event: str, **fields: Any) -> None:
        if not self.enabled or self._event_handle is None:
            return

        payload = {
            'timestamp': datetime.now(timezone.utc).astimezone().isoformat(),
            'uptime_sec': round(time.monotonic() - self._start_monotonic, 3),
            'event': event,
            'pid': self._pid,
            'host': self._host,
        }
        payload.update(self._memory_snapshot())
        with self._lock:
            payload.update({key: _json_safe(value) for key, value in self._state.items()})
            payload.update({key: _json_safe(value) for key, value in fields.items()})
            self._state['last_event'] = event
            self._event_handle.write(json.dumps(payload, sort_keys=True) + '\n')
            self._event_handle.flush()
            self._write_heartbeat_locked(extra={'last_event_payload': payload})

    @contextmanager
    def stage(self, name: str, **fields: Any):
        if not self.enabled:
            yield
            return

        started_at = time.perf_counter()
        self.set_batch_context(stage=name, status='running')
        self.record_event(f'{name}:start', **fields)
        try:
            yield
        except Exception as exc:
            duration_sec = round(time.perf_counter() - started_at, 6)
            self.record_event(
                f'{name}:exception',
                duration_sec=duration_sec,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                **fields,
            )
            if 'out of memory' in str(exc).lower():
                self.dump_cuda_memory(label=f'{name}:exception')
            self._write_traceback_header(f'exception in stage {name}')
            self._traceback_handle.write(''.join(traceback.format_exc()))
            self._traceback_handle.flush()
            self.set_batch_context(stage=name, status='exception')
            raise
        else:
            duration_sec = round(time.perf_counter() - started_at, 6)
            self.record_event(f'{name}:end', duration_sec=duration_sec, **fields)
            self.set_batch_context(stage=name, status='idle')

    def dump_cuda_memory(self, *, label: str) -> None:
        if not self.enabled or not torch.cuda.is_available():
            return
        device_index = self._resolve_cuda_device_index()
        if device_index is None:
            return
        try:
            summary = torch.cuda.memory_summary(device=device_index, abbreviated=False)
        except Exception as exc:
            summary = f'Unable to read CUDA memory summary: {exc}'

        try:
            with open(self.cuda_memory_path, 'a', encoding='utf-8') as handle:
                handle.write(f'===== {datetime.now().astimezone().isoformat()} {label} =====\n')
                handle.write(summary)
                handle.write('\n')
        except OSError:
            pass

    def close(self) -> None:
        if not self.enabled or self._closed:
            return

        self._closed = True
        with self._lock:
            self._state['status'] = 'closed'
            self._state['stage'] = 'shutdown'
            self._write_heartbeat_locked()
        self._stop_event.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)

        self.record_event('diagnostics_closed')

        if self._event_handle is not None:
            self._event_handle.close()
            self._event_handle = None
        if self._traceback_handle is not None:
            self._traceback_handle.close()
            self._traceback_handle = None

    def _install_excepthook(self) -> None:
        if self._excepthook_installed:
            return

        self._previous_excepthook = sys.excepthook

        def _hook(exc_type, exc, exc_tb):
            self.record_event(
                'uncaught_exception',
                exception_type=getattr(exc_type, '__name__', str(exc_type)),
                exception_message=str(exc),
            )
            self._write_traceback_header('uncaught exception')
            if self._traceback_handle is not None:
                self._traceback_handle.write(
                    ''.join(traceback.format_exception(exc_type, exc, exc_tb))
                )
                self._traceback_handle.flush()
            if self._previous_excepthook is not None:
                self._previous_excepthook(exc_type, exc, exc_tb)

        sys.excepthook = _hook
        self._excepthook_installed = True

    def _install_signal_handlers(self) -> None:
        signal_names = ('SIGTERM', 'SIGINT', 'SIGHUP', 'SIGABRT')
        for signal_name in signal_names:
            signum = getattr(signal, signal_name, None)
            if signum is None:
                continue
            try:
                previous = signal.getsignal(signum)
                self._previous_signal_handlers[signum] = previous
                signal.signal(signum, self._make_signal_handler(signum, signal_name))
            except (ValueError, OSError, RuntimeError):
                continue

    def _make_signal_handler(self, signum: int, signal_name: str):
        def _handler(received_signum, frame):
            frame_info = None
            if frame is not None:
                frame_info = {
                    'file': frame.f_code.co_filename,
                    'line': frame.f_lineno,
                    'function': frame.f_code.co_name,
                }
            self.record_event(
                'signal_received',
                signal=signal_name,
                signum=received_signum,
                frame=frame_info,
            )
            self.dump_cuda_memory(label=f'signal:{signal_name}')
            self._write_traceback_header(f'signal received: {signal_name}')
            if self._traceback_handle is not None:
                traceback.print_stack(frame, file=self._traceback_handle)
                self._traceback_handle.flush()

            previous = self._previous_signal_handlers.get(signum)
            if callable(previous):
                previous(received_signum, frame)
                return

            if signal_name in {'SIGTERM', 'SIGINT', 'SIGHUP', 'SIGABRT'}:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        return _handler

    def _start_heartbeat_thread(self) -> None:
        if not self.enabled:
            return

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name='runtime-heartbeat',
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_sec):
            self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._write_heartbeat_locked()

    def _write_heartbeat_locked(self, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            'timestamp': datetime.now(timezone.utc).astimezone().isoformat(),
            'uptime_sec': round(time.monotonic() - self._start_monotonic, 3),
            'pid': self._pid,
            'host': self._host,
            **self._memory_snapshot(),
            **{key: _json_safe(value) for key, value in self._state.items()},
        }
        if extra:
            payload.update({key: _json_safe(value) for key, value in extra.items()})

        tmp_path = f'{self.heartbeat_path}.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, sort_keys=True, indent=2)
                handle.write('\n')
            os.replace(tmp_path, self.heartbeat_path)
        except OSError:
            pass

    def _memory_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}

        status = _read_proc_status()
        meminfo = _read_meminfo()

        if 'VmRSS' in status:
            snapshot['rss_mb'] = round(status['VmRSS'] / 1024.0, 2)
        if 'VmHWM' in status:
            snapshot['rss_peak_mb'] = round(status['VmHWM'] / 1024.0, 2)
        if 'VmSize' in status:
            snapshot['vms_mb'] = round(status['VmSize'] / 1024.0, 2)
        if 'Threads' in status:
            snapshot['num_threads'] = status['Threads']
        if 'MemAvailable' in meminfo:
            snapshot['system_mem_available_mb'] = round(meminfo['MemAvailable'] / 1024.0, 2)
        if 'MemFree' in meminfo:
            snapshot['system_mem_free_mb'] = round(meminfo['MemFree'] / 1024.0, 2)

        fd_count = _fd_count()
        if fd_count is not None:
            snapshot['open_fd_count'] = fd_count

        if torch.cuda.is_available():
            device_index = self._resolve_cuda_device_index()
            if device_index is not None:
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                    snapshot['cuda_device_index'] = int(device_index)
                    snapshot['cuda_free_mb'] = _round_mb(free_bytes)
                    snapshot['cuda_total_mb'] = _round_mb(total_bytes)
                    snapshot['cuda_allocated_mb'] = _round_mb(
                        torch.cuda.memory_allocated(device_index)
                    )
                    snapshot['cuda_reserved_mb'] = _round_mb(
                        torch.cuda.memory_reserved(device_index)
                    )
                    snapshot['cuda_max_allocated_mb'] = _round_mb(
                        torch.cuda.max_memory_allocated(device_index)
                    )
                    snapshot['cuda_max_reserved_mb'] = _round_mb(
                        torch.cuda.max_memory_reserved(device_index)
                    )
                except Exception as exc:
                    snapshot['cuda_stats_error'] = str(exc)

        return snapshot

    def _resolve_cuda_device_index(self) -> Optional[int]:
        if not torch.cuda.is_available():
            return None

        if self.device is None:
            try:
                return int(torch.cuda.current_device())
            except Exception:
                return None

        device_str = str(self.device)
        try:
            device = torch.device(device_str)
        except Exception:
            return None

        if device.type != 'cuda':
            return None
        if device.index is not None:
            return int(device.index)
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return None

    def _write_traceback_header(self, title: str) -> None:
        if self._traceback_handle is None:
            return
        self._traceback_handle.write(
            f'\n===== {datetime.now(timezone.utc).astimezone().isoformat()} {title} =====\n'
        )
        self._traceback_handle.flush()
