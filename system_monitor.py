"""
Simple System Resource Monitor
Tracks CPU, GPU, RAM utilization and power consumption
"""

import time
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class SystemMonitor:
    """Background system resource monitor."""
    
    def __init__(self, sampling_interval: float = 1.0, fl_mode: Optional[bool] = None, llm_mode: Optional[bool] = None):
        """
        Args:
            sampling_interval: Time between samples in seconds (default 1.0)
            fl_mode: Boolean indicating if Federated Learning is enabled (optional)
            llm_mode: Boolean indicating if LLM is enabled (optional)
        """
        self.sampling_interval = sampling_interval
        self.fl_mode = fl_mode
        self.llm_mode = llm_mode
        self.stop_flag = False
        self.thread = None
        
        # Timing metrics
        self.start_time = None
        self.stop_time = None
        
        # Storage for metrics
        self.metrics = {
            'cpu_percent': [],
            'ram_percent': [],
            'ram_mb': [],
            'gpu_util': [],
            'gpu_mem_util': [],
            'gpu_mem_mb': [],
            'gpu_power': [],
            'timestamps': []
        }
        
        # GPU setup
        self.gpu_available = False
        self.gpu_count = 0
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = self.gpu_count > 0
                print(f"[Monitor] GPU available: {self.gpu_count} device(s)")
            except Exception as e:
                print(f"[Monitor] GPU init failed: {e}")
    
    def start(self):
        """Start background monitoring."""
        if not PSUTIL_AVAILABLE:
            print("[Monitor] psutil not available, skipping CPU/RAM monitoring")
            return
        
        self.start_time = time.time()
        self.stop_flag = False
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("[Monitor] Started background monitoring")
    
    def stop(self):
        """Stop background monitoring."""
        self.stop_time = time.time()
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=5)
        print("[Monitor] Stopped monitoring")
    
    def _monitor_loop(self):
        """Background loop that samples metrics."""
        try:
            while not self.stop_flag:
                try:
                    timestamp = time.time()
                    
                    # CPU & RAM
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    ram = psutil.virtual_memory()
                    ram_percent = ram.percent
                    ram_mb = ram.used / (1024 * 1024)
                    
                    self.metrics['cpu_percent'].append(cpu_percent)
                    self.metrics['ram_percent'].append(ram_percent)
                    self.metrics['ram_mb'].append(ram_mb)
                    self.metrics['timestamps'].append(timestamp)
                    
                    # GPU (required when available)
                    if self.gpu_available:
                        for gpu_id in range(self.gpu_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                            
                            # Utilization (required)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            self.metrics['gpu_util'].append(util.gpu)
                            self.metrics['gpu_mem_util'].append(util.memory)
                            
                            # Memory (required)
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_mem_mb = mem_info.used / (1024 * 1024)
                            self.metrics['gpu_mem_mb'].append(gpu_mem_mb)
                            
                            # Power (required)
                            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                            power_w = power_mw / 1000.0
                            self.metrics['gpu_power'].append(power_w)
                    
                    time.sleep(self.sampling_interval)
                except Exception as e:
                    pass
        except Exception as e:
            print(f"[Monitor] Error in loop: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics from collected data."""
        def calc_stats(values):
            if not values:
                return {'min': 0.0, 'max': 0.0, 'avg': 0.0, 'count': 0}
            return {
                'min': float(min(values)),
                'max': float(max(values)),
                'avg': float(sum(values) / len(values)),
                'count': len(values)
            }
        
        # Calculate training duration
        duration = None
        if self.start_time and self.stop_time:
            duration = self.stop_time - self.start_time
        
        stats = {
            'cpu_percent': calc_stats(self.metrics['cpu_percent']),
            'ram_percent': calc_stats(self.metrics['ram_percent']),
            'ram_mb': calc_stats(self.metrics['ram_mb']),
        }
        
        if duration is not None:
            minutes = duration / 60.0
            stats['training_time'] = {
                'seconds': float(round(duration, 2)),
                'minutes': float(round(minutes, 2)),
                'hours': float(round(minutes / 60.0, 2))
            }
        
        if self.gpu_available:
            stats['gpu_util'] = calc_stats(self.metrics['gpu_util'])
            stats['gpu_mem_util'] = calc_stats(self.metrics['gpu_mem_util'])
            stats['gpu_mem_mb'] = calc_stats(self.metrics['gpu_mem_mb'])
            if self.metrics['gpu_power']:
                stats['gpu_power_w'] = calc_stats(self.metrics['gpu_power'])
        
        return stats
    
    def save_report(self, output_path: str):
        """Save monitoring report to JSON."""
        stats = self.get_stats()
        report = {
            'timestamp': datetime.now().isoformat(),
            'fl_mode': self.fl_mode,
            'llm_mode': self.llm_mode,
            'system_resources': stats,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[Monitor] Report saved: {output_path}")
        self.print_summary(stats, self.fl_mode, self.llm_mode)
    
    @staticmethod
    def print_summary(stats: Dict, fl_mode: Optional[bool] = None, llm_mode: Optional[bool] = None):
        """Print a summary of resource usage."""
        print("\n" + "=" * 70)
        print("SYSTEM RESOURCE SUMMARY")
        print("=" * 70)
        
        # Training Mode
        if fl_mode is not None:
            mode_str = "Federated Learning (FL)" if fl_mode else "Centralized Learning"
            print(f"\n📊 Training Mode: {mode_str}")
        
        # LLM Mode
        if llm_mode is not None:
            llm_str = "Enabled" if llm_mode else "Disabled"
            print(f"🤖 LLM Mode: {llm_str}")
        
        # Training Time (if available)
        if 'training_time' in stats:
            training = stats['training_time']
            print(f"\n⏱️  Training Time:")
            print(f"  Duration: {training['seconds']:.2f}s ({training['minutes']:.2f}m / {training['hours']:.2f}h)")
        
        cpu = stats.get('cpu_percent', {})
        print(f"\nCPU Usage:")
        print(f"  Min:  {cpu.get('min', 0):.1f}%")
        print(f"  Max:  {cpu.get('max', 0):.1f}%")
        print(f"  Avg:  {cpu.get('avg', 0):.1f}%")
        
        ram_pct = stats.get('ram_percent', {})
        ram_mb = stats.get('ram_mb', {})
        print(f"\nRAM Usage:")
        print(f"  Min:  {ram_pct.get('min', 0):.1f}% ({ram_mb.get('min', 0):.0f} MB)")
        print(f"  Max:  {ram_pct.get('max', 0):.1f}% ({ram_mb.get('max', 0):.0f} MB)")
        print(f"  Avg:  {ram_pct.get('avg', 0):.1f}% ({ram_mb.get('avg', 0):.0f} MB)")
        
        if 'gpu_util' in stats:
            gpu = stats['gpu_util']
            print(f"\nGPU Usage:")
            print(f"  Min:  {gpu.get('min', 0):.1f}%")
            print(f"  Max:  {gpu.get('max', 0):.1f}%")
            print(f"  Avg:  {gpu.get('avg', 0):.1f}%")
            
            gpu_mem = stats.get('gpu_mem_util', {})
            gpu_mem_mb = stats.get('gpu_mem_mb', {})
            print(f"\nGPU Memory:")
            print(f"  Min:  {gpu_mem.get('min', 0):.1f}% ({gpu_mem_mb.get('min', 0):.0f} MB)")
            print(f"  Max:  {gpu_mem.get('max', 0):.1f}% ({gpu_mem_mb.get('max', 0):.0f} MB)")
            print(f"  Avg:  {gpu_mem.get('avg', 0):.1f}% ({gpu_mem_mb.get('avg', 0):.0f} MB)")
            
            if 'gpu_power_w' in stats:
                power = stats['gpu_power_w']
                print(f"\nGPU Power:")
                print(f"  Min:  {power.get('min', 0):.1f}W")
                print(f"  Max:  {power.get('max', 0):.1f}W")
                print(f"  Avg:  {power.get('avg', 0):.1f}W")
        
        print("=" * 70 + "\n")
