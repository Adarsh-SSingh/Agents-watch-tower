import psutil
import platform
import time
from datetime import datetime, timezone


def get_system_info(system_info=None):

    # CPU information
    cpu_info = {
        "usage_percent": psutil.cpu_percent(interval=0.5),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "frequency_mhz": (
            psutil.cpu_freq().current
            if psutil.cpu_freq() else None
        )
    }

    # Memory information
    memory = psutil.virtual_memory()

    memory_info = {
        "total_gb": round(memory.total / (1024 ** 3), 2),
        "used_gb": round(memory.used / (1024 ** 3), 2),
        "available_gb": round(memory.available / (1024 ** 3), 2),
        "usage_percent": memory.percent
    }

    # Disk information
    disk = psutil.disk_usage("/" if platform.system() != "Windows" else "C:\\")

    disk_info = {
        "total_gb": round(disk.total / (1024 ** 3), 2),
        "used_gb": round(disk.used / (1024 ** 3), 2),
        "free_gb": round(disk.free / (1024 ** 3), 2),
        "usage_percent": disk.percent
    }

    # Network information
    network = psutil.net_io_counters()

    network_info = {
        "bytes_sent_mb": round(network.bytes_sent / (1024 ** 2), 2),
        "bytes_received_mb": round(network.bytes_recv / (1024 ** 2), 2),
        "packets_sent": network.packets_sent,
        "packets_received": network.packets_recv,
        "errors_in": network.errin,
        "errors_out": network.errout,
        "dropped_in": network.dropin,
        "dropped_out": network.dropout
    }

    # System uptime
    boot_timestamp = psutil.boot_time()
    uptime_seconds = time.time() - boot_timestamp

    uptime_info = {
        "boot_time": datetime.fromtimestamp(
            boot_timestamp, tz=timezone.utc
        ).isoformat(),
        "uptime_hours": round(uptime_seconds / 3600, 2)
    }

    # Process information
    process_info = {
        "running_processes": len(psutil.pids())
    }

    return {
        "system": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor()
        },
        "cpu": cpu_info,
        "memory": memory_info,
        "disk": disk_info,
        "network": network_info,
        "uptime": uptime_info,
        "processes": process_info,
        "collected_at": datetime.now(timezone.utc).isoformat()
    }