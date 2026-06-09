"""Direct debug of codecarbon init + start."""
print("[A] patching", flush=True)
import codecarbon.core.util as _u
_u._windows_get_physical_sockets = lambda: 1

print("[B] importing OfflineEmissionsTracker", flush=True)
from codecarbon import OfflineEmissionsTracker

print("[C] constructing tracker", flush=True)
cc = OfflineEmissionsTracker(
    project_name='debug',
    country_iso_code='FRA',
    output_dir='outputs/energy',
    log_level='warning',
    save_to_file=True,
    measure_power_secs=2,
    force_cpu_power=45.0,
    force_ram_power=8.0,
    tracking_mode='process',
    allow_multiple_runs=True,
)
print("[D] tracker constructed OK", flush=True)

print("[E] calling cc.start()", flush=True)
cc.start()
print("[F] cc.start() returned OK", flush=True)

print("[G] sleeping 3s to let it measure", flush=True)
import time
time.sleep(3)

print("[H] calling cc.stop()", flush=True)
e = cc.stop()
print(f"[I] cc.stop() returned: {e} kg CO2eq", flush=True)
print("DONE")
