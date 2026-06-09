"""Wrapper de mesure d'énergie : codecarbon avec fallback estimation.

Stratégie hybride pour gérer les bugs Windows de codecarbon :

  1. On TENTE d'initialiser codecarbon (OfflineEmissionsTracker) dans un
     thread avec timeout 15s.
  2. Si l'init réussit → on mesure via codecarbon (CPU forced + GPU NVML).
     Précision typique ±5%.
  3. Si l'init hang ou plante → fallback estimation :
        kWh   = elapsed × estimated_power_watts / 3600 / 1000
        gCO2  = kWh × grid_emission_factor
     Précision typique ±20% sur l'absolu, mais les RANKINGS entre modèles
     sont préservés (même hardware, même formule).

Le `report.raw['method']` indique quelle méthode a été utilisée (`codecarbon`
ou `tdp_estimate`), à inclure dans le tableau du papier pour la transparence.

Pour le papier : citer codecarbon (Lacoste et al., 2019) ET documenter la
méthode hybride avec son rationale. C'est l'approche scientifiquement la plus
honnête : on utilise l'outil standard quand il marche, et on fournit une
estimation reproductible quand il ne marche pas.

Usage typique :

    from src.utils.energy import measure_energy

    with measure_energy(project='opener-conll') as track:
        run_inference()
    print(track.report.as_dict())
    # {'kwh': 0.0021, 'gco2eq': 0.11, 'seconds': 28.3, 'country': 'FRA',
    #  'method': 'codecarbon'}     # or 'tdp_estimate'
"""
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter


@dataclass
class EnergyReport:
    kwh: float = 0.0
    gco2eq: float = 0.0
    seconds: float = 0.0
    country: str = ''
    method: str = 'unknown'
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {'kwh': round(self.kwh, 6),
                'gco2eq': round(self.gco2eq, 4),
                'seconds': round(self.seconds, 2),
                'country': self.country,
                'method': self.method}


class _Tracker:
    def __init__(self):
        self.report = EnergyReport()


# Grid emission factors (gCO2eq / kWh, sources publiques 2024).
_GRID_EMISSION_FACTORS = {
    'FRA': 52,    # France (nuclear-heavy)
    'DEU': 380,
    'USA': 380,
    'GBR': 250,
    'CHN': 530,
    'WORLD': 475,
}


def _try_init_codecarbon(project, region, output_dir, log_level,
                          force_cpu_power, force_ram_power, tracking_mode,
                          timeout_s=15):
    """Tente d'initialiser codecarbon dans un thread avec timeout.

    Returns:
        (tracker, error_msg) — tracker est None si l'init a échoué/timeout.
    """
    # Monkey-patch en premier pour court-circuiter le subprocess PowerShell.
    try:
        import codecarbon.core.util as _cu
        _cu._windows_get_physical_sockets = lambda: 1
    except Exception:
        pass

    result = {'tracker': None, 'error': None}

    def _init():
        try:
            from codecarbon import OfflineEmissionsTracker
            cc = OfflineEmissionsTracker(
                project_name=project,
                country_iso_code=region,
                output_dir=str(output_dir),
                log_level=log_level,
                save_to_file=True,
                measure_power_secs=2,
                force_cpu_power=force_cpu_power,
                force_ram_power=force_ram_power,
                tracking_mode=tracking_mode,
                allow_multiple_runs=True,
            )
            cc.start()
            result['tracker'] = cc
        except Exception as e:
            result['error'] = f'codecarbon init failed: {e!r}'

    t = threading.Thread(target=_init, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        result['error'] = f'codecarbon init timed out after {timeout_s}s'
        # Le thread continue en background (daemon → meurt avec le process).
    return result['tracker'], result['error']


@contextmanager
def measure_energy(
    project: str = 'opener',
    region: str = 'FRA',
    output_dir: str | Path = 'outputs/energy',
    log_level: str = 'error',
    force_cpu_power: float = 45.0,
    force_ram_power: float = 8.0,
    tracking_mode: str = 'process',
    estimated_power_watts: float = 105.0,    # for the fallback TDP estimate
    init_timeout_s: float = 15.0,
):
    """Mesure l'énergie d'un bloc de code.

    Args:
        project              : nom de la run.
        region               : ISO 3-letter country code (FRA, USA, ...).
        force_cpu_power      : TDP CPU forcé pour codecarbon (W).
        force_ram_power      : TDP RAM forcé pour codecarbon (W).
        tracking_mode        : 'process' ou 'machine' (codecarbon).
        estimated_power_watts: TDP système total pour le fallback (W). Défaut
                                105W = ~45W CPU + ~60W GPU pour le hardware
                                Opener (Ryzen 9 4000 + GTX 1660 Ti Max-Q).
        init_timeout_s       : si codecarbon n'est pas initialisé dans ce
                                délai, on bascule en fallback TDP.

    Yields:
        Un `tracker` avec attribut `.report` rempli APRÈS sortie du bloc.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tracker = _Tracker()

    cc, error = _try_init_codecarbon(
        project, region, output_dir, log_level,
        force_cpu_power, force_ram_power, tracking_mode, init_timeout_s,
    )

    t0 = perf_counter()
    try:
        yield tracker
    finally:
        elapsed = perf_counter() - t0

        if cc is not None:
            # Cas nominal : codecarbon a démarré → on récupère ses mesures
            gco2eq_kg = 0.0
            try:
                gco2eq_kg = cc.stop() or 0.0
            except Exception as e:
                error = (error + ' | ' if error else '') + f'cc.stop failed: {e!r}'

            data: dict = {}
            try:
                if cc.final_emissions_data is not None:
                    data = cc.final_emissions_data.values
            except Exception:
                data = {}

            tracker.report = EnergyReport(
                kwh=float(data.get('energy_consumed', 0.0) or 0.0),
                gco2eq=float((gco2eq_kg or 0.0) * 1000),
                seconds=elapsed,
                country=region,
                method='codecarbon',
                raw={**({} if not data else data),
                     **({'init_error': error} if error else {})},
            )
        else:
            # Fallback : estimation TDP × temps
            grid_factor = _GRID_EMISSION_FACTORS.get(
                region, _GRID_EMISSION_FACTORS['WORLD']
            )
            kwh = (estimated_power_watts * elapsed) / 3600.0 / 1000.0
            gco2eq = kwh * grid_factor
            tracker.report = EnergyReport(
                kwh=kwh,
                gco2eq=gco2eq,
                seconds=elapsed,
                country=region,
                method='tdp_estimate',
                raw={'estimated_power_watts': estimated_power_watts,
                     'grid_emission_factor_gco2_per_kwh': grid_factor,
                     'reason': error or 'codecarbon disabled'},
            )
