"""Wrapper minimal autour de codecarbon pour mesurer l'énergie consommée par un run.

Usage typique :

    from src.utils.energy import measure_energy

    with measure_energy(project='opener-conll', region='FRA') as track:
        run_inference()
    print(track.report)
    # {'kwh': 0.0021, 'gco2eq': 0.04, 'seconds': 28.3, 'country': 'FRA'}
"""
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
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {'kwh': round(self.kwh, 6),
                'gco2eq': round(self.gco2eq, 4),
                'seconds': round(self.seconds, 2),
                'country': self.country}


class _Tracker:
    """Petit conteneur pour exposer .report depuis le context manager."""
    def __init__(self):
        self.report = EnergyReport()


@contextmanager
def measure_energy(
    project: str = 'opener',
    region: str = 'FRA',
    output_dir: str | Path = 'outputs/energy',
    log_level: str = 'error',
):
    """Mesure la consommation énergétique d'un bloc de code.

    Args:
        project   : nom de la run (sert au préfixe des fichiers codecarbon).
        region    : ISO 3-letter country code (FRA, USA, DEU…). Détermine le
                    facteur d'émission. Par défaut FRA car notre hardware tourne
                    en France.
        output_dir: où codecarbon écrit son CSV brut.
        log_level : 'debug' | 'info' | 'warning' | 'error' | 'critical'.

    Yields:
        Un objet `tracker` avec attribut `.report` rempli APRÈS sortie du bloc.

    Note : si codecarbon plante (ex: pas de CPU monitoring sur Windows), on
    capture l'exception, on garde la durée brute, et on annote `report.raw['error']`.
    """
    # OfflineEmissionsTracker = pas d'appel API externe, facteur d'émission
    # statique du pays (plus rapide, plus reproductible que EmissionsTracker).
    from codecarbon import OfflineEmissionsTracker

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tracker = _Tracker()
    cc = OfflineEmissionsTracker(
        project_name=project,
        country_iso_code=region,
        output_dir=str(output_dir),
        log_level=log_level,
        save_to_file=True,
        measure_power_secs=1,
    )
    t0 = perf_counter()
    error = None
    try:
        cc.start()
    except Exception as e:
        error = f'tracker.start failed: {e!r}'

    try:
        yield tracker
    finally:
        elapsed = perf_counter() - t0
        gco2eq_kg = 0.0
        try:
            gco2eq_kg = cc.stop() or 0.0
        except Exception as e:
            error = (error + ' | ' if error else '') + f'tracker.stop failed: {e!r}'

        # codecarbon renvoie en kg CO2eq ; on convertit en g
        data = {}
        if hasattr(cc, 'final_emissions_data') and cc.final_emissions_data is not None:
            try:
                data = cc.final_emissions_data.values
            except Exception:
                data = {}

        tracker.report = EnergyReport(
            kwh=float(data.get('energy_consumed', 0.0) or 0.0),
            gco2eq=float((gco2eq_kg or 0.0) * 1000),
            seconds=elapsed,
            country=region,
            raw={**({} if data is None else data),
                 **({'error': error} if error else {})},
        )
