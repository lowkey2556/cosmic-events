"""VSX (the International Variable Star Index) look-ups, through astroquery only.

What it does: every VSX query in the project goes through here. Backends, in settings.yaml → vsx.backends
order:
  * vizier  — VizieR catalogue B/vsx/vsx (main server, then its mirrors; see vizier.py);
  * heasarc — NASA HEASARC's copy of VSX (table `aavsovsx`, ~10 million rows) through astroquery's
              TAP interface, used when no VizieR server answers.
The first backend that works is used for the whole run and printed. Results come back with the same
column names whichever backend answered:
  name, ra, dec, type, var_flag (0 = confirmed variable), max, max_band, min, min_band,
  min_is_amplitude (the 'min' value is an amplitude, not a magnitude), period.
AAVSO's website is never scraped.

Why: VizieR was unreachable for more than a day while this was built; the same catalogue is served
by HEASARC, so labels, candidates and the ZTF labelling do not depend on one service.
"""
import time

import pandas as pd

from .vizier import pick_server

COLS = ["name", "ra", "dec", "type", "var_flag", "max", "max_band", "min", "min_band", "min_is_amplitude", "period"]
_backend = {}


def backend(cfg):
    """'vizier' or 'heasarc': the first configured backend that answers (cached for the run)."""
    if "name" in _backend:
        return _backend["name"]
    order = ((cfg.settings.get("vsx") or {}).get("backends") if cfg is not None else None) or ["vizier", "heasarc"]
    errors = []
    for b in order:
        try:
            if b == "vizier":
                v_cfg = (cfg.settings.get("vizier") if cfg is not None else None) or {}
                pick_server(v_cfg.get("servers"), timeout=int(v_cfg.get("probe_timeout_s", 30)))
            else:
                _heasarc("SELECT TOP 1 name FROM aavsovsx WHERE name = 'SS Cyg'")
            _backend["name"] = b
            print(f"VSX: using the {b} backend")
            return b
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{b}: {exc}")
            print(f"VSX backend {b} not available ({type(exc).__name__}); trying the next one")
    raise RuntimeError("No VSX backend answered: " + " | ".join(errors))


def _heasarc(adql, tries=3):
    from astroquery.heasarc import Heasarc
    for i in range(tries):
        try:
            return Heasarc().query_tap(adql).to_table().to_pandas()
        except Exception:  # noqa: BLE001 - retried, then raised
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def _q(s):
    return str(s).replace("'", "''")


def _from_heasarc(d):
    if d.empty:
        return pd.DataFrame(columns=COLS)
    out = pd.DataFrame({
        "name": d["name"].astype(str).str.strip(), "ra": d["ra"].astype(float), "dec": d["dec"].astype(float),
        "type": d["variability_type"].astype(str).str.strip(),
        "var_flag": pd.to_numeric(d.get("variability_flag", 0), errors="coerce"),
        "max": pd.to_numeric(d.get("max_mag"), errors="coerce") if "max_mag" in d else float("nan"),
        "max_band": d.get("max_mag_system", ""), "min": pd.to_numeric(d.get("min_mag"), errors="coerce")
        if "min_mag" in d else float("nan"), "min_band": d.get("min_mag_system", ""),
        "min_is_amplitude": d["min_mag_type"].astype(str).str.strip().eq("Y") if "min_mag_type" in d else False,
        "period": pd.to_numeric(d.get("period"), errors="coerce") if "period" in d else float("nan")})
    return out[COLS]


def _from_vizier(d):
    if d.empty:
        return pd.DataFrame(columns=COLS)
    g = lambda c, default=float("nan"): d[c] if c in d else default
    out = pd.DataFrame({
        "name": d["Name"].astype(str).str.strip(), "ra": d["RAJ2000"].astype(float), "dec": d["DEJ2000"].astype(float),
        "type": d["Type"].astype(str).str.strip(), "var_flag": pd.to_numeric(g("V", 0), errors="coerce"),
        "max": pd.to_numeric(g("max"), errors="coerce"), "max_band": g("n_max", ""),
        "min": pd.to_numeric(g("min"), errors="coerce"), "min_band": g("n_min", ""),
        "min_is_amplitude": g("f_min", "").astype(str).str.strip().eq("(") if "f_min" in d else False,
        "period": pd.to_numeric(g("Period"), errors="coerce")})
    return out[COLS]


def _vizier(cfg, row_limit):
    from .vizier import vizier
    return vizier(cfg, columns=["Name", "V", "RAJ2000", "DEJ2000", "Type", "max", "n_max", "f_min", "min", "n_min",
                                "Period"], row_limit=row_limit)


def by_name(cfg, name):
    """VSX rows whose name is exactly `name` (empty if none)."""
    if backend(cfg) == "heasarc":
        return _from_heasarc(_heasarc("SELECT name, ra, dec, variability_type, variability_flag FROM aavsovsx "
                                      f"WHERE name = '{_q(name)}'"))
    res = _vizier(cfg, 5).query_constraints(catalog="B/vsx/vsx", Name=f"={name}")
    return _from_vizier(res[0].to_pandas()) if len(res) else pd.DataFrame(columns=COLS)


def cone(cfg, ra, dec, radius_deg, types=None):
    """VSX rows within radius of (ra, dec); with `types`, only confirmed variables whose type starts with one of them."""
    if backend(cfg) == "heasarc":
        cond = ""
        if types:
            cond = " AND variability_flag = 0 AND (" + " OR ".join(f"variability_type LIKE '{_q(t)}%'" for t in types) + ")"
        return _from_heasarc(_heasarc(
            "SELECT name, ra, dec, variability_type, variability_flag FROM aavsovsx WHERE "
            f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {float(ra)}, {float(dec)}, {float(radius_deg)})) = 1{cond}"))
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    res = _vizier(cfg, -1).query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius_deg * u.deg, catalog="B/vsx/vsx")
    out = _from_vizier(res[0].to_pandas()) if len(res) else pd.DataFrame(columns=COLS)
    if types:
        out = out[(out["var_flag"] == 0) & out["type"].str.startswith(tuple(types))]
    return out


def by_type(cfg, type_prefix, max_range, dec_min, limit):
    """Confirmed and suspected variables of one type (prefix match) with brightest magnitude in max_range."""
    lo, hi = max_range
    if backend(cfg) == "heasarc":
        return _from_heasarc(_heasarc(
            f"SELECT TOP {int(limit)} name, ra, dec, variability_type, variability_flag, max_mag, max_mag_system, "
            "min_mag, min_mag_type, min_mag_system, period FROM aavsovsx "
            f"WHERE variability_type LIKE '{_q(type_prefix)}%' AND max_mag BETWEEN {lo} AND {hi} AND dec > {dec_min}"))
    res = _vizier(cfg, int(limit)).query_constraints(catalog="B/vsx/vsx", Type=f"{type_prefix}*", max=f"{lo}..{hi}",
                                                     DEJ2000=f">{dec_min}")
    return _from_vizier(res[0].to_pandas()) if len(res) else pd.DataFrame(columns=COLS)
