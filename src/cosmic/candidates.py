"""Phase 1 — Faint-star candidates for more AAVSO downloads (VSX via astroquery only; AAVSO is never scraped).

What it does: queries the VSX catalogue on VizieR (B/vsx/vsx), one primary type at a time, for stars
  * whose primary type is in our label map (anomalous or normal),
  * whose brightest magnitude ('max') is between candidates.max_mag_range (default 12.5–16, so ZTF
    does not saturate on them) and whose faintest magnitude ('min', a real magnitude, not an
    amplitude) is no fainter than candidates.min_mag_limit (default 17.5): they sit mostly in ~13–17,
  * at Dec > candidates.dec_min (default −25°, the sky ZTF covers),
  * flagged as a proven variable in VSX (V = 0),
  * not already in our AAVSO data (by name, and by position within 10″).
It then picks about candidates.n stars (default 40), half anomalous and half normal, spreading the
picks across types, and ranking first: stars with a GCVS-style name (e.g. "SS Cyg", "V1504 Cyg"),
which are far more likely to have AAVSO observations than survey designations (CSS, ASASSN, …),
and stars inside the ZTF fields chosen in ztf_select (so ZTF can also measure them); within that,
stars whose middle magnitude is closest to 15.
Output: V3_ROOT/outputs/aavso_candidate_stars.csv (name, class, VSX type, magnitude range and
passbands, period, RA/Dec, in chosen ZTF field). Download Jan 2022 – Jan 2025 for each star from
the AAVSO website yourself and put the CSVs directly in PROJECT_ROOT/data/; the pipeline picks them up.
"""
import os
import re
import time

import numpy as np
import pandas as pd

GCVS_NAME = re.compile(r"^(V\d{3,4}|[A-Z]{1,2}|[a-z]{3}) [A-Z][A-Za-z]{2}$")


def balanced_pick(cands, n):
    """Half anomalous, half normal; within each class, round-robin over types in rank order."""
    out = []
    for label, k in ((1, n // 2), (0, n - n // 2)):
        c = cands[cands["label"] == label]
        queues = [g.index.tolist() for _, g in sorted(c.groupby("primary_type"), key=lambda kv: kv[1]["rank"].min())]
        picked = []
        while len(picked) < k and any(queues):
            for qu in queues:
                if qu and len(picked) < k:
                    picked.append(qu.pop(0))
        out += picked
    return cands.loc[out]


def query_type(v, t, c):
    lo, hi = c["max_mag_range"]
    res = v.query_constraints(catalog="B/vsx/vsx", Type=f"{t}*", max=f"{lo}..{hi}", DEJ2000=f">{c['dec_min']}")
    return res[0].to_pandas() if len(res) else pd.DataFrame()


def run(cfg):
    from astroquery.vizier import Vizier

    from .labels import norm_name, primary_type
    from .ztf import aavso_stars, ang_sep_deg, http_get, parse_field_centres
    c = cfg.section("candidates")
    lm = cfg.load_yaml("label_map")
    seps = lm.get("type_separators")
    ours = aavso_stars(cfg)
    our_names = set(ours["data_name"].map(norm_name)) | set(ours["name"].dropna().map(norm_name))
    v = Vizier(columns=["Name", "V", "RAJ2000", "DEJ2000", "Type", "max", "n_max", "f_min", "min", "n_min", "Period"],
               row_limit=int(c["per_type_query_limit"]))
    frames = []
    for label, types in ((1, lm["anomalous"]), (0, lm["normal"])):
        for t in types:
            try:
                d = query_type(v, t, c)
            except Exception as exc:  # noqa: BLE001 - reported, type skipped
                print(f"  VSX query failed for {t}: {exc}")
                continue
            if len(d):
                d["primary_type"] = d["Type"].map(lambda s: primary_type(s, seps))
                d = d[d["primary_type"] == t].assign(label=label)
                frames.append(d)
            print(f"  {t:5s}: {len(d)} stars in range", flush=True)
            time.sleep(1.0)
    cands = pd.concat(frames, ignore_index=True)
    min_is_mag = cands["f_min"].fillna("").astype(str).str.strip() != "("
    # 'min' must be a real magnitude: not an amplitude ('(' flag) and fainter (larger) than 'max'
    cands = cands[(cands["V"] == 0) & min_is_mag & cands["min"].notna() & (cands["min"] > cands["max"])
                  & (cands["min"] <= c["min_mag_limit"])]
    cands = cands[~cands["Name"].map(norm_name).isin(our_names)]
    # novae named after a year after the download window (e.g. "N Sgr 2026") have no data in it
    year = cands["Name"].str.extract(r"^N \w+ (\d{4})")[0].astype(float)
    cands = cands[~(year > int(c["download_end_year"]))]
    if ours["ra"].notna().any():
        o = ours.dropna(subset=["ra", "dec"])
        d_ours = np.array([ang_sep_deg(o["ra"].to_numpy(), o["dec"].to_numpy(), r, de).min() * 3600
                           for r, de in zip(cands["RAJ2000"], cands["DEJ2000"])]) if len(cands) else np.array([])
        cands = cands[d_ours > 10]
    in_ztf = pd.Series(False, index=cands.index)
    fields_file = os.path.join(cfg.output_dir, "ztf_fields_chosen.csv")
    if os.path.exists(fields_file) and len(cands):
        chosen = pd.read_csv(fields_file)["field"]
        centres = parse_field_centres(http_get(cfg.section("ztf")["fields_url"]).decode())
        cen = centres[centres["field"].isin(chosen)]
        in_ztf = pd.Series([bool((ang_sep_deg(cen["ra"].to_numpy(), cen["dec"].to_numpy(), r, de) < 3.5).any())
                            for r, de in zip(cands["RAJ2000"], cands["DEJ2000"])], index=cands.index)
    cands = cands.assign(in_chosen_ztf_field=in_ztf, gcvs_name=cands["Name"].str.match(GCVS_NAME))
    cands["rank"] = (~cands["gcvs_name"]).astype(int) * 2 + (~cands["in_chosen_ztf_field"]).astype(int)
    # within a rank, prefer stars whose middle magnitude is closest to 15 (the centre of the ~13-17 band)
    cands["mid_mag_offset"] = ((cands["max"] + cands["min"]) / 2 - 15.0).abs()
    cands = cands.sort_values(["rank", "mid_mag_offset"])
    pick = balanced_pick(cands, int(c["n"]))
    out = pick.assign(**{"class": pick["label"].map({1: "anomalous", 0: "normal"})}).rename(columns={
        "Name": "name", "Type": "vsx_type", "max": "mag_brightest", "n_max": "band_brightest", "min": "mag_faintest",
        "n_min": "band_faintest", "Period": "period_days", "RAJ2000": "ra_deg", "DEJ2000": "dec_deg"})
    cols = ["name", "class", "vsx_type", "primary_type", "mag_brightest", "band_brightest", "mag_faintest",
            "band_faintest", "period_days", "ra_deg", "dec_deg", "in_chosen_ztf_field", "gcvs_name"]
    out = out[cols].reset_index(drop=True)
    path = os.path.join(cfg.output_dir, "aavso_candidate_stars.csv")
    out.to_csv(path, index=False)
    print(f"\n{len(cands)} VSX stars passed the filters; picked {len(out)} "
          f"({int((out['class'] == 'anomalous').sum())} anomalous, {int((out['class'] == 'normal').sum())} normal)")
    print(out.to_string(index=False))
    print(f"saved {path}\nDownload Jan 2022 - Jan 2025 for each from the AAVSO website and put the CSVs in {cfg.data_dir}")
    return out
