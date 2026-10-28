"""VizieR access with automatic fall-back to the official mirror servers.

What it does: VSX (catalogue B/vsx/vsx) is read through VizieR. The main server
(vizier.cds.unistra.fr) is sometimes unreachable from Colab (connection time-outs). `vizier()`
tries the servers in settings.yaml → vizier.servers in order with one tiny test query and returns
an astroquery Vizier object bound to the first server that answers; the choice is remembered for
the rest of the run. If none answers, it stops with a clear message instead of failing on every query.

Why: the same catalogue is served by mirrors (Harvard CfA, IUCAA India, NAOJ Japan, ...), so one
server being down should not stop label lookups or the candidate search.
"""
import time

DEFAULT_SERVERS = ["vizier.cds.unistra.fr", "vizier.cfa.harvard.edu", "vizier.iucaa.in", "vizier.nao.ac.jp"]
_working = {}


def pick_server(servers=None, timeout=30):
    """First server that answers a one-row VSX query (cached for the run)."""
    from astroquery.vizier import Vizier
    servers = list(servers or DEFAULT_SERVERS)
    key = tuple(servers)
    if key in _working:
        return _working[key]
    tried = []
    for srv in servers:
        t0 = time.time()
        try:
            v = Vizier(columns=["Name"], row_limit=1, vizier_server=srv)
            v.TIMEOUT = timeout
            r = v.query_constraints(catalog="B/vsx/vsx", Name="=SS Cyg")
            if len(r):
                print(f"VizieR: using {srv} (answered in {time.time() - t0:.1f} s)")
                _working[key] = srv
                return srv
            tried.append(f"{srv}: empty answer")
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            tried.append(f"{srv}: {type(exc).__name__}")
            print(f"VizieR server {srv} not reachable ({type(exc).__name__}); trying the next mirror")
    raise RuntimeError("No VizieR server answered (" + "; ".join(tried) + "). The service may be down or blocked "
                       "from this machine; wait a few minutes and run the cell again.")


def vizier(cfg=None, timeout=120, **kwargs):
    """astroquery Vizier(**kwargs) bound to a working server (settings: vizier.servers, vizier.timeout_s)."""
    from astroquery.vizier import Vizier
    v_cfg = (cfg.settings.get("vizier") if cfg is not None else None) or {}
    srv = pick_server(v_cfg.get("servers"), timeout=int(v_cfg.get("probe_timeout_s", 30)))
    v = Vizier(vizier_server=srv, **kwargs)
    v.TIMEOUT = int(v_cfg.get("timeout_s", timeout))
    return v
