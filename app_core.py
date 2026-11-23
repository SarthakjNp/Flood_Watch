"""Flood Watch (research prototype): data and analysis helpers used by streamlit_app.py.
Everything here works offline: it reads the model, the recorded live-data files (fixtures) and the dataset on disk."""
import csv, io, json, os, re, sys, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("FLOOD_BASE", str(Path(__file__).resolve().parent)))
WORK = Path(os.environ.get("FLOOD_WORK", str(BASE / "work")))
FIX = WORK / "fixtures"
ALERTS = WORK / "alerts"
DATA = Path(os.environ.get("FLOOD_ROOT", str(BASE / "Data" / "sen1floods11_v1.1")))
IST = timezone(timedelta(hours=5, minutes=30))

HEAVY, VERY, EXTREME = 64.5, 115.6, 204.5          # IMD daily rainfall classes (mm per day)
PILOT_STATES = ["Assam", "Kerala", "Bihar", "Maharashtra", "West Bengal"]
MAX_UPLOAD_MB = 150
COOLDOWN_HOURS = 12
BANNER = "Research prototype. Not an official warning. Do not use for emergency decisions."
CREDITS = ("Satellite flood labels: Sen1Floods11, Cloud to Street. Forecast and river data: Open-Meteo.com (GloFAS, Copernicus). "
           "Flood reports: GDACS (European Union). Boundaries: Natural Earth. Rainfall values are model-derived, not rain-gauge readings.")
DEFINITIONS = [
    ("Heavy rainfall", "Model-derived rain of 64.5 mm or more in a day at any monitored point (very heavy: 115.6 mm, extremely heavy: 204.5 mm)."),
    ("Elevated river flow", "River discharge at a monitored point above the 90th percentile of its last 61 recorded days."),
    ("Reported flood", "GDACS lists a flood event in or near the state in the period."),
    ("Satellite-detected flood water", "An uploaded satellite image analysed by this project's model."),
]
CLASS_COLOR = {"below heavy": "#9ecae1", "heavy": "#fdae6b", "very heavy": "#e6550d", "extremely heavy": "#a50f15", "no data": "#cccccc"}


# ---------------------------------------------------------------- rain classes
def rain_class(mm):
    if mm is None or (isinstance(mm, float) and np.isnan(mm)):
        return "no data"
    if mm >= EXTREME: return "extremely heavy"
    if mm >= VERY: return "very heavy"
    if mm >= HEAVY: return "heavy"
    return "below heavy"


# ---------------------------------------------------------------- recorded data (fixtures)
def load_fix(name):
    p = FIX / name
    return json.loads(p.read_text()) if p.exists() else None

def rec_date(fix):
    return datetime.fromisoformat(fix["recorded_at"]).astimezone(IST).date()

def rec_datetime(fix):
    return datetime.fromisoformat(fix["recorded_at"]).astimezone(IST)

def probe_frame(fix, state=None):
    cols, meta = {}, []
    for i, p in enumerate(fix["probes"]):
        if state and p.get("state") != state:
            continue
        vals = pd.to_numeric(pd.Series(p["precip_mm"]), errors="coerce").values
        cols[i] = pd.Series(vals, index=pd.to_datetime(p["dates"]).date)
        meta.append(dict(probe=i, lat=p["lat"], lon=p["lon"]))
    if not cols:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(cols).sort_index(), pd.DataFrame(meta).set_index("probe")

def future_rows(df, rec):
    return df[np.array([d >= rec for d in df.index])]

def month_windows(rec):
    cur_start = rec.replace(day=1)
    prev_end = cur_start - timedelta(days=1)
    return {"This month (to date)": (cur_start, rec - timedelta(days=1)), "Last month": (prev_end.replace(day=1), prev_end)}

def month_stats(df, rec):
    mean, mx, rows = df.mean(axis=1), df.max(axis=1), []
    for label, (a, b) in month_windows(rec).items():
        idx = [d for d in df.index if a <= d <= b]
        want = max((b - a).days + 1, 0)
        sub = mx.loc[idx] if idx else pd.Series(dtype=float)
        rows.append({"Period": f"{label}: {a:%d %b} to {b:%d %b}" if want else label, "Days covered": f"{len(idx)} of {want}",
                     "State-mean rain total (mm)": round(float(mean.loc[idx].sum()), 1) if idx else None,
                     "Highest daily value at a point (mm)": round(float(sub.max()), 1) if len(sub) else None,
                     "Heavy days": int(((sub >= HEAVY) & (sub < VERY)).sum()),
                     "Very heavy days": int(((sub >= VERY) & (sub < EXTREME)).sum()),
                     "Extremely heavy days": int((sub >= EXTREME).sum())})
    return pd.DataFrame(rows)

def heavy_days(df, meta, rec):
    rows = []
    obs = df[np.array([d < rec for d in df.index])]
    for d, row in obs.iterrows():
        if row.max() >= HEAVY:
            k = row.idxmax()
            rows.append({"Date": f"{d:%d %b %Y}", "Class": rain_class(float(row.max())), "Highest value (mm)": round(float(row.max()), 1),
                         "Wettest point (lat, lon)": f"{meta.loc[k, 'lat']:.2f}, {meta.loc[k, 'lon']:.2f}"})
    return pd.DataFrame(rows)

def forecast_points(df, meta, rec):
    fut = future_rows(df, rec)
    out = meta.copy()
    for d in fut.index:
        out[f"{d:%d %b} (mm)"] = fut.loc[d].round(1).values
    out["Highest in next 72h (mm)"] = fut.max(axis=0).round(1).values
    out["Class"] = [rain_class(v) for v in fut.max(axis=0).values]
    return out.reset_index(drop=True)

def horizon_summary(df, rec):
    fut, rows = future_rows(df, rec), []
    for h, lab in ((1, "Next 24 h"), (2, "Next 48 h"), (3, "Next 72 h")):
        pm = fut.iloc[:h].max(axis=0)
        rows.append({"Horizon": lab, "Points at or above 64.5 mm": f"{int((pm >= HEAVY).sum())} of {len(pm)}",
                     "Share (%)": round(100 * float((pm >= HEAVY).mean()), 0),
                     "Highest value (mm)": round(float(pm.max()), 1) if pm.notna().any() else None,
                     "Class": rain_class(float(pm.max())) if pm.notna().any() else "no data"})
    return pd.DataFrame(rows)

def national_ranking():
    fix = load_fix("rain_national.json")
    if not fix:
        return pd.DataFrame(), None
    rec, rows = rec_date(fix), []
    for state in sorted({p["state"] for p in fix["probes"]}):
        df, _ = probe_frame(fix, state)
        fut = future_rows(df, rec)
        row = {"State": state, "Points": df.shape[1]}
        for h, lab in ((1, "24h"), (2, "48h"), (3, "72h")):
            pm = fut.iloc[:h].max(axis=0)
            row[f"{lab} share of points >= 64.5 mm (%)"] = round(100 * float((pm >= HEAVY).mean()), 0)
            row[f"{lab} highest (mm)"] = round(float(pm.max()), 1) if pm.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows), rec


# ---------------------------------------------------------------- states, rivers, reports
def find_shp():
    env = os.environ.get("FLOOD_NE_SHP")
    if env and Path(env).exists():
        return Path(env)
    for top in (BASE, Path.home() / "Downloads"):
        if not top.exists():
            continue
        for root, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if d not in (".venv", "sen1floods11_v1.1", "work")]
            for f in files:
                n = f.lower()
                if n.endswith(".shp") and "admin_1" in n and "line" not in n:
                    return Path(root) / f
    return None

def load_states():
    import geopandas as gpd
    shp = find_shp()
    if shp is None:
        raise FileNotFoundError("Natural Earth admin-1 shapefile not found under Flood_Watch\\Data.")
    g = gpd.read_file(shp)
    g.columns = [c.lower() for c in g.columns]
    return g[g["admin"] == "India"].copy().reset_index(drop=True)

def neighbours(gdf, state):
    g = gdf.loc[gdf["name"] == state, "geometry"].iloc[0].buffer(0.02)
    return sorted(n for n, geom in zip(gdf["name"], gdf["geometry"]) if n != state and geom.intersects(g))

def river_table(fix):
    rec, rows, series = rec_date(fix), [], {}
    for r in fix["rivers"]:
        s = pd.Series(pd.to_numeric(pd.Series(r["daily"]["river_discharge"]), errors="coerce").values,
                      index=pd.to_datetime(r["daily"]["time"]).date)
        past, fut = s[[d < rec for d in s.index]].dropna(), s[[d >= rec for d in s.index]].dropna()
        if len(past) < 10:
            continue
        p90, now = float(np.percentile(past, 90)), float(past.iloc[-1])
        peak = float(fut.max()) if len(fut) else now
        rows.append({"Point": r["name"], "Latest (m3/s)": round(now), "Percentile in last 61 days": round(float((past <= now).mean() * 100)),
                     "7-day forecast peak (m3/s)": round(peak), "Elevated river flow": "YES" if (now > p90 or peak > p90) else "no"})
        series[r["name"]] = (s, p90)
    return pd.DataFrame(rows), series

def parse_date(x):
    try:
        t = pd.to_datetime(x, errors="coerce")
        return None if pd.isna(t) else t.date()
    except Exception:
        return None

def events_for_state(events, geom):
    from shapely.geometry import Point
    g = geom.buffer(0.1)
    return [e for e in events if e.get("lat") is not None and e.get("lon") is not None and g.contains(Point(e["lon"], e["lat"]))]

def events_frame(evts):
    rows = [{"Event": e.get("name"), "Alert level": e.get("alertlevel"), "From": str(parse_date(e.get("fromdate")) or ""),
             "To": str(parse_date(e.get("todate")) or ""), "Report": e.get("link") or ""} for e in evts]
    return pd.DataFrame(rows)

def event_counts(evts, rec):
    out = {}
    for label, (a, b) in month_windows(rec).items():
        n = 0
        for e in evts:
            f, t = parse_date(e.get("fromdate")), parse_date(e.get("todate")) or parse_date(e.get("fromdate"))
            if f and t and f <= max(b, a) and t >= a and f <= (rec if label.startswith("This") else b):
                n += 1
        out[label] = n
    return out

def state_context(state, gdf):
    ctx = dict(state=state, geom=gdf.loc[gdf["name"] == state, "geometry"].iloc[0])
    fix = load_fix(f"rain_{state.replace(' ', '_')}.json")
    nat = load_fix("rain_national.json")
    ctx["detail"] = fix is not None
    src = fix or nat
    if not src:
        ctx["empty"] = True
        return ctx
    df, meta = probe_frame(src, None if fix else state)
    if df.empty:
        ctx["empty"] = True
        return ctx
    ctx.update(df=df, meta=meta, rec=rec_date(src), recorded_at=rec_datetime(src))
    ctx["horizons"] = horizon_summary(df, ctx["rec"])
    ctx["points"] = forecast_points(df, meta, ctx["rec"])
    if fix:
        ctx["months"] = month_stats(df, ctx["rec"])
        ctx["heavy_list"] = heavy_days(df, meta, ctx["rec"])
    rv = load_fix(f"rivers_{state.replace(' ', '_')}.json")
    if rv and rv.get("rivers"):
        ctx["rivers"], ctx["river_series"] = river_table(rv)
    gd = load_fix("gdacs_floods_india.json")
    ctx["gdacs_available"] = gd is not None
    ctx["events"] = events_for_state(gd["events"], ctx["geom"]) if gd else []
    rank, _ = national_ranking()
    nb = neighbours(gdf, state)
    ctx["neighbours"] = rank[rank["State"].isin(nb)].reset_index(drop=True) if len(rank) else pd.DataFrame()
    ctx["summary"] = summary_text(ctx)
    return ctx

def summary_text(ctx):
    st, rec = ctx["state"], ctx["rec"]
    parts = []
    if ctx.get("months") is not None:
        m = ctx["months"]; cur, prev = m.iloc[0], m.iloc[1]
        hc = int(cur["Heavy days"] + cur["Very heavy days"] + cur["Extremely heavy days"])
        hp = int(prev["Heavy days"] + prev["Very heavy days"] + prev["Extremely heavy days"])
        parts.append(f"In {st}, heavy rainfall (64.5 mm or more in a day at a monitored location) occurred on {hc} day(s) so far this month "
                     f"({cur['Days covered']} days recorded) and on {hp} day(s) last month ({prev['Days covered']} days recorded).")
    h = ctx["horizons"]
    n3 = h.iloc[2]
    parts.append(f"The recorded forecast for the next 72 hours shows heavy-rain-level values at {n3['Points at or above 64.5 mm']} monitored points "
                 f"(highest {n3['Highest value (mm)']} mm, class: {n3['Class']}).")
    if ctx.get("gdacs_available"):
        n = len(ctx["events"])
        parts.append(f"{n} flood event(s) in or near this state appear in the GDACS list for the last 90 days." if n
                     else "No large flood event in or near this state was found in the GDACS list recorded for the last 90 days.")
    parts.append("Heavy rainfall is not by itself a flood.")
    return " ".join(parts)


# ---------------------------------------------------------------- figures (matplotlib Figure objects, no global state)
def _fig(w=7, h=3.2):
    from matplotlib.figure import Figure
    f = Figure(figsize=(w, h)); return f, f.subplots()

def rain_history_fig(ctx):
    df, rec = ctx["df"], ctx["rec"]
    obs = df[np.array([d < rec for d in df.index])]
    fig, ax = _fig(8, 3.4)
    x = pd.to_datetime(obs.index)
    ax.plot(x, obs.max(axis=1).values, color="#d95f02", label="Highest point")
    ax.plot(x, obs.mean(axis=1).values, color="#1b6ca8", label="State mean")
    top = float(np.nanmax(obs.max(axis=1).values)) if len(obs) else 0
    for v, nm in ((HEAVY, "heavy 64.5"), (VERY, "very heavy 115.6"), (EXTREME, "extremely heavy 204.5")):
        if top * 1.15 >= v:
            ax.axhline(v, color="grey", ls="--", lw=0.8); ax.text(x[0], v, nm, fontsize=7, va="bottom")
    ax.set_ylabel("mm per day"); ax.set_title(f"{ctx['state']}: recorded daily rain, last {len(obs)} days"); ax.legend(fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); return fig

def forecast_fig(ctx):
    fut = future_rows(ctx["df"], ctx["rec"])
    fig, ax = _fig(6, 3.2)
    labels = [f"{d:%d %b}" for d in fut.index]
    mx, mean = fut.max(axis=1).values, fut.mean(axis=1).values
    ax.bar(np.arange(len(labels)) - 0.18, mx, 0.36, color=[CLASS_COLOR[rain_class(v)] for v in mx], label="Highest point", edgecolor="#555")
    ax.bar(np.arange(len(labels)) + 0.18, mean, 0.36, color="#1b6ca8", label="State mean")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels); ax.axhline(HEAVY, color="grey", ls="--", lw=0.8)
    ax.set_ylabel("mm per day"); ax.set_title(f"{ctx['state']}: forecast rain (recorded)"); ax.legend(fontsize=8)
    fig.tight_layout(); return fig

def state_map_fig(ctx, gdf, show_neighbours=True):
    import geopandas as gpd
    fig, ax = _fig(5.5, 5)
    if show_neighbours:
        gdf[gdf["name"].isin(neighbours(gdf, ctx["state"]))].plot(ax=ax, color="#f0f0f0", edgecolor="#999", linewidth=0.5)
    gdf[gdf["name"] == ctx["state"]].plot(ax=ax, color="#ffffff", edgecolor="black", linewidth=1)
    pts = ctx["points"]
    ax.scatter(pts["lon"], pts["lat"], c=[CLASS_COLOR[c] for c in pts["Class"]], s=60, edgecolor="black", linewidth=0.5, zorder=3)
    for cls, col in CLASS_COLOR.items():
        if cls in set(pts["Class"]):
            ax.scatter([], [], c=col, edgecolor="black", s=40, label=cls)
    ax.legend(fontsize=7, title="Highest in next 72 h", title_fontsize=7, loc="lower left")
    minx, miny, maxx, maxy = ctx["geom"].bounds
    ax.set_xlim(minx - 1, maxx + 1); ax.set_ylim(miny - 1, maxy + 1)
    ax.set_title(f"{ctx['state']}: monitored points", fontsize=10); ax.set_aspect("equal"); fig.tight_layout(); return fig

def river_fig(ctx, name):
    s, p90 = ctx["river_series"][name]
    fig, ax = _fig(7, 3.2)
    x = pd.to_datetime(s.index)
    rec = pd.Timestamp(ctx["rec"])
    ax.plot(x[x < rec], s.values[x < rec], color="#1b6ca8", label="recorded")
    ax.plot(x[x >= rec], s.values[x >= rec], color="#d95f02", label="forecast")
    ax.axhline(p90, color="grey", ls="--", lw=0.8); ax.text(x[0], p90, "90th percentile of last 61 days", fontsize=7, va="bottom")
    ax.axvline(rec, color="black", lw=0.6); ax.set_ylabel("river discharge (m3/s)"); ax.set_title(name, fontsize=10); ax.legend(fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); return fig

def fig_png(fig, dpi=160):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); return buf.getvalue()


# ---------------------------------------------------------------- PDF report
def build_state_pdf(ctx, gdf):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape
    ss = getSampleStyleSheet(); small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=7.5, leading=9)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5)
    cap = ParagraphStyle("c", parent=ss["BodyText"], fontSize=8, textColor=colors.grey)

    def table(df, widths=None):
        data = [[Paragraph(f"<b>{escape(str(c))}</b>", small) for c in df.columns]]
        for _, r in df.iterrows():
            data.append([Paragraph(escape("" if pd.isna(v) else str(v)), small) for v in r.values])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    def image(fig, width_cm):
        png = fig_png(fig, 150)
        from PIL import Image as PILImage
        w, h = PILImage.open(io.BytesIO(png)).size
        return Image(io.BytesIO(png), width=width_cm * cm, height=width_cm * cm * h / w)

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7); canvas.setFillColor(colors.grey)
        canvas.drawString(1.5 * cm, 1.0 * cm, BANNER); canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Page {doc.page}"); canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                            title=f"Flood Watch state report: {ctx['state']}")
    E = [Paragraph(f"State situation report: {escape(ctx['state'])}", ss["Title"]),
         Paragraph(f"Generated {datetime.now(IST):%d %b %Y %H:%M} IST from data recorded {ctx['recorded_at']:%d %b %Y %H:%M} IST. "
                   f"<b>{escape(BANNER)}</b>", cap), Spacer(1, 6),
         Paragraph("Summary", ss["Heading2"]), Paragraph(escape(ctx["summary"]), body), Spacer(1, 4),
         Paragraph("Definitions used in this report", ss["Heading3"]),
         table(pd.DataFrame(DEFINITIONS, columns=["Term", "Used only when"]), [4 * cm, 13.5 * cm])]
    if ctx.get("months") is not None:
        E += [Paragraph("This month and last month", ss["Heading2"]), table(ctx["months"]), Spacer(1, 4),
              image(rain_history_fig(ctx), 17)]
        if len(ctx["heavy_list"]):
            E += [Paragraph("Days with heavy-rainfall-level values (not floods)", ss["Heading3"]), table(ctx["heavy_list"].head(25))]
        else:
            E += [Paragraph("No day in the recorded period reached the heavy-rainfall level (64.5 mm) at any monitored point.", body)]
    else:
        E += [Paragraph("Detailed month-by-month history is recorded only for the pilot states.", body)]
    E += [Paragraph("Next 72 hours (recorded forecast)", ss["Heading2"]), table(ctx["horizons"]), Spacer(1, 4),
          image(forecast_fig(ctx), 11), Spacer(1, 4), image(state_map_fig(ctx, gdf), 11)]
    if ctx.get("rivers") is not None and len(ctx["rivers"]):
        E += [Paragraph("River panel", ss["Heading2"]), table(ctx["rivers"]), Spacer(1, 4)]
        first = ctx["rivers"].iloc[0]["Point"]; E += [image(river_fig(ctx, first), 14)]
    E += [Paragraph("Reported flood events (GDACS, last 90 days)", ss["Heading2"])]
    if ctx.get("events"):
        E += [table(events_frame(ctx["events"]))]
    else:
        E += [Paragraph("No large flood event found in the GDACS list for this state. Absence does not prove no flooding.", body)]
    E += [Paragraph("Satellite analysis", ss["Heading2"]),
          Paragraph("No satellite analysis is included in this report. Use the Upload and analyse page for imagery.", body)]
    if len(ctx.get("neighbours", [])):
        E += [Paragraph("Neighbouring states (recorded forecast, coarse grid)", ss["Heading2"]), table(ctx["neighbours"])]
    E += [Paragraph("Method and limitations", ss["Heading2"]),
          Paragraph("Rainfall values come from a weather model through Open-Meteo and typically underestimate gauge extremes. Monitored points are a "
                    "regular grid of about 50 km, so local flash floods can be missed. Flood reports come from GDACS, which lists large events only. "
                    "River values are model-based discharge, not gauge readings. The forecast shown is recorded data, not a live feed, and is not "
                    "produced or validated by this project. Heavy rainfall is not reported as flood unless a report or satellite evidence exists.", body),
          Spacer(1, 4), Paragraph(escape(CREDITS), cap)]
    doc.build(E, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ---------------------------------------------------------------- satellite upload analysis
def get_infer():
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    import flood_infer
    return flood_infer

def load_model():
    meta = json.loads((WORK / "models" / "model_meta.json").read_text())
    fi = get_infer()
    model, device = fi.load_model(WORK / "models" / "unet_fp16.pt", meta)
    return model, device, meta, fi

def _check_tif(path):
    import rasterio
    try:
        src = rasterio.open(path)
    except Exception:
        raise ValueError("The file could not be read as a GeoTIFF. Upload a valid georeferenced .tif with VV and VH bands.")
    with src:
        if src.crs is None:
            raise ValueError("The file has no georeferencing (coordinate system). Upload a georeferenced GeoTIFF.")
        if src.count < 2:
            raise ValueError("The file needs at least 2 bands (VV and VH radar).")

def analyse_bytes(data, name, model, device, meta, fi):
    import rasterio
    if not name.lower().endswith((".tif", ".tiff")):
        raise ValueError("Only .tif or .tiff files are accepted (JPG, PNG and screenshots cannot be analysed).")
    if len(data) > MAX_UPLOAD_MB * 1e6:
        raise ValueError(f"File is larger than {MAX_UPLOAD_MB} MB.")
    fd, path = tempfile.mkstemp(suffix=".tif"); os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(data)
        _check_tif(path)
        res = fi.predict_tif(path, model, device, meta["threshold"])
        with rasterio.open(path) as src:
            x, _ = fi.to_model_input(src.read())
        res["display"] = x[1]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return res

def overlay_fig(display, mask, title="", second=None, second_title=""):
    from matplotlib.figure import Figure
    step = max(1, int(max(display.shape) // 1400))
    d, m = display[::step, ::step], mask[::step, ::step]
    n = 2 if second is None else 3
    fig = Figure(figsize=(5 * n, 5)); axs = fig.subplots(1, n)
    axs[0].imshow(d, cmap="gray", vmin=0, vmax=1); axs[0].set_title("Radar (VH)"); axs[0].axis("off")
    ov = np.zeros(m.shape + (4,), float); ov[m] = (0.1, 0.4, 1.0, 0.6)
    axs[1].imshow(d, cmap="gray", vmin=0, vmax=1); axs[1].imshow(ov); axs[1].set_title(title); axs[1].axis("off")
    if second is not None:
        axs[2].imshow(second[::step, ::step], cmap="gray", vmin=0, vmax=1); axs[2].set_title(second_title); axs[2].axis("off")
    fig.tight_layout(); return fig

def mask_geotiff_bytes(mask, transform, crs):
    import rasterio
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", height=mask.shape[0], width=mask.shape[1], count=1, dtype="uint8", crs=crs, transform=transform) as dst:
            dst.write(mask.astype("uint8"), 1)
        return mem.read()

def compare_pair(before, after):
    if before["mask"].shape != after["mask"].shape or not all(abs(a - b) < 1e-9 for a, b in zip(tuple(before["transform"]), tuple(after["transform"]))):
        raise ValueError("The two images must cover exactly the same area on the same pixel grid (same size and georeferencing).")
    new = after["mask"] & ~before["mask"]
    valid = int(after["stats"]["valid_pixels"]); px = after["stats"]["pixel_area_km2"]
    return dict(mask=new, percent_new=100.0 * int(new.sum()) / max(valid, 1), km2_new=float(int(new.sum()) * px),
                percent_before=before["stats"]["percent_water"], percent_after=after["stats"]["percent_water"])

def sample_chip_list():
    import rasterio
    f = next((DATA / "splits").glob("flood_test_data*"))
    ids = []
    for line in f.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z][A-Za-z\-]*_\d+)", line)
        if m: ids.append(m.group(1))
    rows = []
    for c in ids:
        p = DATA / "data" / "LabelHand" / f"{c}_LabelHand.tif"
        if not p.exists():
            continue
        with rasterio.open(p) as src:
            lab = src.read(1)
        share = float((lab == 1).sum() / max((lab != -1).sum(), 1))
        rows.append(dict(chip=c, water_share=share, india=c.startswith("India")))
    good = [r for r in rows if r["water_share"] >= 0.08]
    rows = good if good else sorted(rows, key=lambda r: -r["water_share"])[:10]
    rows.sort(key=lambda r: (not r["india"], -r["water_share"]))
    return rows

def load_sample(chip, model, device, meta, fi):
    import rasterio
    s1 = DATA / "data" / "S1GRDHand" / f"{chip}_S1Hand.tif"
    res = fi.predict_tif(s1, model, device, meta["threshold"])
    with rasterio.open(s1) as src:
        x, _ = fi.to_model_input(src.read())
    res["display"] = x[1]
    with rasterio.open(DATA / "data" / "LabelHand" / f"{chip}_LabelHand.tif") as src:
        lab = src.read(1)
    h, w = res["mask"].shape
    lab = lab[:h, :w]
    valid = lab != -1
    tp = int((res["mask"] & (lab == 1) & valid).sum()); fp = int((res["mask"] & (lab == 0)).sum()); fn = int((~res["mask"] & (lab == 1)).sum())
    res["label"] = (lab == 1)
    res["iou"] = tp / max(tp + fp + fn, 1)
    jp = DATA / "data" / "JRCWaterHand" / f"{chip}_JRCWaterHand.tif"
    if jp.exists():
        with rasterio.open(jp) as src:
            res["permanent"] = src.read(1)[:h, :w] == 1
    return res


# ---------------------------------------------------------------- advisory outbox (rule-based, no machine learning, offline)
def last_sent(state):
    p = ALERTS / "log.csv"
    if not p.exists():
        return None
    last = None
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["state"] == state and r["decision"] == "approved" and r["mode"] != "simulation":
            last = datetime.fromisoformat(r["time_ist"])
    return last

def make_candidates(simulate_state=None):
    cands = []
    rivers_flag = {}
    for state in PILOT_STATES:
        fix = load_fix(f"rain_{state.replace(' ', '_')}.json")
        if not fix:
            continue
        df, meta = probe_frame(fix); rec = rec_date(fix); at = rec_datetime(fix)
        fut = future_rows(df, rec).astype(float).copy()
        sim = simulate_state == state
        if sim and len(fut) >= 2:
            k = min(3, fut.shape[1]); fut.iloc[0, :k] = 130.0; fut.iloc[1, :k] = 70.0
        day0, two = fut.iloc[0], fut.iloc[:2].max(axis=0)
        n_watch, n_adv = int((two >= HEAVY).sum()), int((day0 >= VERY).sum())
        river_flag, river_note = False, ""
        rv = load_fix(f"rivers_{state.replace(' ', '_')}.json")
        if rv and rv.get("rivers"):
            rt, _ = river_table(rv)
            river_flag = bool(len(rt) and (rt["Elevated river flow"] == "YES").any())
            river_note = ", ".join(rt.loc[rt["Elevated river flow"] == "YES", "Point"].tolist())
        if sim and state == "Assam":
            river_flag, river_note = True, "SIMULATED river rise"
        level = "Advisory" if (n_adv > 0 or river_flag) else ("Watch" if n_watch > 0 else None)
        if level is None:
            continue
        hours = 24 if level == "Advisory" else 48
        top = fut.iloc[:2 if level == "Watch" else 1].max(axis=0)
        k = top.idxmax()
        msg = (f"{'*** SIMULATION, NOT REAL DATA ***' + chr(10) if sim else ''}UNOFFICIAL research-prototype advisory.\n"
               f"State: {state}. Level: {level}. Valid: from {at:%d %b %Y %H:%M} IST for the next {hours} hours.\n"
               + (f"Forecast heavy rain: up to {top.max():.0f} mm in a day ({rain_class(float(top.max()))}), at {int((top >= HEAVY).sum())} of {len(top)} monitored points "
                  f"(wettest near {meta.loc[k, 'lat']:.1f}N, {meta.loc[k, 'lon']:.1f}E).\n" if top.max() >= HEAVY else
                  f"Forecast rain stays below the heavy-rain level (highest {top.max():.0f} mm in a day).\n")
               + (f"Elevated river flow flagged at: {river_note}.\n" if river_flag else "")
               + "This is not an official warning. Check IMD, CWC and your District Disaster Management Authority.")
        ls = last_sent(state)
        cool = (not sim) and ls is not None and (datetime.now(IST).replace(tzinfo=None) - ls).total_seconds() < COOLDOWN_HOURS * 3600
        cands.append(dict(state=state, level=level, message=msg, simulated=sim, cooldown=cool, n_watch=n_watch, n_adv=n_adv,
                          recorded_at=f"{at:%d %b %Y %H:%M} IST"))
    return cands

def record_decision(cand, decision):
    (ALERTS / "outbox").mkdir(parents=True, exist_ok=True)
    ts = datetime.now(IST).replace(tzinfo=None)
    fn = ""
    if decision == "approved":
        fn = f"{ts:%Y%m%d_%H%M%S}_{cand['state'].replace(' ', '_')}_{cand['level']}{'_SIMULATION' if cand['simulated'] else ''}.txt"
        (ALERTS / "outbox" / fn).write_text(cand["message"], encoding="utf-8")
    p = ALERTS / "log.csv"
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_ist", "state", "level", "mode", "decision", "file"])
        w.writerow([ts.isoformat(timespec="seconds"), cand["state"], cand["level"], "simulation" if cand["simulated"] else "recorded-data", decision, fn])
    return fn


# ---------------------------------------------------------------- research results, maps and diagrams (used by the Home and Research pages)
EXP_LABELS = {"e0_otsu_provided": "Otsu baseline (ready-made masks)", "e0_otsu_per_chip": "Otsu (own threshold per chip)",
              "e1_baseline": "U-Net baseline (E1)", "e2a_bce_only": "U-Net, BCE loss only (E2a)",
              "e3_leave_india_out": "U-Net, leave-India-out (E3)", "e5_final": "U-Net tuned (E5)"}
SET_LABELS = {"test": "Full test set", "test_india": "India test chips", "bolivia": "Bolivia (held-out event)", "india_all": "All India chips (E3)"}
PILL_COLOR = {"below heavy": "#4a90b8", "heavy": "#e08a1e", "very heavy": "#d1451b", "extremely heavy": "#8b0d18", "no data": "#888888"}

def results_summary():
    root = WORK / "results"
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for expdir in sorted(root.iterdir()):
        if not expdir.is_dir() or expdir.name.startswith("e4_trial") or expdir.name == "overfit_check":
            continue
        per = {}
        for sd in sorted(expdir.glob("seed*")):
            mj = sd / "metrics.json"
            if mj.exists():
                for name, v in json.loads(mj.read_text())["sets"].items():
                    per.setdefault(name, []).append(v["metrics"])
        for name, lst in per.items():
            row = dict(experiment=expdir.name, label=EXP_LABELS.get(expdir.name, expdir.name), eval_set=name, seeds=len(lst))
            for k in ("iou", "f1", "precision", "recall"):
                vals = [m[k] for m in lst]
                row[k] = float(np.mean(vals)); row[k + "_std"] = float(np.std(vals))
            rows.append(row)
    return pd.DataFrame(rows)

def get_result(df, exp, name):
    if df is None or df.empty:
        return None
    r = df[(df["experiment"] == exp) & (df["eval_set"] == name)]
    return None if r.empty else r.iloc[0]

def exp_bar_fig(df):
    from matplotlib.figure import Figure
    exps = [e for e in ("e0_otsu_provided", "e1_baseline", "e5_final", "e3_leave_india_out") if (df["experiment"] == e).any()]
    sets = [s for s in ("test", "test_india", "bolivia") if (df["eval_set"] == s).any()]
    colors = {"e0_otsu_provided": "#9aa5b1", "e1_baseline": "#4a90b8", "e5_final": "#0b3c5d", "e3_leave_india_out": "#e08a1e"}
    fig = Figure(figsize=(9, 4)); ax = fig.subplots()
    w = 0.8 / max(len(exps), 1)
    for j, e in enumerate(exps):
        xs, ys, es = [], [], []
        for i, s in enumerate(sets):
            r = get_result(df, e, s)
            if r is not None:
                xs.append(i + (j - (len(exps) - 1) / 2) * w); ys.append(r["iou"]); es.append(r["iou_std"])
        ax.bar(xs, ys, w * 0.92, yerr=es, capsize=3, color=colors[e], label=EXP_LABELS[e])
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([SET_LABELS[s] for s in sets], fontsize=9)
    ax.set_ylabel("Water IoU (mean, bars = std over seeds)"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); return fig

def national_map_fig(rank, gdf, col):
    from matplotlib.figure import Figure
    from matplotlib import cm, colors as mcolors
    g = gdf.merge(rank, left_on="name", right_on="State", how="left")
    fig = Figure(figsize=(6, 6.4)); ax = fig.subplots()
    g.plot(ax=ax, color="#e6e9ee", edgecolor="#9aa5b1", linewidth=0.3)
    have = g[g[col].notna()]
    if len(have):
        have.plot(ax=ax, column=col, cmap="YlOrRd", vmin=0, vmax=100, edgecolor="#555", linewidth=0.4)
    sm = cm.ScalarMappable(norm=mcolors.Normalize(0, 100), cmap="YlOrRd")
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Share of points at or above 64.5 mm (%)", fontsize=8)
    ax.set_xlim(67, 98.5); ax.set_ylim(6, 37.5); ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout(); return fig

def architecture_fig():
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch
    fig = Figure(figsize=(12, 4.6)); ax = fig.subplots(); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="#0b3c5d", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="#0b2236")
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#0b3c5d", lw=1.3))
    ax.text(0.1, 4.35, "A. Satellite flood mapping (trained)", fontsize=10, weight="bold", color="#0b3c5d")
    box(0.1, 3.0, 2.1, 1.1, "Sen1Floods11\nradar chips + hand labels", "#dbe9f5"); arrow(2.25, 3.55, 2.75, 3.55)
    box(2.8, 3.0, 2.1, 1.1, "Preprocess\nclip dB, crops, flips", "#dbe9f5"); arrow(4.95, 3.55, 5.45, 3.55)
    box(5.5, 3.0, 2.3, 1.1, "U-Net (ResNet34)\nE0 to E5 experiments", "#bcd8ee"); arrow(7.85, 3.55, 8.35, 3.55)
    box(8.4, 3.0, 1.7, 1.1, "Water model\nunet_fp16.pt", "#bcd8ee"); arrow(10.15, 3.55, 10.5, 3.55)
    box(10.55, 3.0, 1.35, 1.1, "Upload page\n% water, km2", "#d6efe3")
    ax.text(0.1, 2.45, "B. Live situation (recorded feeds, not trained)", fontsize=10, weight="bold", color="#0b3c5d")
    box(0.1, 1.1, 2.1, 1.1, "Open-Meteo\nrain + river forecast", "#fdebd0"); box(2.5, 1.1, 2.1, 1.1, "GDACS\nflood reports", "#fdebd0")
    arrow(2.25, 1.65, 2.45, 1.65); arrow(4.65, 1.65, 5.15, 1.65)
    box(5.2, 1.1, 2.3, 1.1, "Rain classes +\nevidence ladder", "#fdebd0"); arrow(7.55, 1.65, 8.05, 1.65)
    box(8.1, 1.1, 1.9, 1.1, "State report\nPDF + forecast", "#d6efe3"); arrow(10.05, 1.65, 10.45, 1.65)
    box(10.5, 1.1, 1.4, 1.1, "Live watch\nranking", "#d6efe3")
    ax.text(0.1, 0.55, "C. Advisories", fontsize=10, weight="bold", color="#0b3c5d")
    box(5.2, 0.0, 2.3, 0.7, "Rules (Watch / Advisory)", "#f6dede"); arrow(7.55, 0.35, 8.05, 0.35)
    box(8.1, 0.0, 3.8, 0.7, "Human approval, then outbox files (offline)", "#f6dede")
    fig.tight_layout(); return fig

def load_png(path, max_w=1150):
    from PIL import Image as PILImage
    p = Path(path)
    if not p.exists():
        return None
    im = PILImage.open(p).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    return im

def india_gap_text(df):
    a, b = get_result(df, "e1_baseline", "test_india"), get_result(df, "e3_leave_india_out", "test_india")
    if a is None or b is None:
        return ""
    return (f"On the same India test chips, the normal model scored {a['iou']:.3f} and the leave-India-out model (never shown an India chip) scored {b['iou']:.3f}, "
            f"a difference of {a['iou'] - b['iou']:+.3f}. With only a handful of chips this gap is within the noise.")
