"""Flood Watch: flood monitoring platform (local Streamlit app with a top navigation bar)."""
import os
from datetime import datetime
import pandas as pd
import streamlit as st
import app_core as core

# ------------------------------------------------------------------ EDIT THESE (shown on the About page)
PROJECT = dict(
    title="Flood Watch",
    subtitle="Monitor rainfall, river conditions, reported floods and satellite flood mapping across India",
    student="Sarthak Jain", guide="Ms. Neha Sehgal",
    institute="Delhi Technical Campus, Greater Noida",
    programme="B.Tech (AI-DS)",
    tech=("Python, PyTorch and segmentation-models-pytorch (U-Net), rasterio, geopandas, matplotlib, reportlab and Streamlit. "
          "Dataset: Sen1Floods11, Open-Meteo and GDACS."),
)

st.set_page_config(page_title="Flood Watch", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
#MainMenu, footer, .stAppDeployButton {visibility: hidden;}
.fw-hero {background: linear-gradient(120deg, #0b3c5d 0%, #1d6fa5 60%, #2fa4c7 100%); color: #fff; padding: 26px 30px; border-radius: 14px; margin-bottom: 16px;}
.fw-hero h1 {margin: 0 0 6px 0; font-size: 36px; color: #fff;}
.fw-hero p {margin: 0 0 12px 0; font-size: 16.5px; opacity: .95; color: #fff;}
.fw-badge {display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); padding: 3px 12px; border-radius: 999px; font-size: 12.5px; margin: 0 6px 6px 0; color: #fff;}
.fw-pagehead h2 {margin: 0; color: #0b3c5d;}
.fw-pagehead p {margin: 2px 0 14px 0; color: #667085;}
.fw-card {background: #ffffff; border: 1px solid #e3e8ee; border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(16,24,40,.07); height: 100%;}
.fw-label {font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; color: #667085;}
.fw-metric {font-size: 28px; font-weight: 700; color: #0b3c5d; line-height: 1.25;}
.fw-note {font-size: 12.5px; color: #667085;}
.fw-text {background: #ffffff; border: 1px solid #e3e8ee; border-left: 5px solid #1d6fa5; border-radius: 10px; padding: 14px 18px; color: #1f2933; line-height: 1.55;}
.fw-pill {display: inline-block; padding: 2px 11px; border-radius: 999px; font-size: 12px; color: #fff; margin-right: 6px;}
.fw-feature {background: #ffffff; border: 1px solid #e3e8ee; border-radius: 12px; padding: 14px 16px; height: 100%; color: #1f2933;}
.fw-feature b {color: #0b3c5d;}
.fw-footer {margin-top: 32px; padding: 14px 18px; border-top: 1px solid #d0d5dd; font-size: 12.5px; color: #667085;}
.st-key-nav div[role="radiogroup"] {gap: 6px; background: #0b3c5d; padding: 8px 12px; border-radius: 10px; flex-wrap: wrap;}
.st-key-nav label {background: transparent; padding: 6px 14px; border-radius: 8px; cursor: pointer;}
.st-key-nav label > div:first-child {display: none;}
.st-key-nav label p {color: #ffffff; margin: 0; font-weight: 600;}
.st-key-nav label:has(input:checked) {background: #1d9bd1;}
</style>
"""

@st.cache_resource
def get_states():
    return core.load_states()

@st.cache_resource
def get_model():
    return core.load_model()

@st.cache_data
def get_samples():
    return core.sample_chip_list()

# ------------------------------------------------------------------ small UI helpers
def card(label, value, note="", accent="#0b3c5d"):
    return (f"<div class='fw-card' style='border-top:4px solid {accent}'><div class='fw-label'>{label}</div>"
            f"<div class='fw-metric'>{value}</div><div class='fw-note'>{note}</div></div>")

def pill(cls):
    return f"<span class='fw-pill' style='background:{core.PILL_COLOR.get(cls, '#888')}'>{cls}</span>"

def page_header(title, desc):
    st.markdown(f"<div class='fw-pagehead'><h2>{title}</h2><p>{desc}</p></div>", unsafe_allow_html=True)

def text_box(html):
    st.markdown(f"<div class='fw-text'>{html}</div>", unsafe_allow_html=True)

def recorded_notice():
    fix = core.load_fix("rain_national.json") or core.load_fix("rain_Assam.json")
    if fix:
        st.info(f"Data shown was recorded on {core.rec_datetime(fix):%d %b %Y at %H:%M} IST. Forecast values are as of that time.")
    else:
        st.warning("No recorded data found. Run notebook 2 (record data) while connected to the internet.")

def footer():
    st.markdown(f"<div class='fw-footer'><b>{core.BANNER}</b><br>{core.CREDITS}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ pages
def page_overview():
    st.markdown(f"""
    <div class='fw-hero'>
      <h1>{PROJECT['title']}</h1>
      <p>{PROJECT['subtitle']}</p>
      <span class='fw-badge'>Rain forecast</span><span class='fw-badge'>River conditions</span>
      <span class='fw-badge'>Reported floods</span><span class='fw-badge'>Satellite flood mapping</span><span class='fw-badge'>State reports</span>
    </div>""", unsafe_allow_html=True)
    recorded_notice()
    rank, rec = core.national_ranking()
    if rank.empty:
        st.stop()
    gdf = get_states()
    col, hi = "72h share of points >= 64.5 mm (%)", "72h highest (mm)"
    show = rank.sort_values([col, hi], ascending=False).reset_index(drop=True)
    n_heavy = int((rank[col] > 0).sum())
    wet = rank.sort_values(hi, ascending=False).iloc[0]
    gd = core.load_fix("gdacs_floods_india.json")
    c = st.columns(4)
    c[0].markdown(card("Next 72 hours", n_heavy, f"of {len(rank)} states have a point at heavy-rain level", "#e08a1e"), unsafe_allow_html=True)
    c[1].markdown(card("Highest forecast rain", f"{wet[hi]:.0f} mm", f"in a day, in {wet['State']}", "#d1451b"), unsafe_allow_html=True)
    c[2].markdown(card("Reported floods", len(gd["events"]) if gd else "n/a", "large events listed by GDACS in India, last 90 days", "#1d6fa5"), unsafe_allow_html=True)
    c[3].markdown(card("Data recorded", f"{core.rec_datetime(core.load_fix('rain_national.json')):%d %b}", "see the Monitor page for details", "#2a9d8f"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### Forecast rain exposure, next 72 hours")
        st.pyplot(core.national_map_fig(rank, gdf, col))
    with right:
        st.markdown("#### States to watch")
        st.dataframe(show[["State", col, hi]].head(8).rename(columns={col: "Points at heavy-rain level (%)", hi: "Highest rain (mm/day)"}), hide_index=True)
        text_box("Heavy rainfall is not by itself a flood. A flood is named only when a report or satellite evidence exists. "
                 "Open <b>State Report</b> for a full picture of any state.")
    st.markdown("#### What you can do here")
    feats = [("Monitor", "Rank states by forecast heavy-rain exposure for the next 24, 48 and 72 hours, with a national map."),
             ("State Report", "Forecast, rain history, river panel, reported floods and neighbouring states for any state, with a downloadable PDF."),
             ("Satellite Analysis", "Upload a Sentinel-1 radar image to map water, or compare before and after an event to find newly flooded areas."),
             ("Advisories", "Review rule-based Watch and Advisory messages and approve them for the local outbox. Nothing is sent online.")]
    cols = st.columns(4)
    for k, (t, d) in enumerate(feats):
        cols[k].markdown(f"<div class='fw-feature'><b>{t}</b><br>{d}</div>", unsafe_allow_html=True)
    st.write("")
    st.caption("Use the navigation bar at the top to move between pages.")


def show_result(res, fi, title, key):
    s = res["stats"]
    c = st.columns(4)
    c[0].markdown(card("Water in image", f"{s['percent_water']:.1f}%", "permanent water included", "#1d6fa5"), unsafe_allow_html=True)
    c[1].markdown(card("Water area", f"{s['water_km2']:.2f} sq km", "from pixel size and location", "#1d6fa5"), unsafe_allow_html=True)
    c[2].markdown(card("Area analysed", f"{s['valid_pixels'] * s['pixel_area_km2']:.1f} sq km", "no-data pixels excluded", "#9aa5b1"), unsafe_allow_html=True)
    c[3].markdown(card("Water coverage level", fi.severity(s["percent_water"]), "minor, moderate, major or severe", "#e08a1e"), unsafe_allow_html=True)
    st.write("")
    fig = core.overlay_fig(res["display"], res["mask"], title)
    st.pyplot(fig)
    d1, d2 = st.columns(2)
    d1.download_button("Download overlay (PNG)", core.fig_png(fig), file_name=f"{key}_overlay.png", mime="image/png", key=f"png_{key}")
    d2.download_button("Download water mask (GeoTIFF)", core.mask_geotiff_bytes(res["mask"], res["transform"], res["crs"]),
                       file_name=f"{key}_water_mask.tif", mime="image/tiff", key=f"tif_{key}")
    st.caption("Percent of the image covered by water, including permanent water such as rivers and lakes. It is not a flood forecast.")


def page_mapping():
    page_header("Satellite Analysis", "Find water in Sentinel-1 radar images (VV and VH bands). Files are processed in memory and deleted straight after.")
    try:
        model, device, meta, fi = get_model()
    except Exception as e:
        st.error(f"The flood-mapping model was not found. Finish notebook 1 up to Cell 19 first. Details: {e}")
        return
    tab1, tab2, tab3 = st.tabs(["Example images", "Upload an image", "Before and after"])
    with tab1:
        samples = get_samples()
        if not samples:
            st.warning("No example images found in the dataset folder.")
        else:
            labels = [f"{s['chip']}{'  (India)' if s['india'] else ''}" for s in samples]
            i = st.selectbox("Choose an example radar image", range(len(samples)), format_func=lambda k: labels[k])
            chip = samples[i]["chip"]
            if f"s_{chip}" not in st.session_state:
                with st.spinner("Analysing..."):
                    st.session_state[f"s_{chip}"] = core.load_sample(chip, model, device, meta, fi)
            res = st.session_state[f"s_{chip}"]
            show_result(res, fi, "Detected water", f"example_{chip}")
            if "permanent" in res:
                new = res["mask"] & ~res["permanent"]
                st.write(f"Water that is **not** in the permanent-water layer (a rough estimate of newly flooded area): "
                         f"**{100 * new.sum() / max(res['stats']['valid_pixels'], 1):.1f}%** of the image.")
    with tab2:
        st.write("Upload a georeferenced Sentinel-1 GeoTIFF with VV and VH bands. JPG, PNG and screenshots cannot be analysed because they carry no calibrated radar values.")
        up = st.file_uploader("Radar GeoTIFF", type=["tif", "tiff"], key="up1")
        if up is not None and st.button("Analyse image", key="b1", type="primary"):
            try:
                with st.spinner("Analysing..."):
                    st.session_state["up_res"] = (up.name, core.analyse_bytes(up.getvalue(), up.name, model, device, meta, fi))
            except Exception as e:
                st.session_state.pop("up_res", None); st.error(str(e))
        if "up_res" in st.session_state:
            name, res = st.session_state["up_res"]
            st.write(f"Result for **{name}** ({res['stats']['note']})")
            show_result(res, fi, "Detected water", "upload")
    with tab3:
        st.write("Upload two images of the **same area on the same pixel grid**: one before and one after the event.")
        b = st.file_uploader("Before", type=["tif", "tiff"], key="upb")
        a = st.file_uploader("After", type=["tif", "tiff"], key="upa")
        if b is not None and a is not None and st.button("Compare", key="b2", type="primary"):
            try:
                with st.spinner("Analysing both images..."):
                    rb = core.analyse_bytes(b.getvalue(), b.name, model, device, meta, fi)
                    ra = core.analyse_bytes(a.getvalue(), a.name, model, device, meta, fi)
                    st.session_state["pair"] = (ra, core.compare_pair(rb, ra))
            except Exception as e:
                st.session_state.pop("pair", None); st.error(str(e))
        if "pair" in st.session_state:
            ra, cmp_ = st.session_state["pair"]
            c = st.columns(3)
            c[0].markdown(card("Water before", f"{cmp_['percent_before']:.1f}%"), unsafe_allow_html=True)
            c[1].markdown(card("Water after", f"{cmp_['percent_after']:.1f}%"), unsafe_allow_html=True)
            c[2].markdown(card("Newly flooded", f"{cmp_['percent_new']:.1f}%", f"{cmp_['km2_new']:.2f} sq km", "#d1451b"), unsafe_allow_html=True)
            st.pyplot(core.overlay_fig(ra["display"], cmp_["mask"], "Newly flooded water (after minus before)"))
            st.caption("Newly flooded means water in the after image that was not water in the before image. This is satellite-detected flood water, not a forecast.")


def page_monitor():
    page_header("Monitor", "States ranked by forecast heavy-rain exposure. This is a rain indicator, not a flood forecast.")
    recorded_notice()
    rank, rec = core.national_ranking()
    if rank.empty:
        return
    gdf = get_states()
    h = st.radio("Forecast horizon", ["24h", "48h", "72h"], horizontal=True, index=2)
    col, hi = f"{h} share of points >= 64.5 mm (%)", f"{h} highest (mm)"
    show = rank.sort_values([col, hi], ascending=False).reset_index(drop=True)
    top = show.iloc[0]
    c = st.columns(3)
    c[0].markdown(card("Highest exposure", top["State"] if top[col] > 0 else "None", f"{top[col]:.0f}% of its points at heavy-rain level" if top[col] > 0 else "no state reaches 64.5 mm", "#d1451b"), unsafe_allow_html=True)
    c[1].markdown(card("States with a heavy-rain point", int((rank[col] > 0).sum()), f"out of {len(rank)} states recorded", "#e08a1e"), unsafe_allow_html=True)
    c[2].markdown(card("Highest forecast value", f"{rank[hi].max():.1f} mm", "in a day, at any monitored point", "#1d6fa5"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.05, 1])
    left.pyplot(core.national_map_fig(rank, gdf, col))
    right.dataframe(show[["State", "Points", col, hi]].head(12), hide_index=True)
    right.bar_chart(show.head(10).set_index("State")[col])
    st.caption("Share of monitored points where the forecast reaches 64.5 mm in a day (the IMD heavy-rain level). Horizons use calendar days in IST starting today: "
               "24h is today, 48h is today and tomorrow, 72h is three days. Coarse grid of about 100 km.")
    with st.expander("All horizons"):
        st.dataframe(rank, hide_index=True)


def page_state():
    page_header("State Report", "Forecast, rain history, river panel, reported floods and neighbours for one state, with a downloadable PDF.")
    recorded_notice()
    gdf = get_states()
    names = sorted(gdf["name"])
    state = st.selectbox("State", names, index=names.index("Assam") if "Assam" in names else 0)
    ctx = core.state_context(state, gdf)
    if ctx.get("empty"):
        st.warning("No recorded data for this state."); return
    if not ctx["detail"]:
        st.info(f"Detailed history and river data are recorded only for: {', '.join(core.PILOT_STATES)}. This state shows the coarse forecast only.")
    text_box(ctx["summary"])
    st.write("")
    c = st.columns(3)
    for k, (_, r) in enumerate(ctx["horizons"].iterrows()):
        c[k].markdown(card(r["Horizon"], f"{r['Highest value (mm)']} mm", f"{r['Points at or above 64.5 mm']} points at heavy-rain level<br>{pill(r['Class'])}",
                           core.PILL_COLOR.get(r["Class"], "#888")), unsafe_allow_html=True)
    st.write("")
    tabs = st.tabs(["Next 72 hours", "Rain history", "River", "Reported floods", "Neighbours", "Report (PDF)"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.pyplot(core.forecast_fig(ctx)); c2.pyplot(core.state_map_fig(ctx, gdf))
        st.write("Forecast by monitored point (a regular grid, not districts):")
        st.dataframe(ctx["points"], hide_index=True)
    with tabs[1]:
        if ctx.get("months") is None:
            st.info("History is recorded only for the pilot states.")
        else:
            st.dataframe(ctx["months"], hide_index=True)
            st.pyplot(core.rain_history_fig(ctx))
            if len(ctx["heavy_list"]):
                st.write("Days with heavy-rainfall-level values (these are rain days, not floods):"); st.dataframe(ctx["heavy_list"], hide_index=True)
            else:
                st.write("No day in the recorded period reached the heavy-rainfall level (64.5 mm) at any monitored point.")
    with tabs[2]:
        if ctx.get("rivers") is None or ctx["rivers"].empty:
            st.info("River data is recorded only for Assam.")
        else:
            st.dataframe(ctx["rivers"], hide_index=True)
            pick = st.selectbox("River point", list(ctx["river_series"].keys()))
            st.pyplot(core.river_fig(ctx, pick))
            st.caption("Elevated river flow means the latest or forecast discharge is above the 90th percentile of the last 61 recorded days. These are model-based discharge values, not gauge readings.")
    with tabs[3]:
        if ctx["events"]:
            st.dataframe(core.events_frame(ctx["events"]), hide_index=True)
        else:
            st.write("No large flood event found in the GDACS list for this state. Absence does not prove no flooding.")
        st.caption("GDACS lists large events only. Heavy rainfall alone is not reported as a flood.")
    with tabs[4]:
        if len(ctx["neighbours"]):
            st.write("Neighbouring Indian states (recorded forecast, coarse grid):"); st.dataframe(ctx["neighbours"], hide_index=True)
        else:
            st.write("No neighbour data.")
    with tabs[5]:
        st.write("A readable PDF with the summary, definitions, charts, forecast, river panel, reported floods, neighbours and limitations.")
        if st.button("Generate report", key="gen", type="primary"):
            with st.spinner("Building the report..."):
                st.session_state["pdf"] = (state, core.build_state_pdf(ctx, gdf))
        if "pdf" in st.session_state and st.session_state["pdf"][0] == state:
            st.download_button("Download report (PDF)", st.session_state["pdf"][1],
                               file_name=f"StateReport_{state.replace(' ', '_')}_{datetime.now():%Y%m%d}.pdf", mime="application/pdf")


def page_alerts():
    page_header("Advisories", "Rule-based and human-approved. Approved messages are saved as files in a local outbox. Nothing is sent over the internet.")
    recorded_notice()
    c = st.columns(2)
    c[0].markdown(card("Watch", "64.5 mm or more", "forecast at any monitored point within 48 hours", "#e08a1e"), unsafe_allow_html=True)
    c[1].markdown(card("Advisory", "115.6 mm or more", "within 24 hours, or an elevated river flow flag", "#d1451b"), unsafe_allow_html=True)
    st.write("")
    sim = st.selectbox("Practice scenario (adds an artificial heavy-rain event to one state, clearly marked)", ["None (recorded data only)"] + core.PILOT_STATES)
    cands = core.make_candidates(None if sim.startswith("None") else sim)
    if not cands:
        st.info("No advisories are needed under the rules for the recorded data. Choose a practice scenario above to see how the process works.")
    for i, cnd in enumerate(cands):
        with st.container(border=True):
            st.markdown(f"**{cnd['state']}: {cnd['level']}**" + ("  (PRACTICE SCENARIO)" if cnd["simulated"] else ""))
            st.code(cnd["message"])
            if cnd["cooldown"]:
                st.info(f"Cooldown active: an advisory for {cnd['state']} was approved less than {core.COOLDOWN_HOURS} hours ago.")
            ok = st.checkbox("I have reviewed this advisory", key=f"rev_{i}_{cnd['state']}_{sim}")
            a, r = st.columns(2)
            if a.button("Approve (write to outbox)", key=f"ap_{i}_{cnd['state']}_{sim}", disabled=(not ok) or cnd["cooldown"], type="primary"):
                st.success(f"Saved to the outbox: {core.record_decision(cnd, 'approved')}")
            if r.button("Reject", key=f"rj_{i}_{cnd['state']}_{sim}"):
                core.record_decision(cnd, "rejected"); st.info("Rejected and logged.")
    st.subheader("Decision log")
    log = core.ALERTS / "log.csv"
    if log.exists():
        st.dataframe(pd.read_csv(log).tail(20), hide_index=True)
        st.write(f"Outbox folder: `{core.ALERTS / 'outbox'}` ({len(list((core.ALERTS / 'outbox').glob('*.txt')))} file(s))")
    else:
        st.write("No decisions yet.")


def page_about():
    page_header("About and Help", "How the platform works, how to read it, where the data comes from, and where to find official warnings.")
    t = st.tabs(["How it works", "Reading the information", "Data sources", "Limits and safety", "About the project"])
    with t[0]:
        st.markdown("**Monitor and State Report** use recorded rain forecasts and river discharge from Open-Meteo, and flood reports from GDACS. Rain is grouped into IMD categories, "
                    "and a flood is named only when a report or satellite evidence exists.")
        st.markdown("**Satellite Analysis** uses a deep-learning model (a U-Net) trained on the Sen1Floods11 benchmark to find water in Sentinel-1 radar images. "
                    "Radar sees through cloud, which matters during monsoon floods.")
        st.markdown("**Advisories** apply simple rules to the forecast. A person reviews each message before it is written to a local file. The platform never broadcasts.")
    with t[1]:
        st.markdown("Rain classes use the India Meteorological Department's daily categories:")
        st.table(pd.DataFrame([["Heavy", "64.5 to 115.5 mm per day"], ["Very heavy", "115.6 to 204.4 mm per day"], ["Extremely heavy", "204.5 mm per day or more"]], columns=["Class", "Rain in 24 hours"]))
        st.markdown("Terms used consistently across the platform:")
        st.table(pd.DataFrame(core.DEFINITIONS, columns=["Term", "Used only when"]))
        st.markdown("Water coverage levels in Satellite Analysis (minor, moderate, major, severe) are simple display labels based on the share of the image covered by water.")
    with t[2]:
        st.markdown("- **Open-Meteo** (open-meteo.com): rain forecasts and river discharge based on GloFAS (Copernicus).\n"
                    "- **GDACS** (gdacs.org): flood reports from the Global Disaster Alert and Coordination System.\n"
                    "- **Sen1Floods11** (Cloud to Street): Sentinel-1 radar images and labels used to train the water-mapping model.\n"
                    "- **Natural Earth**: state boundaries. **OpenStreetMap contributors**: map data.")
        st.caption(core.CREDITS)
    with t[3]:
        st.markdown("- Rainfall and river values come from weather and river models, not rain gauges, and usually understate extremes.\n"
                    "- The state view uses a regular grid of points, not districts, and can miss local flash floods.\n"
                    "- GDACS lists large events only. No flood found does not mean safe.\n"
                    "- Satellite water maps include permanent water such as rivers and lakes.\n"
                    "- Uploads must be georeferenced Sentinel-1 VV and VH GeoTIFFs.")
        st.warning("This platform is a research prototype and not an official warning service. For warnings and instructions, follow the India Meteorological Department (IMD), "
                   "the Central Water Commission (CWC), the National Disaster Management Authority (NDMA) and your State or District Disaster Management Authority.")
        st.markdown("Under Section 54 of the Disaster Management Act, 2005, circulating a false alarm or warning about a disaster can be an offence, so this platform never broadcasts. "
                    "Uploaded files are processed in memory and deleted, and no personal data is collected.")
    with t[4]:
        st.table(pd.DataFrame([["Project", f"{PROJECT['title']}: {PROJECT['subtitle']}"], ["Student", PROJECT["student"]], ["Guide", PROJECT["guide"]],
                               ["Institute", PROJECT["institute"]], ["Programme", PROJECT["programme"]], ["Tech stack", PROJECT["tech"]]],
                              columns=["Field", "Detail"]))


# ------------------------------------------------------------------ navigation (top bar)
PAGES = [("Overview", page_overview), ("Monitor", page_monitor), ("State Report", page_state),
         ("Satellite Analysis", page_mapping), ("Advisories", page_alerts), ("About and Help", page_about)]

st.markdown(CSS, unsafe_allow_html=True)
use_nav = hasattr(st, "navigation") and os.environ.get("FLOOD_NAV", "auto") != "radio"
nav = None
if use_nav:
    try:
        nav = st.navigation([st.Page(fn, title=t, default=True) if k == 0 else st.Page(fn, title=t, url_path=t.lower().replace(" ", "-"))
                             for k, (t, fn) in enumerate(PAGES)], position="top")
    except TypeError:
        use_nav = False
if use_nav and nav is not None:
    nav.run()
else:
    choice = st.radio("Navigation", [t for t, _ in PAGES], horizontal=True, label_visibility="collapsed", key="nav")
    dict(PAGES)[choice]()
footer()
