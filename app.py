# ============================================================
# SafeGuard AI — v3.1  Professional UI
# CHANGES from v3.0:
#   - Face emotion removed from results panel and UI
#   - Aggression score added to results panel
#   - Light / Dark theme toggle in sidebar
# ============================================================

import os, sys, time, base64, tempfile, threading
import numpy as np
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db_manager  import (init_db, verify_admin, get_all_incidents,
                                   get_stats, resolve_incident, clear_all_incidents,
                                   get_all_settings, set_setting)
from alerts.alert_manager import determine_alert_level, compute_fused_score
from utils.config         import VISUAL_WEIGHT, AUDIO_WEIGHT

init_db()

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

st.set_page_config(page_title="SafeGuard AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "logged_in": False, "admin": None, "page": "dashboard",
        "last_result": None, "last_video_b64": None,
        "last_video_source": "", "dark_mode": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ════════════════════════════════════════════════════════════
# THEME — CSS vars switch based on dark_mode flag
# ════════════════════════════════════════════════════════════
DARK = st.session_state.dark_mode

if DARK:
    BG_BASE    = "#0b0e14"
    BG_SURFACE = "#111520"
    BG_CARD    = "#161b27"
    BG_HOVER   = "#1c2333"
    TEXT_1     = "#eef2ff"
    TEXT_2     = "#8892aa"
    TEXT_3     = "#4a5568"
    BORDER     = "rgba(255,255,255,0.06)"
    BORDER_ACC = "rgba(79,142,247,0.2)"
else:
    BG_BASE    = "#f0f4ff"
    BG_SURFACE = "#e8edf8"
    BG_CARD    = "#ffffff"
    BG_HOVER   = "#dde4f5"
    TEXT_1     = "#0f1523"
    TEXT_2     = "#3a4560"
    TEXT_3     = "#8892aa"
    BORDER     = "rgba(0,0,0,0.08)"
    BORDER_ACC = "rgba(79,142,247,0.3)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

:root {{
  --bg-base:       {BG_BASE};
  --bg-surface:    {BG_SURFACE};
  --bg-card:       {BG_CARD};
  --bg-hover:      {BG_HOVER};
  --accent:        #4f8ef7;
  --accent-dim:    rgba(79,142,247,0.12);
  --red:           #f75d5d;
  --red-dim:       rgba(247,93,93,0.1);
  --amber:         #f7b955;
  --amber-dim:     rgba(247,185,85,0.1);
  --green:         #4af2a1;
  --green-dim:     rgba(74,242,161,0.1);
  --text-1:        {TEXT_1};
  --text-2:        {TEXT_2};
  --text-3:        {TEXT_3};
  --border:        {BORDER};
  --border-accent: {BORDER_ACC};
  --font:          'Inter', sans-serif;
  --font-mono:     'JetBrains Mono', monospace;
  --font-display:  'Syne', sans-serif;
  --radius:        12px;
  --radius-sm:     8px;
}}

html,body,[class*="css"] {{
  background:{BG_BASE} !important;
  color:{TEXT_1} !important;
  font-family:var(--font) !important;
}}
.stApp {{ background:var(--bg-base) !important; }}
.main .block-container {{
  background:transparent !important;
  padding: 1.5rem 2rem 3rem !important;
  max-width: 1400px !important;
}}

[data-testid="stSidebar"] {{
  background:var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}}
[data-testid="stSidebar"] * {{ color:var(--text-2) !important; }}
[data-testid="stSidebarContent"] {{ padding: 0 !important; }}

.stButton > button {{
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-2) !important;
  font-family: var(--font) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  border-radius: var(--radius-sm) !important;
  padding: 8px 16px !important;
  transition: all 0.15s ease !important;
}}
.stButton > button:hover {{
  background: var(--bg-hover) !important;
  border-color: var(--border-accent) !important;
  color: var(--text-1) !important;
}}
.btn-primary > button {{
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #fff !important;
  font-weight: 600 !important;
}}
.btn-primary > button:hover {{
  background: #3d7ef5 !important;
  border-color: #3d7ef5 !important;
  color: #fff !important;
}}
.btn-danger > button {{
  background: var(--red-dim) !important;
  border-color: rgba(247,93,93,0.3) !important;
  color: var(--red) !important;
}}
.btn-record > button {{
  background: rgba(247,93,93,0.15) !important;
  border: 1px solid var(--red) !important;
  color: var(--red) !important;
  font-weight:600 !important;
}}

[data-testid="stFileUploader"] {{
  background: var(--bg-card) !important;
  border: 1.5px dashed rgba(79,142,247,0.25) !important;
  border-radius: var(--radius) !important;
}}

.stProgress > div > div {{
  background: linear-gradient(90deg, var(--accent), #7db3ff) !important;
  border-radius: 4px !important;
}}

[data-testid="stExpander"] {{
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}}

.stTextInput input, .stSelectbox > div {{
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-1) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-mono) !important;
  font-size:13px !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  background: var(--bg-surface) !important;
  border-radius: var(--radius-sm) !important;
  padding: 3px !important;
  gap: 2px !important;
  border: 1px solid var(--border) !important;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  border-radius: 6px !important;
  color: var(--text-2) !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  padding: 8px 20px !important;
}}
.stTabs [aria-selected="true"] {{
  background: var(--bg-card) !important;
  color: var(--text-1) !important;
}}

[data-testid="stDataFrame"] {{
  background: var(--bg-card) !important;
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
}}

hr {{ border-color: var(--border) !important; margin: 20px 0 !important; }}
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-thumb {{ background: rgba(128,128,128,0.3); border-radius: 4px; }}

h1 {{ font-family: var(--font-display) !important; font-size: 24px !important;
     color: var(--text-1) !important; font-weight: 800 !important; letter-spacing:-0.5px; }}
h2,h3 {{ font-family: var(--font) !important; color: var(--text-1) !important; font-weight:600 !important; }}

.sg-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
}}
.sg-card-accent {{ border-top: 2px solid var(--accent); }}
.sg-card-red    {{ border-top: 2px solid var(--red); }}
.sg-card-amber  {{ border-top: 2px solid var(--amber); }}
.sg-card-green  {{ border-top: 2px solid var(--green); }}

.sg-stat-label {{
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-3);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 8px;
}}
.sg-stat-value {{
  font-family: var(--font-display);
  font-size: 2.2rem;
  font-weight: 800;
  line-height: 1;
}}
.sg-stat-sub {{ font-size: 11px; color: var(--text-3); margin-top: 6px; }}

.sg-section-title {{
  font-size: 11px;
  font-weight: 600;
  color: var(--text-3);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.sg-dot {{ width:5px; height:5px; border-radius:50%; display:inline-block; flex-shrink:0; }}

.sg-badge {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--font-mono);
}}
.sg-badge-red   {{ background:var(--red-dim);   color:var(--red);   border:1px solid rgba(247,93,93,0.25); }}
.sg-badge-amber {{ background:var(--amber-dim); color:var(--amber); border:1px solid rgba(247,185,85,0.25); }}
.sg-badge-green {{ background:var(--green-dim); color:var(--green); border:1px solid rgba(74,242,161,0.25); }}
.sg-badge-blue  {{ background:var(--accent-dim);color:var(--accent);border:1px solid var(--border-accent); }}

.sg-video-wrap {{
  background: #000;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin: 8px 0 12px;
}}
.sg-video-wrap video {{ width:100%; max-height:380px; display:block; }}

.sg-score-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 0;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}}
.sg-score-row:last-child {{ border-bottom: none; }}
.sg-score-label {{ color: var(--text-2); font-family: var(--font-mono); font-size:11px; }}
.sg-score-value {{ font-family: var(--font-mono); font-size:11px; font-weight:600; }}

.sg-transcript {{
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-2);
  line-height: 1.8;
  min-height: 56px;
}}

.sg-phrase-tag {{
  display: inline-block;
  background: var(--red-dim);
  border: 1px solid rgba(247,93,93,0.2);
  color: var(--red);
  border-radius: 4px;
  padding: 2px 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  margin: 2px 3px 2px 0;
}}

.sg-await {{
  background: var(--bg-card);
  border: 1.5px dashed rgba(79,142,247,0.15);
  border-radius: var(--radius);
  padding: 48px 24px;
  text-align: center;
}}

.sg-login-wrap {{ max-width: 380px; margin: 80px auto 0; }}

.sg-cam-placeholder {{
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  height: 300px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: var(--text-3);
  font-size: 13px;
  margin-bottom: 12px;
}}

.recording-pulse {{
  display: inline-block;
  width: 8px; height: 8px;
  background: var(--red);
  border-radius: 50%;
  animation: pulse 1s infinite;
  margin-right: 6px;
}}
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:0.4;transform:scale(0.8)}} }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
def lc(l):  return {"UNSAFE":"var(--red)","REVIEW":"var(--amber)","SAFE":"var(--green)"}.get(l,"var(--accent)")
def lc_raw(l): return {"UNSAFE":"#f75d5d","REVIEW":"#f7b955","SAFE":"#4af2a1"}.get(l,"#4f8ef7")
def sc(s):
    if s >= 0.65: return "var(--red)"
    if s >= 0.50: return "var(--amber)"
    return "var(--green)"

def section_title(text, color="var(--accent)"):
    st.markdown(f"""<div class='sg-section-title'>
    <span class='sg-dot' style='background:{color}'></span>{text}</div>""",
    unsafe_allow_html=True)

def stat_card(label, value, color="var(--accent)", sub=""):
    cls = {"var(--red)":"sg-card-red","var(--amber)":"sg-card-amber",
           "var(--green)":"sg-card-green"}.get(color,"sg-card-accent")
    st.markdown(f"""<div class='sg-card {cls}'>
    <div class='sg-stat-label'>{label}</div>
    <div class='sg-stat-value' style='color:{color}'>{value}</div>
    <div class='sg-stat-sub'>{sub}</div></div>""", unsafe_allow_html=True)

def result_badge(level):
    cfg = {
        "UNSAFE": ("sg-card sg-card-red",  "var(--red)",   "🔴 THREAT DETECTED — UNSAFE"),
        "REVIEW": ("sg-card sg-card-amber", "var(--amber)", "🟡 FLAGGED FOR REVIEW"),
        "SAFE":   ("sg-card sg-card-green", "var(--green)", "🟢 CONTENT SAFE"),
    }
    cls, color, label = cfg.get(level, ("sg-card","var(--accent)","— UNKNOWN"))
    st.markdown(f"""<div class='{cls}' style='text-align:center;padding:18px 12px;margin-bottom:16px'>
    <div style='font-family:var(--font-display);font-size:16px;font-weight:800;
    color:{color};letter-spacing:1px'>{label}</div></div>""", unsafe_allow_html=True)

def badge_html(level):
    cls = {"UNSAFE":"sg-badge sg-badge-red","REVIEW":"sg-badge sg-badge-amber",
           "SAFE":"sg-badge sg-badge-green"}.get(level,"sg-badge sg-badge-blue")
    dot = {"UNSAFE":"🔴","REVIEW":"🟡","SAFE":"🟢"}.get(level,"●")
    return f"<span class='{cls}'>{dot} {level}</span>"

def video_player(b64_data, mime="video/mp4", filename="", size_kb=0):
    meta = f'<div style="font-family:var(--font-mono);font-size:10px;color:var(--text-3);padding:6px 4px;display:flex;justify-content:space-between"><span>📁 {filename}</span><span>{size_kb} KB</span></div>' if filename else ""
    st.markdown(f"""<div class='sg-video-wrap'>
    <video controls preload='metadata' style='width:100%;max-height:380px;display:block'>
      <source src='data:{mime};base64,{b64_data}' type='{mime}'>
    </video></div>{meta}""", unsafe_allow_html=True)

def results_panel(r):
    """Full analysis results panel — face emotion removed, aggression score added."""
    if not r:
        st.markdown("""<div class='sg-await'>
        <div style='font-size:32px;margin-bottom:12px'>🛡️</div>
        <div style='font-family:var(--font-display);font-size:13px;color:var(--text-3);
        letter-spacing:1px;margin-bottom:8px'>AWAITING ANALYSIS</div>
        <div style='font-size:12px;color:var(--text-3)'>Upload a file or record video<br>
        to see results here</div></div>""", unsafe_allow_html=True)
        return

    result_badge(r.get("alert_level", "SAFE"))

    vs    = r.get("visual_score", 0)
    ts    = r.get("toxicity_score", 0)
    ag    = r.get("aggression_score", 0.0)
    fused = r.get("fused_score", 0)
    emo   = (r.get("emotion", "neutral") or "neutral").title()
    econf = r.get("emotion_conf", 0)
    esrc  = (r.get("emotion_source", "") or "").replace("_", " ").upper()
    temo  = (r.get("transcript_emotion", "neutral") or "neutral").title()
    tconf = float(r.get("transcript_emotion_conf", 0) or 0.0)
    is_ag = r.get("is_speech_aggressive", False)

    # Aggression colour
    ag_color = "var(--red)" if ag >= 0.75 else "var(--amber)" if ag >= 0.55 else "var(--green)"

    st.markdown(f"""<div class='sg-card' style='padding:16px 20px;margin-bottom:12px'>
    <div class='sg-score-row'><span class='sg-score-label'>VISUAL THREAT</span>
      <span class='sg-score-value' style='color:{sc(vs)}'>{vs*100:.1f}%</span></div>
    <div class='sg-score-row'><span class='sg-score-label'>AUDIO TOXICITY</span>
      <span class='sg-score-value' style='color:{sc(ts)}'>{ts*100:.1f}%</span></div>
    <div class='sg-score-row'><span class='sg-score-label'>SPEECH AGGRESSION</span>
      <span class='sg-score-value' style='color:{ag_color}'>{ag*100:.1f}%{'  ⚠' if is_ag else ''}</span></div>
    <div class='sg-score-row'><span class='sg-score-label'>FUSED RISK</span>
      <span class='sg-score-value' style='color:{sc(fused)}'>{fused*100:.1f}%</span></div>
    <div class='sg-score-row'><span class='sg-score-label'>EMOTION</span>
      <span class='sg-score-value' style='color:var(--amber)'>{emo}</span></div>
    <div class='sg-score-row'><span class='sg-score-label'>EMOTION SOURCE</span>
      <span class='sg-score-value' style='color:var(--text-2)'>{esrc or '—'}</span></div>
    <div class='sg-score-row'><span class='sg-score-label'>CONFIDENCE</span>
      <span class='sg-score-value' style='color:var(--accent)'>{econf*100:.0f}%</span></div>
    </div>""", unsafe_allow_html=True)

    # Transcript emotion sub-card
    st.markdown(f"""<div class='sg-card' style='padding:12px 16px;margin-bottom:12px'>
    <div class='sg-score-row'><span class='sg-score-label'>TRANSCRIPT EMOTION</span>
      <span class='sg-score-value' style='color:var(--text-2)'>{temo}</span></div>
    <div class='sg-score-row'><span class='sg-score-label'>TRANSCRIPT CONF</span>
      <span class='sg-score-value' style='color:var(--text-3)'>{tconf*100:.0f}%</span></div>
    </div>""", unsafe_allow_html=True)

    # Toxic phrases
    phrases = r.get("toxic_phrases", [])
    if isinstance(phrases, str): phrases = [phrases] if phrases else []
    if phrases:
        section_title("TOXIC PHRASES DETECTED", "var(--red)")
        tags = "".join(f"<span class='sg-phrase-tag'>{p}</span>" for p in phrases[:8])
        st.markdown(f"<div style='margin-bottom:12px'>{tags}</div>", unsafe_allow_html=True)

    # Transcript
    section_title("SPEECH TRANSCRIPT", "var(--accent)")
    transcript = r.get("transcript", "") or ""
    st.markdown(f"""<div class='sg-transcript'>
    {'<span style="color:var(--text-3);font-style:italic">No speech detected.</span>'
     if not transcript or transcript == '[Snapshot — no audio]'
     else transcript}
    </div>""", unsafe_allow_html=True)

    if r.get("errors"):
        with st.expander("⚠️ Processing notes"):
            for e in r["errors"]: st.caption(e)


# ════════════════════════════════════════════════════════════
# LOGIN
# ════════════════════════════════════════════════════════════
def page_login():
    st.markdown("""<div class='sg-login-wrap'>
    <div style='text-align:center;margin-bottom:32px'>
      <div style='font-family:var(--font-display);font-size:28px;font-weight:800;
      color:var(--text-1);letter-spacing:-0.5px'>SafeGuard AI</div>
      <div style='font-size:12px;color:var(--text-3);margin-top:4px;font-family:var(--font-mono)'>
      SECURITY INTELLIGENCE PLATFORM</div>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
        section_title("ADMINISTRATOR LOGIN")
        u = st.text_input("Username", placeholder="admin")
        p = st.text_input("Password", type="password", placeholder="••••••••")
        st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
        if st.button("Sign In", use_container_width=True):
            import hashlib
            uname = u.strip(); pword = p.strip()
            admin = verify_admin(uname, pword)
            if not admin:
                default_hash = hashlib.sha256("admin123".encode()).hexdigest()
                pw_hash      = hashlib.sha256(pword.encode()).hexdigest()
                if uname == "admin" and (pword == "admin123" or pw_hash == default_hash):
                    admin = {"username": "admin"}
            if admin:
                st.session_state.logged_in = True
                st.session_state.admin     = admin
                st.session_state.page      = "dashboard"
                st.rerun()
            else:
                st.error("Invalid credentials — use admin / admin123")
        st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SIDEBAR  — with light/dark toggle
# ════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:20px 16px 8px'>
          <div style='font-family:var(--font-display);font-size:16px;font-weight:800;
          color:var(--text-1)'>SafeGuard</div>
          <div style='font-family:var(--font-mono);font-size:9px;color:var(--text-3);
          letter-spacing:1.5px;margin-top:2px'>AI SECURITY v3.1</div>
        </div>
        <div style='margin:12px 12px;height:1px;background:var(--border)'></div>
        """, unsafe_allow_html=True)

        stats = get_stats()
        cur   = st.session_state.page

        nav = [
            ("dashboard", "📊", "Dashboard"),
            ("analysis",  "🎬", "Media Analysis"),
            ("alerts",    "🚨", f"Alerts  {stats['unsafe']} open" if stats['unsafe'] else "Alerts"),
            ("incidents", "📋", "Incident Log"),
            ("settings",  "⚙️", "Settings"),
        ]
        for pid, icon, label in nav:
            if st.button(f"{icon}  {label}", key=f"nav_{pid}", use_container_width=True):
                st.session_state.page = pid; st.rerun()

        st.markdown(f"""
        <div style='margin:12px 12px;height:1px;background:var(--border)'></div>
        <div style='padding:8px 16px;font-family:var(--font-mono);font-size:10px;color:var(--text-3)'>
          <div style='margin-bottom:4px'>TOTAL   <span style='color:var(--accent);float:right'>{stats['total']}</span></div>
          <div style='margin-bottom:4px'>UNSAFE  <span style='color:var(--red);float:right'>{stats['unsafe']}</span></div>
          <div>RESOLVED <span style='color:var(--green);float:right'>{stats['resolved']}</span></div>
        </div>
        <div style='margin:12px 12px;height:1px;background:var(--border)'></div>
        """, unsafe_allow_html=True)

        # ── Light / Dark toggle ───────────────────────────────
        theme_label = "☀️  Light Mode" if DARK else "🌙  Dark Mode"
        if st.button(theme_label, use_container_width=True, key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.markdown("<div style='margin:8px 12px;height:1px;background:var(--border)'></div>",
                    unsafe_allow_html=True)

        if st.button("Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()


# ════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown(f"""
    <h1>Security Overview</h1>
    <div style='font-family:var(--font-mono);font-size:11px;color:var(--text-3);margin-bottom:24px'>
    {datetime.now().strftime("%A, %d %B %Y  %H:%M:%S")}</div>""", unsafe_allow_html=True)

    stats = get_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Active Threats",  stats['unsafe'],   "var(--red)",   "Unresolved incidents")
    with c2: stat_card("Under Review",    stats['review'],   "var(--amber)", "Flagged for review")
    with c3: stat_card("Resolved",        stats['resolved'], "var(--green)", "Cleared by admin")
    with c4: stat_card("Total Analyzed",  stats['total'],    "var(--accent)","All sessions")

    st.markdown("<br>", unsafe_allow_html=True)

    unsafe_list = get_all_incidents(level_filter="UNSAFE")[:3]
    if unsafe_list:
        section_title("ACTIVE THREATS", "var(--red)")
        for inc in unsafe_list:
            ts  = inc.get("timestamp","")[:19]
            vs  = inc.get("violence_score", 0)
            emo = (inc.get("emotion_detected","?") or "?").title()
            src = inc.get("video_source","?")
            st.markdown(f"""<div class='sg-card sg-card-red' style='padding:12px 18px;margin-bottom:8px;
            display:flex;align-items:center;justify-content:space-between'>
            <div>
              <div style='font-size:13px;font-weight:600;color:var(--red);margin-bottom:3px'>🔴 {src}</div>
              <div style='font-family:var(--font-mono);font-size:11px;color:var(--text-3)'>
              {ts} · Violence {vs*100:.1f}% · {emo}</div>
            </div>
            <span class='sg-badge sg-badge-red'>UNSAFE</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("RECENT ACTIVITY")
    incidents = get_all_incidents(limit=8)
    if incidents:
        import pandas as pd
        df = pd.DataFrame([{
            "Time":       i.get("timestamp","")[:19],
            "Source":     i.get("video_source","?"),
            "Level":      i.get("alert_level","?"),
            "Violence":   f"{i.get('violence_score',0)*100:.1f}%",
            "Emotion":    (i.get("emotion_detected","?") or "?").title(),
            "Status":     i.get("status","?"),
        } for i in incidents])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.markdown("<div class='sg-card' style='text-align:center;padding:32px;color:var(--text-3)'>No incidents yet</div>",
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MEDIA ANALYSIS
# ════════════════════════════════════════════════════════════
def page_analysis():
    st.markdown("<h1>Media Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:var(--font-mono);font-size:11px;color:var(--text-3);margin-bottom:20px'>UPLOAD · RECORD · ANALYZE</div>", unsafe_allow_html=True)

    tab_upload, tab_live = st.tabs(["  📁  Upload File  ", "  📹  Live Record  "])

    # ── TAB 1: UPLOAD ─────────────────────────────────────────
    with tab_upload:
        col_l, col_r = st.columns([1, 1], gap="large")
        with col_l:
            section_title("UPLOAD MEDIA FILE")
            uploaded = st.file_uploader(
                "Drag & drop or browse",
                type=["mp4","avi","mov","mkv","wav","mp3"],
                label_visibility="collapsed")

            if uploaded:
                suffix   = os.path.splitext(uploaded.name)[1].lower()
                is_video = suffix in [".mp4",".avi",".mov",".mkv"]
                is_audio = suffix in [".wav",".mp3"]

                if is_video:
                    section_title("VIDEO PREVIEW")
                    uploaded.seek(0); vbytes = uploaded.read(); uploaded.seek(0)
                    mime = {"mp4":"video/mp4","avi":"video/x-msvideo",
                            "mov":"video/quicktime","mkv":"video/x-matroska"}.get(suffix.lstrip("."),"video/mp4")
                    video_player(base64.b64encode(vbytes).decode(), mime,
                                 uploaded.name, uploaded.size//1024)
                elif is_audio:
                    section_title("AUDIO PREVIEW")
                    uploaded.seek(0); st.audio(uploaded); uploaded.seek(0)

                st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                if st.button("🔍  Analyze File", use_container_width=True, key="btn_analyze_upload"):
                    uploaded.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.read()); tmp_path = tmp.name

                    section_title("PIPELINE")
                    steps = [
                        ("Frame Extraction",        not is_audio),
                        ("Visual Model (ResNet18)",  not is_audio),
                        ("Audio Extraction",         True),
                        ("Whisper ASR",              True),
                        ("DistilBERT Toxicity",      True),
                        ("Aggression Detection",     True),
                        ("Emotion Analysis",         True),
                        ("Multimodal Fusion",        True),
                    ]
                    bars = {s: st.progress(0, text=f"  {s}") for s, _ in steps}
                    from pipeline import analyze_video, analyze_audio_only
                    rh = {}
                    def _run():
                        try:
                            if is_audio: rh["r"] = analyze_audio_only(tmp_path, "UPLOAD")
                            else:        rh["r"] = analyze_video(tmp_path, "UPLOAD")
                        except Exception as e: rh["err"] = str(e)
                    t = threading.Thread(target=_run, daemon=True); t.start()
                    for sn, active in steps:
                        if active:
                            for pct in range(0, 101, 25):
                                bars[sn].progress(pct/100, text=f"  {sn}..."); time.sleep(0.06)
                        bars[sn].progress(1.0 if active else 0.0,
                            text=f"  {'✓' if active else '—'}  {sn}")
                    t.join(timeout=180)
                    try: os.remove(tmp_path)
                    except: pass

                    if "err" in rh: st.error(f"Analysis failed: {rh['err']}")
                    elif "r" in rh:
                        if is_video:
                            uploaded.seek(0)
                            st.session_state.last_video_b64    = base64.b64encode(uploaded.read()).decode()
                            st.session_state.last_video_source = uploaded.name
                        st.session_state.last_result = rh["r"]
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            results_panel(st.session_state.last_result)

    # ── TAB 2: LIVE RECORD ────────────────────────────────────
    with tab_live:
        col_l, col_r = st.columns([1, 1], gap="large")

        with col_l:
            section_title("WEBCAM RECORDING")
            st.markdown("""<div class='sg-card' style='padding:16px 20px;margin-bottom:16px'>
            <div style='font-size:12px;color:var(--text-2);line-height:1.7'>
            <b style='color:var(--text-1)'>How it works:</b><br>
            1. Set the recording duration below<br>
            2. Click <b style='color:var(--accent)'>Start Recording</b><br>
            3. Camera records for exact N seconds then stops<br>
            4. Full analysis runs automatically
            </div></div>""", unsafe_allow_html=True)

            duration = st.slider("Recording duration", 3, 60, 10,
                                 format="%d sec", key="live_duration")

            st.markdown("<div class='btn-record'>", unsafe_allow_html=True)
            start_rec = st.button("⏺  Start Recording", use_container_width=True, key="btn_start_rec")
            st.markdown("</div>", unsafe_allow_html=True)

            if start_rec:
                ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(RECORDINGS_DIR, f"rec_{ts_str}.mp4")

                rec_progress = st.progress(0)
                rec_status   = st.empty()
                rec_status.markdown(f"""<div style='font-family:var(--font-mono);font-size:12px;
                color:var(--text-2)'><span class='recording-pulse'></span>Recording...</div>""",
                unsafe_allow_html=True)

                import cv2
                rh = {}
                vbytes = b""

                def _do_record():
                    try:
                        import sounddevice as sd
                        import scipy.io.wavfile as wav_io
                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        fps_cam = cap.get(cv2.CAP_PROP_FPS) or 20
                        fps_cam = max(10, min(fps_cam, 30))
                        fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
                        writer  = cv2.VideoWriter(save_path, fourcc, fps_cam, (640, 480))

                        samplerate = 16000
                        audio_data = []
                        def audio_callback(indata, frames, time_info, status):
                            audio_data.extend(indata[:, 0].tolist())
                        stream = sd.InputStream(samplerate=samplerate, channels=1,
                                                callback=audio_callback)
                        stream.start()

                        start_t = time.time()
                        while time.time() - start_t < duration:
                            ret, frame = cap.read()
                            if ret: writer.write(frame)

                        stream.stop(); stream.close()
                        cap.release(); writer.release()

                        wav_path = save_path.replace(".mp4", ".wav")
                        a   = np.array(audio_data, dtype=np.float32)
                        a   = np.clip(a, -1.0, 1.0)
                        a16 = (a * 32767.0).astype(np.int16)
                        wav_io.write(wav_path, samplerate, a16)
                        rh["video_path"] = save_path
                        rh["wav_path"]   = wav_path
                    except Exception as e:
                        rh["err"] = str(e)

                t_rec = threading.Thread(target=_do_record, daemon=True); t_rec.start()
                start_t = time.time()
                while t_rec.is_alive():
                    elapsed   = time.time() - start_t
                    pct       = min(elapsed / max(duration, 1), 1.0)
                    secs_left = max(0, duration - int(elapsed))
                    rec_progress.progress(pct, text=f"⏺  {int(elapsed)}s / {duration}s  ({secs_left}s remaining)")
                    time.sleep(0.25)
                t_rec.join()
                rec_progress.progress(1.0, text="✓  Recording complete")

                if "err" in rh:
                    rec_status.empty()
                    st.error(f"Recording failed: {rh['err']}")
                else:
                    rec_status.markdown("<div style='font-family:var(--font-mono);font-size:12px;"
                                        "color:var(--green)'>✓ Recorded — analyzing...</div>",
                                        unsafe_allow_html=True)

                    if os.path.exists(rh["video_path"]):
                        with open(rh["video_path"], "rb") as vf:
                            vbytes = vf.read()
                        section_title("RECORDED VIDEO PREVIEW")
                        video_player(base64.b64encode(vbytes).decode(), "video/mp4",
                                     os.path.basename(rh["video_path"]), len(vbytes)//1024)

                    from pipeline import analyze_video
                    try:
                        result = analyze_video(rh["video_path"], "LIVE_RECORD",
                                               wav_path_override=rh.get("wav_path"))
                        st.session_state.last_result       = result
                        st.session_state.last_video_b64    = base64.b64encode(vbytes).decode()
                        st.session_state.last_video_source = os.path.basename(rh["video_path"])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

            elif st.session_state.get("last_video_source","").startswith("rec_"):
                section_title("LAST RECORDING")
                if st.session_state.last_video_b64:
                    video_player(st.session_state.last_video_b64, "video/mp4",
                                 st.session_state.last_video_source)

        with col_r:
            results_panel(st.session_state.last_result)


# ════════════════════════════════════════════════════════════
# ALERT CENTER
# ════════════════════════════════════════════════════════════
def page_alerts():
    st.markdown("<h1>Alert Center</h1>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        fl = st.selectbox("Filter", ["ALL","UNSAFE","REVIEW","SAFE"], label_visibility="collapsed")
    with c3:
        st.markdown("<div class='btn-danger'>", unsafe_allow_html=True)
        if st.button("🗑  Clear All", use_container_width=True):
            clear_all_incidents(); st.success("Cleared."); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    incidents = get_all_incidents(limit=100, level_filter=None if fl=="ALL" else fl)
    if not incidents:
        st.markdown("<div class='sg-await'><div style='color:var(--text-3)'>No incidents found</div></div>",
                    unsafe_allow_html=True)
        return

    for inc in incidents:
        iid    = inc.get("incident_id")
        ts     = inc.get("timestamp","")[:19]
        src    = inc.get("video_source","?")
        level  = inc.get("alert_level","?")
        vs     = inc.get("violence_score", 0)
        tox    = inc.get("toxicity_score", 0)
        emo    = (inc.get("emotion_detected","?") or "?").title()
        status = inc.get("status","OPEN")
        phrases= inc.get("toxic_phrases", [])
        trans  = inc.get("transcript","")

        with st.expander(f"{ts}  ·  {src}  ·  Violence {vs*100:.1f}%"):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"""
                <div style='display:flex;gap:8px;align-items:center;margin-bottom:12px;flex-wrap:wrap'>
                  {badge_html(level)}
                  <span class='sg-badge sg-badge-blue'>{emo}</span>
                  <span style='font-family:var(--font-mono);font-size:11px;color:var(--text-3)'>
                    Visual {vs*100:.1f}%  ·  Toxicity {tox*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
                if phrases:
                    tags = "".join(f"<span class='sg-phrase-tag'>{p}</span>"
                                   for p in (phrases if isinstance(phrases,list) else [phrases])[:6])
                    st.markdown(f"<div style='margin-bottom:10px'>{tags}</div>", unsafe_allow_html=True)
                if trans:
                    st.markdown(f"<div class='sg-transcript'>{trans[:300]}{'...' if len(trans)>300 else ''}</div>",
                                unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<div style='font-family:var(--font-mono);font-size:11px;color:var(--text-3);"
                            f"margin-bottom:12px'>Status: <span style='color:var(--text-1)'>{status}</span></div>",
                            unsafe_allow_html=True)
                if status == "OPEN":
                    if st.button("✓ Resolve", key=f"res_{iid}", use_container_width=True):
                        resolve_incident(iid); st.rerun()


# ════════════════════════════════════════════════════════════
# INCIDENT LOG
# ════════════════════════════════════════════════════════════
def page_incidents():
    st.markdown("<h1>Incident Log</h1>", unsafe_allow_html=True)

    stats = get_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Unsafe",  stats['unsafe'],  "var(--red)")
    with c2: stat_card("Review",  stats['review'],  "var(--amber)")
    with c3: stat_card("Safe",    stats['safe'],    "var(--green)")
    with c4: stat_card("Total",   stats['total'],   "var(--accent)")
    st.markdown("<br>", unsafe_allow_html=True)

    incidents = get_all_incidents(limit=200)
    if not incidents:
        st.markdown("<div class='sg-await'><div style='color:var(--text-3)'>No incidents recorded yet</div></div>",
                    unsafe_allow_html=True)
        return

    col_f1, col_f2, _ = st.columns([1, 1, 2])
    with col_f1:
        fl = st.selectbox("Level", ["ALL","UNSAFE","REVIEW","SAFE"])
    with col_f2:
        src_filter = st.selectbox("Source", ["ALL","UPLOAD","LIVE_RECORD","LIVE"])

    filtered = [i for i in incidents if
                (fl=="ALL" or i.get("alert_level")==fl) and
                (src_filter=="ALL" or i.get("video_source","")==src_filter)]

    st.markdown("<br>", unsafe_allow_html=True)
    section_title(f"INCIDENTS  ({len(filtered)} records)")

    for inc in filtered:
        iid    = inc.get("incident_id")
        ts     = inc.get("timestamp","")[:19]
        src    = inc.get("video_source","?")
        level  = inc.get("alert_level","?")
        vs     = inc.get("violence_score", 0)
        tox    = inc.get("toxicity_score", 0)
        emo    = (inc.get("emotion_detected","?") or "?").title()
        status = inc.get("status","OPEN")
        phrases= inc.get("toxic_phrases", [])
        trans  = inc.get("transcript","")

        with st.expander(f"  {ts}   ·   {src}   ·   {level}   ·   {vs*100:.1f}%"):
            col_info, col_video = st.columns([1, 1], gap="large")

            with col_info:
                st.markdown(f"""
                <div style='display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;align-items:center'>
                  {badge_html(level)}
                  <span class='sg-badge sg-badge-blue'>{emo}</span>
                  <span style='font-family:var(--font-mono);font-size:10px;color:var(--text-3)'>ID #{iid}</span>
                </div>
                <div class='sg-card' style='padding:12px 16px;margin-bottom:12px'>
                <div class='sg-score-row'><span class='sg-score-label'>VISUAL</span>
                  <span class='sg-score-value' style='color:{sc(vs)}'>{vs*100:.1f}%</span></div>
                <div class='sg-score-row'><span class='sg-score-label'>TOXICITY</span>
                  <span class='sg-score-value' style='color:{sc(tox)}'>{tox*100:.1f}%</span></div>
                <div class='sg-score-row'><span class='sg-score-label'>STATUS</span>
                  <span class='sg-score-value' style='color:var(--text-2)'>{status}</span></div>
                </div>""", unsafe_allow_html=True)

                if phrases:
                    pl   = phrases if isinstance(phrases, list) else [phrases]
                    tags = "".join(f"<span class='sg-phrase-tag'>{p}</span>" for p in pl[:6])
                    st.markdown(f"<div style='margin-bottom:10px'>{tags}</div>", unsafe_allow_html=True)

                if trans:
                    section_title("TRANSCRIPT")
                    st.markdown(f"<div class='sg-transcript' style='font-size:11px'>{trans[:400]}</div>",
                                unsafe_allow_html=True)

                if status == "OPEN":
                    if st.button("✓ Mark Resolved", key=f"res_inc_{iid}", use_container_width=True):
                        resolve_incident(iid); st.rerun()

            with col_video:
                section_title("RECORDED VIDEO")
                rec_files    = sorted(os.listdir(RECORDINGS_DIR)) if os.path.isdir(RECORDINGS_DIR) else []
                matched_path = None
                for fname in rec_files:
                    if fname.endswith(".mp4") and (src in fname or fname.replace(".mp4","") in src):
                        matched_path = os.path.join(RECORDINGS_DIR, fname); break
                if not matched_path and src not in ("LIVE","UPLOAD","LIVE_RECORD"):
                    candidate = os.path.join(RECORDINGS_DIR, src)
                    if os.path.exists(candidate): matched_path = candidate

                if matched_path and os.path.exists(matched_path):
                    with open(matched_path, "rb") as vf: vb = vf.read()
                    video_player(base64.b64encode(vb).decode(), "video/mp4",
                                 os.path.basename(matched_path), len(vb)//1024)
                elif src == "UPLOAD":
                    st.markdown("""<div class='sg-cam-placeholder' style='height:160px'>
                    <div style='font-size:24px'>📁</div>
                    <div style='font-size:12px'>Uploaded file not stored<br>
                    <span style='font-size:11px;color:var(--text-3)'>Re-upload to view again</span>
                    </div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class='sg-cam-placeholder' style='height:160px'>
                    <div style='font-size:24px'>📹</div>
                    <div style='font-size:12px;text-align:center'>Video not available<br>
                    <span style='font-size:11px;color:var(--text-3)'>Recording may have been cleared</span>
                    </div></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SETTINGS
# ════════════════════════════════════════════════════════════
def page_settings():
    st.markdown("<h1>System Configuration</h1>", unsafe_allow_html=True)
    settings = get_all_settings()
    c1, c2   = st.columns(2, gap="large")

    with c1:
        st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
        section_title("DETECTION THRESHOLDS")
        vt  = st.slider("Violence Threshold",  0.50, 0.95, float(settings.get("violence_threshold", 0.65)), 0.01)
        tt  = st.slider("Toxicity Threshold",  0.40, 0.90, float(settings.get("toxicity_threshold", 0.60)), 0.01)
        fps = st.slider("Frame Sample Rate",   1,    30,   int(settings.get("frame_sample_rate", 5)), 1)
        section_title("FUSION WEIGHTS")
        vw  = st.slider("Visual Weight",       0.0,  1.0,  float(settings.get("visual_weight", 0.70)), 0.05)
        aw  = round(1.0 - vw, 2)
        st.markdown(f"<div style='font-family:var(--font-mono);font-size:11px;color:var(--text-3)'>"
                    f"Audio weight auto-calculated: <span style='color:var(--accent)'>{aw}</span></div>",
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
        section_title("ALERT SETTINGS")
        asound = st.toggle("Alert Sound", value=settings.get("alert_sound","true")=="true")
        st.toggle("Auto-save Incidents", value=True)
        section_title("WHISPER MODEL")
        wm = st.selectbox("Model Size", ["tiny","base","small","medium","large"],
            index=["tiny","base","small","medium","large"].index(
                settings.get("whisper_model","base")))
        section_title("SYSTEM")
        st.markdown("""<div style='font-family:var(--font-mono);font-size:11px;color:var(--text-2);line-height:2.2'>
        VERSION    <span style='float:right;color:var(--accent)'>v3.1</span><br>
        VISUAL     <span style='float:right;color:var(--green)'>ResNet18</span><br>
        AGGRESSION <span style='float:right;color:var(--green)'>Wav2Vec2</span><br>
        ASR        <span style='float:right;color:var(--accent)'>Whisper</span><br>
        NLP        <span style='float:right;color:var(--accent)'>DistilBERT</span><br>
        DATABASE   <span style='float:right;color:var(--green)'>SQLite ✓</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
    if st.button("Save Configuration", use_container_width=True):
        set_setting("violence_threshold", str(vt))
        set_setting("toxicity_threshold", str(tt))
        set_setting("visual_weight",      str(vw))
        set_setting("audio_weight",       str(aw))
        set_setting("frame_sample_rate",  str(fps))
        set_setting("alert_sound",        "true" if asound else "false")
        set_setting("whisper_model",      wm)
        st.success("Configuration saved.")
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# MAIN ROUTER
# ════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        page_login(); return
    render_sidebar()
    page = st.session_state.page
    if   page == "dashboard": page_dashboard()
    elif page == "analysis":  page_analysis()
    elif page == "alerts":    page_alerts()
    elif page == "incidents": page_incidents()
    elif page == "settings":  page_settings()
    else:                     page_dashboard()

if __name__ == "__main__":
    main()