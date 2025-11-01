# MannMitra – Youth Wellness App (Streamlit)

import os
import json
import time
import sqlite3
import random
from datetime import datetime, date
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
from colorsys import rgb_to_yiq

# Optional libs (app still runs if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    import google.generativeai as genai
    HAS_GEMINI = bool(GEMINI_API_KEY)
    if HAS_GEMINI:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
except Exception:
    HAS_GEMINI = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except Exception:
    HAS_GTTS = False

try:
    import bcrypt
    HAS_BCRYPT = True
except Exception:
    HAS_BCRYPT = False

try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False

# ---------------------- Settings ----------------------
st.set_page_config(
    page_title="MannMitra",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",  # keep sidebar accessible
)


# ---------- Minimal, safe CSS (no overlays) ----------
st.markdown("""
<style>
/* =================== MannMitra Global Theme (GPT-like) =================== */
:root{
  --mm-pw: 750px;

  /* Core palette (tweak these three to fine-tune the vibe) */
  --mm-bg-1:#0e0f13;   /* top of gradient */
  --mm-bg-2:#0b0d10;   /* bottom of gradient */
  --mm-card:#1a1d22;   /* card / panel color */

  /* Accents */
  --mm-accent:#e5484d; /* tab underline accent */
  --mm-warm:#ff7a45;   /* warm primary button */
  --mm-text:#e9e9ea;   /* text on dark */
}

/* App background (GPT-like neutral gradient) */
html, body, .stApp{ background: linear-gradient(180deg, var(--mm-bg-1), var(--mm-bg-2)) !important; }

/* Keep Streamlit header under our toggle */
header[data-testid="stHeader"]{ z-index:0 !important; }

/* =================== Layout & Card =================== */
.block-container{ padding-top:.5rem; padding-bottom:.5rem; }

.mm-card{
  width:min(96vw, var(--mm-pw)) !important;
  max-width:var(--mm-pw) !important;
  margin:0 auto;
  box-sizing:border-box;
  padding:24px 28px 28px;
  border-radius:18px;
  border:2px solid rgba(255,255,255,0.10);
  background:var(--mm-card);
  color:var(--mm-text);
  box-shadow:0 10px 24px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.06) inset;
  transition:transform .1s ease, border-color .2s ease;
}
@media (hover:hover){ .mm-card:hover{ transform:translateY(-1px); } }

.mm-card [data-testid="stForm"],
.mm-card [data-testid="stVerticalBlock"]>div{
  background:transparent !important; box-shadow:none !important; border:none !important;
}

/* Centered wrappers / spacing (keep your existing distances) */
.mm-viewport-top{ display:flex; justify-content:center; align-items:flex-start; padding-top:48px; }
.mm-viewport-top.login-gap{ padding-top:42px; }
.mm-viewport-top.confetti-gap{ padding-top:42px; }

/* Titles & tabs */
.mm-title-center{ font-size:2rem; font-weight:800; margin:0 0 8px; text-align:center; line-height:1.2; }
.mm-title{ font-size:2.2rem; font-weight:800; margin:0 0 12px; text-align:center; }
.mm-body{ font-size:1.15rem; line-height:1.7; text-align:center; }
.mm-card [data-baseweb="tab-highlight"]{ border-color:var(--mm-accent) !important; }
.mm-card [data-baseweb="tab-list"]{ display:flex !important; justify-content:center !important; gap:40px !important; }
.mm-card [data-baseweb="tab"]{ flex:0 0 auto !important; }

/* Alert spacing: ≈1cm gap (38px) below card content */
.mm-card [data-testid="stAlert"], .mm-card .stAlert{ margin-top:38px !important; }

/* =================== Buttons =================== */
.stButton>button{
  border-radius:10px; font-size:18px !important; height:46px !important; padding:4px 0 !important;
  width:100% !important; /* main actions fill container */
}
.stButton>button[kind="primary"]{
  background:var(--mm-warm) !important; border:1px solid var(--mm-warm) !important; color:#191a1e !important;
  box-shadow:0 2px 0 rgba(0,0,0,.05);
}
.stButton>button[kind="primary"]:hover{ filter:brightness(0.95); }
.stButton>button[kind="primary"]:active{ transform:translateY(1px); }

/* Secondary buttons: soft warm outline */
.stButton>button:not([kind="primary"]){
  border:1px solid rgba(255,122,69,0.35) !important;
}

/* Ensure single buttons (logout/stealth/etc.) expand */
div[data-testid="stButton"]>button{ width:100% !important; }

/* =================== Chat Input Row =================== */
.input-inner, .mm-input-row{ display:flex !important; align-items:center !important; gap:8px !important; }
.input-inner .stTextInput>div, .mm-input-row .stTextInput>div{ display:flex !important; align-items:center !important; width:100% !important; }
.input-inner .stTextInput input, .mm-input-row .stTextInput input{ height:40px !important; padding:8px 10px !important; border-radius:10px !important; }
/* Equal sizes for Send / Mic / Speaker */
.input-inner .stButton button, .st_mic_recorder>button, .mm-speaker button{ width:60px !important; height:40px !important; margin:0 !important; }

/* =================== Sidebar =================== */
[data-testid='stSidebar']{ transition:opacity .2s ease; }

/* --- Sidebar toggle: always visible, keep default chevrons (robust across versions) --- */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[title="Hide sidebar"],
button[title="Expand sidebar"]{
  position:fixed !important;
  top:14px !important; left:14px !important;
  display:flex !important; visibility:visible !important; opacity:1 !important;
  pointer-events:auto !important; z-index:10000 !important;
}

/* Subtle bubble so it stands out on dark bg */
[data-testid="collapsedControl"] > div,
[data-testid="stSidebarCollapseButton"],
button[title="Hide sidebar"],
button[title="Expand sidebar"]{
  background:rgba(255,255,255,0.08) !important;
  padding:6px 10px !important;
  border-radius:10px !important;
  backdrop-filter:blur(5px) !important;
  transition:background .2s ease !important;
}
[data-testid="collapsedControl"] > div:hover,
[data-testid="stSidebarCollapseButton"]:hover,
button[title="Hide sidebar"]:hover,
button[title="Expand sidebar"]:hover{
  background:rgba(255,255,255,0.16) !important;
}

/* IMPORTANT: keep the default chevrons visible — do NOT hide SVGs or inject custom icons */
</style>
""", unsafe_allow_html=True)


# ====================== GLOBAL BUTTON THEME (STATIC PRESETS) ======================

ALLOWED_BUTTON_COLORS = {
    "Slate":            "#64748b",   # you liked this
    "Pastel Blue":      "#93c5fd",
    "Mint":             "#10b981",
    "Peach":            "#fdba74",
    "Coral (lighter)":  "#ff9aa5",   # lighter than default coral
    "Lavender":         "#c4b5fd",
}

DEFAULT_BUTTON_HEX = ALLOWED_BUTTON_COLORS["Slate"]  # <- app default (static)

def _hex_to_rgb(h: str):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def _contrast_text_for_bg(hex_bg: str) -> str:
    r,g,b = _hex_to_rgb(hex_bg)
    yiq = (r*299 + g*587 + b*114) / 1000
    return "#000000" if yiq >= 160 else "#ffffff"

def _rgba(hex_bg: str, alpha: float) -> str:
    r,g,b = _hex_to_rgb(hex_bg)
    return f"rgba({r},{g},{b},{alpha})"

def _mix_with_white(hex_bg: str, pct: float=0.5) -> str:
    r,g,b = _hex_to_rgb(hex_bg)
    rw = int(r + (255-r)*pct); gw = int(g + (255-g)*pct); bw = int(b + (255-b)*pct)
    return f"#{rw:02x}{gw:02x}{bw:02x}"

def apply_global_button_theme_hex(hex_color: str):
    """Recolor ALL buttons + tab accent using an approved hex color."""
    bg = hex_color
    text = _contrast_text_for_bg(bg)
    ring = _rgba(bg, 0.40)
    tab_text = _mix_with_white(bg, 0.50)
    underline = bg

    st.markdown(f"""
    <style>
      /* Buttons (primary/secondary/download) */
      .stButton > button,
      .stDownloadButton > button,
      div.stButton > button[kind="secondary"],
      div.stButton > button[kind="primary"] {{
        background: {bg} !important;
        color: {text} !important;
        border: 1px solid transparent !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 20px rgba(0,0,0,.20);
      }}
      .stButton > button:hover, .stDownloadButton > button:hover {{
        filter: brightness(0.97);
        box-shadow: 0 0 0 3px {ring} !important;
      }}
      .stButton > button:focus, .stDownloadButton > button:focus {{
        outline: none !important;
        box-shadow: 0 0 0 3px {ring} !important;
      }}

      /* Tabs accent to match */
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        border-bottom: 2px solid {underline} !important;
        color: {tab_text} !important;
      }}
      div[data-testid="stTabs"] button[role="tab"]:hover {{
        color: {tab_text} !important;
      }}
    </style>
    """, unsafe_allow_html=True)



# ---------- Paths ----------
CONTENT_DIR = "content"
WHO5_JSON = os.path.join(CONTENT_DIR, "who5.json")
HELPLINES_JSON = os.path.join(CONTENT_DIR, "helplines_in.json")

os.makedirs("data", exist_ok=True)
DB_PATH = "data/mannmitra.db"

# ---------------------- DB utils ----------------------
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash BLOB NOT NULL,
            pin TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_scores (
            username TEXT,
            day TEXT,
            score INTEGER,
            band TEXT,
            band_idx INTEGER DEFAULT 0,
            PRIMARY KEY (username, day)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gratitude (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            day TEXT,
            ts TEXT,
            q1 TEXT, q2 TEXT, q3 TEXT,
            mood INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            ts TEXT,
            activity_type TEXT,
            meta TEXT
        )
    """)
    return conn

conn = get_db()

def hash_pw(plain: str) -> bytes:
    if not HAS_BCRYPT:
        return plain.encode("utf-8")
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt())

def verify_pw(plain: str, hashed: bytes) -> bool:
    if not HAS_BCRYPT:
        return hashed == plain.encode("utf-8")
    try:
        return bcrypt.checkpw(plain.encode(), hashed)
    except Exception:
        return False

# ---------------------- Language strings ----------------------
LANG = {
    "English": {"welcome":"Welcome to MannMitra 🌿","disclaimer":"...", "start":"Start","login":"Login","logout":"Logout","hide":"Hide (Stealth Mode)","unhide":"Unhide","pin":"PIN","chat_placeholder":"Type your message…","stealth_screen":"Please wait… 🕐 Something went wrong. Try again later..."},
}

# ---------------------- Session state ----------------------
ss = st.session_state
ss.setdefault("lang", "English")
ss.setdefault("authenticated", False)
ss.setdefault("username", None)
ss.setdefault("stealth", False)
ss.setdefault("active_tab", "Chat")
ss.setdefault("intro_step", "welcome")
ss.setdefault("start_anim_time", None)
ss.setdefault("chat", [])
ss.setdefault("force_sidebar_open", True)  # ensures we can always reopen the sidebar


# ---------------------- Sidebar Visibility  ----------------------
ss.setdefault("sidebar_open", True)
ss.setdefault("applied_theme_hex", "#64748b")  # Persist across reruns within session

DEFAULT_BUTTON_HEX = "#64748b"
ALLOWED_BUTTON_COLORS = {
    "Slate": "#64748b",
    "Pastel Blue": "#93c5fd",
    "Mint": "#10b981",
    "Peach": "#fdba74",
    "Coral (lighter)": "#ff9aa5",
    "Lavender": "#c4b5fd",
}

# Hide sidebar for intro screens (welcome, onboarding, confetti)
# Trigger confetti immediately on first render of the confetti screen
ss.setdefault("_confetti_shown", False)
if ss.get("intro_step", "") == "confetti" and not ss._confetti_shown:
    st.balloons()  # instant confetti effect on entry
    ss._confetti_shown = True

hidden_intro_steps = {"welcome", "onboarding", "confetti"}
main_tabs = {"Chat", "Exercises", "Diary", "Games", "CommuniGrow", "Badges & Logs", "Helpline"}
SHOW_SIDEBAR = (
    bool(ss.get("authenticated", False))
    and ss.get("intro_step", "") not in hidden_intro_steps
    and ss.get("active_tab", "Chat") in main_tabs
    and not ss.get("stealth", False)
)

# ========== SIDEBAR HIDDEN ==========
if not SHOW_SIDEBAR:
    st.markdown("""
    <style>
      [data-testid="stSidebar"] { display: none !important; }
      #mm-open-btn, #mm-close-btn { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # --- Secret unlock (tiny hotspot + timed popover) ---
    import datetime as _dt, sqlite3

    ss.setdefault("_secret_clicks", 0)
    ss.setdefault("_secret_window_start", None)
    ss.setdefault("stealth_unlock_until", None)

    now = _dt.datetime.now()
    # reset click window after 5s
    if ss._secret_window_start and (now - ss._secret_window_start).total_seconds() > 5:
        ss._secret_clicks = 0
        ss._secret_window_start = None

    # 1) Tiny hotspot (almost invisible)
    st.markdown("""
    <style>
      #mm-hotspot { position: fixed; right: 12px; bottom: 12px; z-index: 9999; }
      #mm-hotspot button {
        width: 22px; height: 22px; padding: 0; border-radius: 50%;
        background: rgba(255,255,255,0.03); border: none; color: transparent;
      } 
      #mm-hotspot button:hover { background: rgba(255,255,255,0.08); }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div id="mm-hotspot">', unsafe_allow_html=True)
    if st.button(" ", key="mm_hotspot_btn", help="Tap thrice"):
       if not ss._secret_window_start:
        ss._secret_window_start = now
       ss._secret_clicks += 1
       if ss._secret_clicks >= 3:
        ss.stealth_unlock_until = now + _dt.timedelta(seconds=20)
        ss._secret_clicks = 0
        ss._secret_window_start = None
       st.rerun()

    
    st.markdown('</div>', unsafe_allow_html=True)


    # 2) Small PIN popover (only visible for a short time)
    show_unlock = ss.get("stealth_unlock_until") and now < ss.stealth_unlock_until
    if show_unlock and ss.get("stealth", False):
        st.markdown("""
        <style>
          #mm-unlock {
            position: fixed; bottom: 26px; right: 44px; z-index: 10000;
            background: rgba(0,0,0,0.82); padding: 12px 12px; width: 200px;
            border-radius: 10px; border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 8px 20px rgba(0,0,0,0.45);
          }
          #mm-unlock p { color:#e6e6e6; margin:0 0 6px 0; font-size:.9rem; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div id="mm-unlock">', unsafe_allow_html=True)
        st.markdown("<p>🔒 Stealth</p>", unsafe_allow_html=True)

        pin_try = st.text_input(
            "PIN", type="password",
            key="stealth_pin_popup",
            placeholder="Enter PIN",
            label_visibility="collapsed",
        )

        if st.button("Unlock", key="stealth_unlock_btn"):
            # Use existing connection if available; otherwise open a short-lived one
            real_pin = ""
            try:
                cur = conn.execute("SELECT pin FROM users WHERE username=?", (ss.username,))
                row = cur.fetchone()
                real_pin = row[0] if row else ""
            except Exception:
                con2 = sqlite3.connect(os.path.join("data", "mannmitra.db"))
                cur2 = con2.cursor()
                cur2.execute("SELECT pin FROM users WHERE username=?", (ss.username,))
                r = cur2.fetchone()
                con2.close()
                real_pin = r[0] if r else ""

            if pin_try and pin_try == real_pin:
                ss.stealth = False
                ss.stealth_unlock_until = None
                st.rerun()
            else:
                st.error("Wrong PIN")

        st.markdown('</div>', unsafe_allow_html=True)



# ========== SIDEBAR VISIBLE ==========
else:
    # 1) Sidebar open/close behavior
    if ss.sidebar_open:
        st.markdown("""
        <style>
          [data-testid="stSidebar"] {
            width: 18rem !important;
            min-width: 18rem !important;
            transform: translateX(0) !important;
            opacity: 1 !important;
            visibility: visible !important;
            transition: all .25s ease;
            border-right: 1px solid rgba(255,255,255,0.08);
          }
          [data-testid="collapsedControl"],
          [data-testid="stSidebarCollapseButton"],
          button[title*="sidebar"] { display: none !important; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
          [data-testid="stSidebar"] {
            width: 0 !important;
            min-width: 0 !important;
            transform: translateX(-100%) !important;
            opacity: 0 !important;
            visibility: hidden !important;
            transition: all .25s ease;
          }
          [data-testid="collapsedControl"],
          [data-testid="stSidebarCollapseButton"],
          button[title*="sidebar"] { display: none !important; }
        </style>
        """, unsafe_allow_html=True)

    # 2) Floating open/close buttons
    st.markdown("""
    <style>
      #mm-open-btn, #mm-close-btn {
        position: fixed; top: 2px; left: 14px;
        z-index: 10000; overflow: visible;
      }
      #mm-open-btn button, #mm-close-btn button {
        font-size: 18px; padding: 6px 10px;
        border-radius: 10px; background: rgba(255,255,255,0.08);
        backdrop-filter: blur(5px);
        outline: none !important; box-shadow: none !important;
      }
      #mm-open-btn button:hover, #mm-close-btn button:hover {
        background: rgba(255,255,255,0.16);
      }
      header[data-testid="stHeader"] { z-index: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    if not ss.sidebar_open:
        st.markdown('<div id="mm-open-btn">', unsafe_allow_html=True)
        if st.button("☰", key="open_sidebar"):
            ss.sidebar_open = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div id="mm-close-btn">', unsafe_allow_html=True)
        if st.button("×", key="close_sidebar"):
            ss.sidebar_open = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # 3) Compact sidebar spacing
    st.markdown("""
    <style>
      section[data-testid="stSidebar"] .block-container {
          padding-top: 0.4rem !important;
          padding-bottom: 0.4rem !important;
      }
      section[data-testid="stSidebar"] * {
          margin-bottom: 0.2rem !important;
        
      
      }
    </style>
    """, unsafe_allow_html=True)


    # 4) Sidebar Content
    with st.sidebar:
        L = LANG[ss.lang]
        st.markdown("### MannMitra 🌿")

        # ---------- Appearance ----------
        st.markdown("*Appearance*")

        ss.setdefault("theme_mode", "Preset")
        ss.setdefault("theme_preset", "Slate")
        ss.setdefault("theme_custom", ss.applied_theme_hex)

        mode = st.radio("Color Mode", ["Preset", "Custom"],
                        key="theme_mode", horizontal=True)

        if mode == "Preset":
            preset = st.selectbox(
                "Button Color",
                list(ALLOWED_BUTTON_COLORS.keys()),
                index=list(ALLOWED_BUTTON_COLORS.keys()).index(ss.theme_preset)
                if ss.theme_preset in ALLOWED_BUTTON_COLORS else 0,
                key="theme_preset"
            )
            ss.applied_theme_hex = ALLOWED_BUTTON_COLORS[preset]

        else:
         # --- Custom CSS for widening the OK button ---
            st.markdown("""
              <style>
                /* Match the apply_custom_color button by its key */
                 div[data-testid*="apply_custom_color"] button {
                 width: 100% !important;      /* fills its column */
                 min-width: 250px !important;
                 max-width: 400px !important;
                 font-size: 16px !important;
                 height: 40px !important;
                 border-radius: 8px !important;
                 padding: 4px 0 !important;
              }
              </style>
              """, unsafe_allow_html=True)

            # --- Columns layout for color picker and OK button ---
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            with c1:
                 custom_color = st.color_picker(
                    "Pick a color", value=ss.theme_custom, key="theme_custom")
            with c2:
                 if st.button("OK", key="apply_custom_color"):
                    ss.applied_theme_hex = custom_color


        # ---------- Privacy ----------
        st.divider()
        st.markdown("*Privacy*")
        if not ss.stealth:
            if st.button(L["hide"], use_container_width=True, key="btn_hide"):
                ss.stealth = True
                st.rerun()
        else:
            pin_try = st.text_input(L["pin"], type="password", key="pin_unhide")
            st.markdown("<div style='height:7px'></div>", unsafe_allow_html=True)
            if st.button(L["unhide"], use_container_width=True, key="btn_unhide"):
                cur = conn.execute("SELECT pin FROM users WHERE username=?", (ss.username,))
                row = cur.fetchone()
                real_pin = row[0] if row else ""
                if pin_try and pin_try == real_pin:
                    ss.stealth = False
                    st.rerun()
                else:
                    st.error("Wrong PIN")

        # ---------- Quick Tabs ----------
        st.divider()
        st.markdown("*Quick Tabs*")
        tabs = ["Chat", "Exercises", "Diary", "Games", "CommuniGrow", "Badges & Logs", "Helpline"]
        current = ss.active_tab if ss.active_tab in tabs else "Chat"
        new_tab = st.radio(" ", tabs, index=tabs.index(current),
                           key="radio_tabs", label_visibility="collapsed")
        if new_tab != ss.active_tab:
            ss.active_tab = new_tab
            st.rerun()

        # ---------- Logout ----------
        st.divider()
        if st.button(L["logout"], use_container_width=True, key="btn_logout"):
            ss.authenticated = False
            ss.username = None
            ss.chat = []
            ss.intro_step = "welcome"
            ss.start_anim_time = None
            st.rerun()

# 5) Apply chosen or default theme globally
apply_global_button_theme_hex(ss.get("applied_theme_hex", DEFAULT_BUTTON_HEX))

# Keep Stop buttons red anywhere
st.markdown("""
<style>
.stop > button { background:#ef4444 !important; color:#fff !important; }
.stop > button:hover { filter:brightness(0.95); }
</style>
""", unsafe_allow_html=True)






def create_user(username, password, pin):
    pw_hash = hash_pw(password)
    username = username.strip()

    try:
        # Remember change counter before the insert
        before = conn.total_changes

        # If username is new -> row inserted; if it already exists -> insert is ignored (no error)
        conn.execute(
            "INSERT OR IGNORE INTO users(username, password_hash, pin) VALUES (?,?,?)",
            (username, pw_hash, pin)
        )
        conn.commit()

        # If total_changes increased, we actually created the account
        if conn.total_changes > before:
            return True, "Account created."
        else:
            return False, "Username already exists."
    except sqlite3.OperationalError as e:
        # Handles brief "database is locked" bursts on very fast double-clicks
        if "locked" in str(e).lower():
            return False, "Please wait a moment and try again."
        raise



def check_user(username, password):
    username = username.strip()
    cur = conn.execute("SELECT password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    if row:
        return bcrypt.checkpw(password.encode(), row[0]) if isinstance(row[0], (bytes, bytearray)) else (row[0] == password)
    return False


if not ss.authenticated:
    # Match pill width
    st.markdown("""
    <style>
      .block-container { max-width: var(--mm-pw) !important; margin-left:auto; margin-right:auto; }
    </style>
    """, unsafe_allow_html=True)

    # 1cm extra gap below pill for login page
    st.markdown("<div class='mm-viewport-top stealth-gap'><div class='mm-card'>", unsafe_allow_html=True)

    # Centered heading
    st.markdown("<div class='mm-title-center'>Login</div>", unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            u = st.text_input("User ID", key="login_uid")
            p = st.text_input("Password", type="password", key="login_pwd")
            submit = st.form_submit_button("Login", type="primary", use_container_width=True)
        if submit:
            if u and p and check_user(u, p):
                ss.authenticated = True
                ss.username = u
                ss.intro_step = "welcome"
                ss.start_anim_time = None
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab_signup:
        with st.form("signup_form", clear_on_submit=False):
          u2 = st.text_input("Choose User ID", key="signup_uid")
          p2 = st.text_input("Choose Password", type="password", key="signup_pwd")
          pin2 = st.text_input("Set 4-digit PIN (for stealth)", key="signup_pin")
    
        # ✅ Submit button must be INSIDE the form
          make = st.form_submit_button(
            "Create Account",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("signup_processing", False)
        )

        if make and not st.session_state.get("signup_processing", False):
            st.session_state["signup_processing"] = True
            try:
                if not (u2 and p2 and pin2 and pin2.isdigit() and len(pin2) == 4):
                    st.warning("Please fill all fields. PIN must be 4 digits.")
                else:
                    ok, msg = create_user(u2, p2, pin2)
                    if ok:
                        st.success("✅ Account created successfully!")
                    else:
                        st.error(f"⚠️ {msg}")
            finally:
                st.session_state["signup_processing"] = False



    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()







# ---------------------- Stealth screen ----------------------
if ss.stealth:
    st.markdown("<div class='mm-viewport-top'><div class='mm-card'>", unsafe_allow_html=True)
    st.info(LANG[ss.lang]['stealth_screen'])
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# ---------------------- Intro sequence ----------------------
def welcome_card():
    # Centered under the pill, same width
    st.markdown("<div class='mm-viewport-top'><div class='mm-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='mm-title'>{LANG[ss.lang]['welcome']}</div>", unsafe_allow_html=True)

    # Caution message centered (your approved copy)
    body_html = """
    <div style='text-align:center; font-size:1rem; color:#f5f5f5; margin:10px 0 22px;'>
    ⚠ <b>Reminder:</b> MannMitra is your supportive space for reflection and emotional wellness.<br>
    It is <i>not a substitute</i> for professional therapy or medical care.<br>
    If you ever feel distressed or unsafe, please reach out to a trusted person or a mental health helpline immediately. 💛
    </div>
    """
    st.markdown(body_html, unsafe_allow_html=True)

    # Center the Start button with columns
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button(LANG[ss.lang]['start'], type="primary", key="btn_start", use_container_width=True):
            ss.intro_step = "confetti"
            ss.start_anim_time = time.time()
            st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)

def _confetti_css_and_layer():
    """Simple falling-confetti layer (centered screen; slower fall)."""
    import random
    st.markdown("""
    <style>
      .confetti-wrap { position: fixed; inset: 0; pointer-events:none; z-index: 0; }
      .confetti-piece {
        position:absolute; width:10px; height:14px; opacity:0.9;
        /* Slower fall & a bit smoother sway */
        animation: fall var(--dur) linear infinite, sway var(--sway) ease-in-out infinite;
        transform: translate3d(0, -100vh, 0) rotate(var(--rot));
      }
      @keyframes fall {
        0%   { transform: translate3d(var(--xstart), -110vh, 0) rotate(0deg); }
        100% { transform: translate3d(var(--xend),   110vh, 0) rotate(360deg); }
      }
      @keyframes sway { 0%,100% { margin-left:-12px; } 50% { margin-left:12px; } }
    </style>
    """, unsafe_allow_html=True)

    colors = ["#ff4d4f","#40a9ff","#73d13d","#faad14","#9254de","#13c2c2","#eb2f96"]
    pieces = []
    for _ in range(160):
        style = (
            f"--dur:{random.uniform(4.5,7.2):.2f}s;"   # <-- slower than before
            f"--sway:{random.uniform(2.6,4.6):.2f}s;"  # a touch smoother
            f"--rot:{random.randint(0,360)}deg;"
            f"--xstart:{random.randint(-24,24)}px;"
            f"--xend:{random.randint(-24,24)}px;"
            f"left:{random.randint(0,100)}%;"
            f"background:{random.choice(colors)};"
            f"width:{random.randint(8,14)}px;height:{random.randint(10,18)}px;"
        )
        pieces.append(f"<div class='confetti-piece' style='{style}'></div>")
    st.markdown(f"<div class='confetti-wrap'>{''.join(pieces)}</div>", unsafe_allow_html=True)


def confetti_screen():
    """Centered message & button, with slower confetti."""
    _confetti_css_and_layer()

    # Center the content in a card that matches your pill width
    st.markdown(
        "<div class='mm-viewport-center'>"
        "<div class='mm-card' style='text-align:center; max-width: var(--mm-pw);'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mm-body" style="margin-bottom: 14px;">
          <p><b>Hey there! 🌟💛</b></p>
          <p>You made it here — and that’s amazing! 🎉✨</p>
          <p>Take a deep breath 🌬, smile 😊, and give yourself a little moment 💖.</p>
          <p>This is your space 🌿🍃 to relax, feel lighter 🌈, and grow 🌸🌞.</p>
          <p>You’re not just opening an app — you’re starting a fun, caring journey with yourself! 🌟🎈💛</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Enter App", type="primary", key="btn_enter_app", use_container_width=True):
        st.session_state.intro_step = "done"
        st.session_state.active_tab = "Chat"
        st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)



def caution_screen():
    st.markdown("<div class='mm-viewport-top'><div class='mm-card'>", unsafe_allow_html=True)
    st.info("⚠ Reminder: MannMitra is a friendly companion, not professional care. Reach out to helplines for urgent support.")
    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------------- Router helper (unchanged) ----------------------
def goto(tab_name: str):
    if ss.active_tab != tab_name:
        ss.active_tab = tab_name
        st.rerun()

# ---------------------- Chat language picker ----------------------
def chat_language_picker():
    # Language selector removed per your design; keep English fixed.
    ss.lang = "English"



# MannMitra
import os
import re
import time
import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import speech_to_text
import streamlit.components.v1 as components

# ---------- ENV / GEMINI SETUP ----------
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except Exception:
    pass

def _get_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    if key[:1] in {"'", '"'} and key[-1:] in {"'", '"'}:
        key = key[1:-1].strip()
    if not key:
        try:
            key = (st.secrets.get("GEMINI_API_KEY", "") or "").strip()
        except Exception:
            key = ""
    return key

GEMINI_API_KEY = _get_api_key()
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")

HAS_GEMINI = False
GEMINI_MODEL = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel(
            MODEL_ID,
            generation_config={"temperature": 0.8, "top_p": 0.9, "top_k": 40, "max_output_tokens": 768},
        )
        HAS_GEMINI = True
    except Exception:
        HAS_GEMINI = False

ss = st.session_state
defaults = {
    "chats": {"friendly": [], "mentor": []},
    "chat_draft_widget": "",
    "_pending_mic_text": None,
    "_clear_draft": False,
    "_typing": False,
}
for k, v in defaults.items():
    if k not in ss:
        ss[k] = v

# ---------- STATIC DATA ----------
HELPLINES = [
    ("Tele-MANAS — 24×7", "14416 / 1800-891-4416"),
    ("NIMHANS Helpline", "080-4611-0007"),
    ("CHILDLINE", "1098"),
]

# ---------- CRISIS DETECTION ----------
_CRISIS_PATTERNS = [
    r"kill myself", r"end my life", r"self[-\s]?harm", r"hurt myself",
    r"cut myself", r"no reason to live", r"can't go on",
    r"goodbye forever", r"want to disappear", r"suicide", r"\\bdie\\b"
]
_CRISIS_RE = re.compile(r"(?i)(" + r"|".join([p.replace(" ", r"\\s+") for p in _CRISIS_PATTERNS]) + r")")

def detect_distress(text: str) -> bool:
    return bool(text) and bool(_CRISIS_RE.search(text))

def helplines_md() -> str:
    lines = ["**If you're feeling unsafe or having thoughts of self-harm, please reach out immediately. You are not alone. 💛**\n"]
    for name, num in HELPLINES:
        lines.append(f"- **{name}** — {num}")
    return "\n".join(lines)

# ---------- TTS CLEANING ----------
def remove_emojis_and_symbols(text: str) -> str:
    emoji_re = re.compile(
        "["u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_re.sub("", text)
    return re.sub(r"[*_~`|#@]", "", text)

def tts_toggle_button_html(text: str, btn_id: str, rate: float = 1.0) -> str:
    clean = remove_emojis_and_symbols(text).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return f"""
    <button id="{btn_id}" class="tts-btn">🔈</button>
    <script>
      const btn = document.getElementById("{btn_id}");
      btn.onclick = () => {{
        const synth = window.speechSynthesis;
        if (synth.speaking) {{ synth.cancel(); return; }}
        const u = new SpeechSynthesisUtterance("{clean}");
        u.rate = {rate};
        synth.speak(u);
      }};
    </script>
    """

# ---------- SAFE GEMINI CALL ----------
def safe_generate_reply(prompt: str) -> str:
    if not HAS_GEMINI or GEMINI_MODEL is None:
        return ""
    try:
        resp = GEMINI_MODEL.generate_content(prompt)
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
    except Exception:
        pass
    return ""

# ---------- STYLING ----------
st.set_page_config(page_title="MannMitra 🌿", layout="wide")
st.markdown("""
<style>
header, footer, #MainMenu {visibility: hidden;}
body, .stApp { background-color: #1E1F20 !important; color: #ECECEC !important; }

.app-wrap { max-width: 1100px; margin: 0 auto 10px; padding: 0 20px; }

/* Topbar / Heading */
.topbar { position: sticky; top: 0; z-index: 1100;
          background: linear-gradient(180deg, rgba(30,31,32,1) 0%, rgba(30,31,32,0.85) 70%, rgba(30,31,32,0) 100%);
          padding-top: 6px 0 2px; margin-bottom: 0px; }
.topbar { margin-bottom: 0; padding-bottom: 2px; }
.hdr { display: flex; align-items: center; margin: 0; }
.hdr h1 { font-family: 'Inter', sans-serif; font-size: 24px; font-weight: 700; color: #ECECEC; margin: 0; }
.hdr h1 .leaf { color: #00A67E; margin-left: 8px; }
.hdr { margin-top: -70px; }  /* or -4px if you want tighter */


/* Select boxes row */
.stSelectbox > div > div {
  background: #2A2B2D !important; border-radius: 8px !important; color: #F1F1F1 !important; border: 1px solid #3A3B3D !important;
}
.stSelectbox label { color: #BFBFBF !important; font-weight: 500; font-size: 14px; margin-bottom: 6px; }

/* Chat card */
.chat-card { background:#232425; border:1px solid #333436; border-radius:16px; padding:18px; margin: 10px 0 20px;
             box-shadow:0 6px 18px rgba(0,0,0,0.3); }

/* Messages */
.msg { display:inline-block; max-width:78%; padding:14px 18px; border-radius:16px; margin-bottom:14px; font-size:15px; line-height:1.5; }
.msg-user { background:#10A37F; color:#fff; margin-left:auto; border-top-right-radius:6px; }
.msg-assistant { background:#2E2F31; color:#EAEAEA; margin-right:auto; border-top-left-radius:6px; }

/* Bottom input dock */
.input-bar { position: fixed; left:0; right:0; bottom:16px; display:flex; justify-content:center; z-index:1000; }
.input-inner { max-width: 960px; display: grid; grid-template-columns: 1fr 96px 96px 96px; gap: 10px; align-items: center; padding: 0 16px; }

/* Text input */
.stTextInput > div > div > input {
  height:56px !important; border-radius:12px !important; padding:0 16px !important;
  background:#2A2B2D !important; color:#ffffff !important; border:1px solid #3A3B3D !important; font-size:16px !important;
  box-shadow: 0 2px 0 rgba(0,0,0,0.25) inset, 0 4px 16px rgba(0,0,0,0.25);
}
.stTextInput > div > div:focus-within { box-shadow: 0 0 0 1px #2F3A39, 0 0 0 3px rgba(16,163,127,0.25); border-radius:12px; }

/* Buttons: SAME size & SAME color */
.stButton button, .st_mic_recorder > button, .tts-btn {
  height:48px !important; width:80px !important; border-radius:12px !important; border:none !important;
  background: linear-gradient(135deg, #10A37F 0%, #12B088 100%) !important; color:#fff !important; font-size:18px !important;
  box-shadow: 0 10px 24px rgba(16,163,127,0.28); transition: transform 0.15s ease, box-shadow 0.15s ease;
}
/* make the custom HTML tts button behave like a Streamlit button */
.tts-btn { display:inline-flex; align-items:center; justify-content:center; cursor:pointer; }

.stButton button:hover, .st_mic_recorder > button:hover, .tts-btn:hover {
  transform: translateY(-2px); box-shadow: 0 14px 34px rgba(16,163,127,0.36);
}
.st_mic_recorder { width: 100%; } .st_mic_recorder > button { width: 100% !important; }
</style>
""", unsafe_allow_html=True)


# ---------- MAIN CHAT ----------
# ---------- MAIN CHAT ----------
def page_chat():
    st.markdown('<div class="app-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="topbar"><div class="hdr"><h1>MannMitra <span class="leaf">🌿</span></h1></div></div>', unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:-8px;'></div>", unsafe_allow_html=True)
    sel_l, sel_r = st.columns([0.5, 0.5])
    with sel_l:
        persona = st.selectbox("Mode", ["Friendly", "Mentor"], key="chat_persona")
    with sel_r:
        language = st.selectbox("Language", ["English", "Hindi", "Hinglish"], key="chat_lang")

    chat_key = "friendly" if persona == "Friendly" else "mentor"
    history = ss.chats.get(chat_key, [])

    # 🧠 FIX: handle mic + clear flags BEFORE creating text_input
    if ss._pending_mic_text:
        ss.chat_draft_widget = ss._pending_mic_text
        ss._pending_mic_text = None
    elif ss._clear_draft:
        ss.chat_draft_widget = ""
        ss._clear_draft = False

    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    for idx, (role, txt) in enumerate(history):
        cls = "msg-assistant" if role == "assistant" else "msg-user"
        align = "flex-start" if role == "assistant" else "flex-end"
        st.markdown(
            f'<div style="display:flex; justify-content:{align};">'
            f'<div class="msg {cls}">{txt}</div></div>',
            unsafe_allow_html=True
        )
        if role == "assistant":
            components.html(tts_toggle_button_html(txt, f"tts-bubble-{idx}"), height=64)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- bottom bar ---
    st.markdown('<div class="input-bar"><div class="input-inner">', unsafe_allow_html=True)
    col_text, col_send, col_mic, col_speak = st.columns([1.0, 0.18, 0.18, 0.18])

    with col_text:
        st.text_input("Type your message", key="chat_draft_widget",
                      placeholder="Type here…", label_visibility="collapsed")

    with col_send:
        send_clicked = st.button("➡", key="send_btn", use_container_width=True)

    with col_mic:
        mic_text = speech_to_text(
            language={"English": "en-IN", "Hindi": "hi-IN", "Hinglish": "en-IN"}.get(language, "en-IN"),
            use_container_width=True, just_once=True, key="stt_browser",
            start_prompt="🎤", stop_prompt="■"
        )
        if mic_text:
            ss._pending_mic_text = mic_text
            st.rerun()

    with col_speak:
        last_assistant = next((t for r, t in reversed(history) if r == "assistant"), "")
        if last_assistant:
            components.html(tts_toggle_button_html(last_assistant, "global-tts"), height=64)
        else:
            components.html('<div style="height:64px;"></div>', height=64)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # handle send
    if send_clicked:
        msg = (ss.chat_draft_widget or "").strip()
        if msg:
            ss.chats.setdefault(chat_key, []).append(("user", msg))
            ss._typing = True
            ss._clear_draft = True
            st.rerun()

    if ss.get("_typing", False):
        time.sleep(1.2)
        ss._typing = False
        last_msg = ss.chats[chat_key][-1][1]
        persona_prompt = (
            "Use a warm, friendly tone with emojis and empathy."
            if persona == "Friendly"
            else "Be a structured, concise mentor. No emojis."
        )
        prompt = f"You are MannMitra 🌿.\nPersona: {persona}\nLanguage: {language}\n{persona_prompt}\nUser: {last_msg}"
        reply = safe_generate_reply(prompt) or (
            "Hey! I’m here with you — what’s on your mind? 😊"
            if persona == "Friendly"
            else "I’m listening. Tell me your concern clearly."
        )
        if detect_distress(last_msg):
            reply = helplines_md() + "\n\n" + reply
        ss.chats[chat_key].append(("assistant", reply))
        ss._clear_draft = True
        st.rerun()



# ---------- EXERCISES PAGE ----------
# ---------------------- IMPORTS ----------------------
from time import time as _now

import time, json
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

ss = st.session_state

# ===== music tracks =====
MUSIC_TRACKS = {
    "🎧 Calm Music — Forest": "assets/calm_sleep1.mp3",
    "🎧 Calm Music — Piano": "assets/calm_sleep2.mp3",
    "🎧 Calm Music — Ocean": "assets/calm_sleep3.mp3",
}
# background music for timed exercises
MUSIC_BG = "assets/exercise.mp3"

# ---------- styles ----------
st.markdown(f"""
<style>
/* ----- Exercises control buttons only ----- */
div[data-ex="controls"] .stButton > button {{
  background: #60a5fa !important;   /* Start color (violet) */
  color: #ffffff !important;
  border: 1px solid transparent !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 16px rgba(0,0,0,.25);
}}
div[data-ex="controls"] .stButton > button:hover {{
  filter: brightness(0.95);
  box-shadow: 0 0 0 3px rgba(167, 139, 250, .35); /* soft ring */
}}
div[data-ex="controls"] .stButton > button:focus {{
  outline: none;
  box-shadow: 0 0 0 3px rgba(167, 139, 250, .55) !important;
}}

/* Stop = red variant */
div[data-ex="controls"] .stop > button {{
  background: #ef4444 !important;   /* red */
}}
div[data-ex="controls"] .stop > button:hover {{
  filter: brightness(0.95);
}}
</style>
""", unsafe_allow_html=True)


# ---------- helper beep ----------
def _beep_js(hz=660, dur_ms=150):
    components.html(
        f"""
        <script>
        (function(){{
          try {{
            if (navigator.vibrate) navigator.vibrate([{int(dur_ms*0.8)}]);
            const AC = window.AudioContext || window.webkitAudioContext;
            const ctx = new AC();
            const osc = ctx.createOscillator();
            const g = ctx.createGain();
            osc.type = "sine";
            osc.frequency.value = {hz};
            osc.connect(g); g.connect(ctx.destination);
            g.gain.setValueAtTime(0.0001, ctx.currentTime);
            osc.start();
            g.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + 0.01);
            g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + {dur_ms}/1000);
            osc.stop(ctx.currentTime + {dur_ms}/1000 + 0.02);
          }} catch(e){{}}
        }})();
        </script>
        """,
        height=0,
    )

# ---------- helper: audio player widget ----------
def _bg_music_widget(show: bool, src: str):
    """Show a visible audio player for background music."""
    if show:
        try:
            with open(src, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
        except FileNotFoundError:
            st.error(f"Background track not found at {src}. Please check the path.")


def _start_session(ex, cycles, category_name):
    ss.update(
        ex_running=True, ex_paused=False, ex_cycle_left=int(cycles),
        ex_total_cycles=int(cycles), ex_pattern=ex["pattern"],
        ex_name=ex["name"], ex_category=category_name,
        ex_phase_idx=0, ex_phase_started_at=_now()
    )

def _advance_timer():
    """Advance the timer one 'tick' (called every rerender)."""
    if not ss.get("ex_running") or ss.get("ex_paused"):
        return

    pattern = ss["ex_pattern"]
    i = ss.get("ex_phase_idx", 0)
    phase_name, secs = pattern[i]

    elapsed = int(_now() - ss.get("ex_phase_started_at", _now()))
    if elapsed >= secs:
        # move to next phase
        i += 1
        if i >= len(pattern):
            # cycle finished
            ss["ex_phase_idx"] = 0
            ss["ex_cycle_left"] -= 1
            _beep_js()
            if ss["ex_cycle_left"] <= 0:
                ss["ex_running"] = False
                return
        else:
            ss["ex_phase_idx"] = i
        ss["ex_phase_started_at"] = _now()

def _render_runner_ui():
    """Draw current phase and schedule next tick. Non-blocking."""
    if not (ss.get("ex_running") and ss.get("ex_category")):
        return

    pattern = ss["ex_pattern"]
    i = ss.get("ex_phase_idx", 0)
    phase_name, secs = pattern[i]

    elapsed = int(_now() - ss.get("ex_phase_started_at", _now()))
    t_show = min(max(elapsed + 1, 1), secs)

    curr_cycle = ss["ex_total_cycles"] - ss["ex_cycle_left"] + 1
    st.success(f"Cycle {curr_cycle} started…")
    st.markdown(f"<div class='bigphase'>{phase_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='counter'>{t_show}</div>", unsafe_allow_html=True)

    if "Inhale" in phase_name:
        st.markdown("<div class='coach'>Inhale slowly…</div>", unsafe_allow_html=True)
    elif "Exhale" in phase_name:
        st.markdown("<div class='coach'>Exhale gently…</div>", unsafe_allow_html=True)
    elif "Hold" in phase_name:
        st.markdown("<div class='coach'>Hold your breath…</div>", unsafe_allow_html=True)
    elif "Tap" in phase_name:
        st.markdown("<div class='coach'>Tap left–right…</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='coach'>Stay mindful…</div>", unsafe_allow_html=True)

    # one-second tick, then advance and re-render
    time.sleep(1)
    _advance_timer()
    st.rerun()



# ---------- state setup ----------
def _ensure_state():
    ss.setdefault("ex_running", False)
    ss.setdefault("ex_paused", False)
    ss.setdefault("ex_cycle_left", 0)
    ss.setdefault("ex_total_cycles", 0)
    ss.setdefault("ex_pattern", [])
    ss.setdefault("ex_name", "")
    ss.setdefault("ex_category", "")
    ss.setdefault("music_active", False)
    ss.setdefault("music_owner_cat", "")

# ---------- data ----------
PMR_GROUPS = [
    "Toes","Calves","Thighs","Stomach","Chest","Hands","Forearms",
    "Upper arms","Shoulders","Neck","Jaw","Eyes/Forehead",
]

EXERCISE_SETS = {
    "Relax & De-stress": [
        {
            "name": "Box Breathing (4-4-4-4)",
            "about": (
                "Breathe in for 4 seconds, hold for 4 seconds, breathe out for 4 seconds, "
                "then hold again for 4. A steady rhythm to calm your nervous system."
            ),
            "when": "Feeling tense or before class; quick calm reset.",
            "pattern": [("Inhale",4),("Hold",4),("Exhale",4),("Hold",4)],
            "default_cycles":4,"max_cycles":12,
        },
        {
            "name": "Physiological Sigh",
            "about": (
                "Take a deep inhale, add a small top-up inhale, then long exhale. "
                "Reduces tension fast when stressed."
            ),
            "when": "Sudden stress or anxiety spikes.",
            "pattern": [("Inhale",2),("Top-up inhale",1),("Exhale (long)",6)],
            "default_cycles":6,"max_cycles":12,
        },
        {
            "name": "Bhramari (Humming Breath)",
            "about": (
                "Inhale gently, then exhale with a soft 'mmm' humming sound to relax the face and jaw."
            ),
            "when": "Overthinking; need gentle focus.",
            "pattern": [("Inhale",3),("Hum exhale",6)],
            "default_cycles":8,"max_cycles":15,
        },
        {
            "name": "5-4-3-2-1 Grounding",
            "about": (
                "Anchor to the present: name 5 things you can see, 4 touch, 3 hear, 2 smell, 1 taste."
            ),
            "when": "Anxious or panicky; need grounding.",
            "pattern": [
                ("See 5 things",25),("Touch 4 things",20),
                ("Hear 3 sounds",15),("Smell 2 scents",10),("Taste/Notice 1",5),
            ],
            "default_cycles":1,"max_cycles":2,
        },
        {
            "name": "Butterfly Hug Tapping",
            "about": (
                "Cross arms, rest hands on upper arms, tap left–right while breathing slowly."
            ),
            "when": "Overwhelm; need self-soothing.",
            "pattern": [("Tap L-R + slow breath",20),("Rest/Notice",10)],
            "default_cycles":3,"max_cycles":8,
        },
    ],
    "Focus & Exam Calm": [
        {
            "name": "2-Minute Visual Reset (20-20-20)",
            "about": "Look 20 feet away for 20 sec, blink and relax eyes.",
            "when": "Eye strain or screen fatigue.",
            "pattern": [("Look 20 ft away",20),("Blink/Relax",5)],
            "default_cycles":6,"max_cycles":12,
        },
        {
            "name": "Finger-Trace Square",
            "about": "Trace an imaginary square: inhale, hold, exhale, hold.",
            "when": "Classroom-safe steady breathing.",
            "pattern": [("Inhale",4),("Hold",2),("Exhale",4),("Hold",2)],
            "default_cycles":4,"max_cycles":12,
        },
        {
            "name": "Micro-Stretch for Study",
            "about": (
                "Loosen neck, wrists, and eyes to refresh focus. "
                "Gentle, no pain—just ease tension."
            ),
            "when": "During study or work breaks.",
            "pattern": [("Neck rolls",20),("Wrist circles",20),("Eye figure-8",20)],
            "default_cycles":2,"max_cycles":6,
        },
    ],
    "Sleep & Relaxation": [
        {"name": "🎧 Calm Music — Forest", "mode":"music",
         "about":"Play calming forest sounds.", "when":"Relax or sleep.",
         "src":"assets/calm_sleep1.mp3"},
        {"name": "🎧 Calm Music — Piano", "mode":"music",
         "about":"Gentle piano background.", "when":"Relax quietly.",
         "src":"assets/calm_sleep2.mp3"},
        {"name": "🎧 Calm Music — Ocean", "mode":"music",
         "about":"Soft rain ambience.", "when":"Sleep or meditation.",
         "src":"assets/calm_sleep3.mp3"},
        {
            "name": "4-7-8 Breathing",
            "about": "Inhale 4s, hold 7s, exhale 8s — helps body wind down.",
            "when": "Bedtime relaxation.",
            "pattern": [("Inhale",4),("Hold",7),("Exhale",8)],
            "default_cycles":4,"max_cycles":8,
        },
        {
            "name": "Progressive Muscle Relaxation (PMR)",
            "about": "Tense for ~5s, release for ~5s across body areas.",
            "when": "Full-body release before sleep.",
            "pattern": sum([[(f"Tense: {g}",5),("Release",5)] for g in PMR_GROUPS], []),
            "default_cycles":1,"max_cycles":2,
        },
    ],
}



# ---------- MAIN UI ----------
def _exercise_picker_ui(category_name: str):
    _ensure_state()
    st.subheader(category_name)

    ex_list = EXERCISE_SETS[category_name]
    names = [e["name"] for e in ex_list]
    # header row: picker + cycles (or just picker for music)
    c1, c2 = st.columns([3, 1])
    with c1:
        sel = st.selectbox("Choose an exercise", names, key=f"sel_{category_name}")
    ex = next(e for e in ex_list if e["name"] == sel)
    is_music = ex.get("mode") == "music"

    if not is_music:
        with c2:
            cycles = st.number_input(
                "Cycles",
                1,
                ex.get("max_cycles", 10),
                ex.get("default_cycles", 3),
                step=1,
                key=f"cycles_{category_name}",
            )
    else:
        cycles = 0

    # metadata -> expander (closed by default)
    with st.expander("About & tips", expanded=False):
        st.markdown(
            f"<div class='aboutbox muted'><b>About:</b> {ex.get('about','')}<br/>"
            f"<b>When to use:</b> {ex.get('when','')}</div>",
            unsafe_allow_html=True,
        )

    # extra options (bg music for timed exercises)
    if not is_music:
        with st.expander("More options", expanded=False):
            play_bg_music = st.toggle(
                "Show background track player",
                value=False,
                key=f"bgm_{category_name}",
                help="Opens a small player for the built-in exercise track.",
            )
            _bg_music_widget(play_bg_music, MUSIC_BG)
    else:
        play_bg_music = False

    # controls (only show what matters)
    # controls (only show what matters)
    is_active = ss.get("ex_running") and ss.get("ex_category") == category_name
    c1b, c2b, c3b, c4b = st.columns(4)

    start_clicked = c1b.button("▶ Start", use_container_width=True, key=f"start_{category_name}")
    pause_clicked = c2b.button("⏸ Pause", use_container_width=True, key=f"pause_{category_name}") if is_active else False
    resume_clicked = c3b.button("⏯ Resume", use_container_width=True, key=f"resume_{category_name}") if is_active else False
    stop_clicked = c4b.button("⏹ Stop", use_container_width=True, key=f"stop_{category_name}")

    # ---- start / pause / resume / stop logic (unchanged) ----
        # start
    if start_clicked:
        if is_music:
            ss["music_active"] = True
            ss["music_owner_cat"] = category_name
            ss["ex_running"] = False
        else:
            ss["music_active"] = False
            ss["music_owner_cat"] = ""
            _start_session(ex, cycles, category_name)

    # pause/resume
    if not is_music and pause_clicked and ss.get("ex_running") and ss.get("ex_category") == category_name:
        ss["ex_paused"] = True
    if not is_music and resume_clicked and ss.get("ex_running") and ss.get("ex_category") == category_name:
        ss["ex_paused"] = False

    # stop
    if stop_clicked:
        if is_music and ss.get("music_active") and ss.get("music_owner_cat") == category_name:
            ss["music_active"] = False
            ss["music_owner_cat"] = ""
        if ss.get("ex_running") and ss.get("ex_category") == category_name:
            ss.update(ex_running=False, ex_paused=False, ex_cycle_left=0, ex_total_cycles=0, ex_pattern=[])


    # music-only items keep a single compact player row
        # play music if active
    if is_music and ss.get("music_active") and ss.get("music_owner_cat") == category_name:
        try:
            st.audio(ex.get("src"))
        except Exception:
            st.info(f"Missing audio: {ex.get('src')}")

    # non-blocking runner (so Pause/Resume work instantly)
    _render_runner_ui()

    # completion toast when done
    if (not ss.get("ex_running")
        and ss.get("ex_total_cycles", 0) > 0
        and ss.get("ex_cycle_left", 0) == 0):
        st.success("Exercise complete! 🥳")
  

# ---------- PAGE ----------
def page_exercises():
    st.markdown("## Exercises 🧘")
    st.write("Pick what you need—calm, focus, or sleep—then Start. The app will guide your breathing and timing.")
    tabs = st.tabs(list(EXERCISE_SETS.keys()))
    for tab,cat in zip(tabs,EXERCISE_SETS.keys()):
        with tab:
            _exercise_picker_ui(cat)





# ---------------------- Games (puzzles one-by-one, hint on demand) ----------------------
# ====================== IMPORTS & BANKS ======================
import streamlit as st
import numpy as np
import time

PUZZLES_BANK = [
    ("What has to be broken before you can use it?", "egg", "It’s fragile and in a shell."),
    ("I speak without a mouth and hear without ears. What am I?", "echo", "Think of sound bouncing."),
    ("What month has 28 days?", "all", "Every month has this many days."),
    ("What gets wetter the more it dries?", "towel", "Used after a bath."),
    ("What has many keys but can’t open a lock?", "piano", "It makes music."),
    ("What can you catch but not throw?", "cold", "A common illness."),
    ("What invention lets you look right through a wall?", "window", "Transparent."),
    ("What goes up but never comes down?", "age", "Time-related."),
    ("What has a head and a tail but no body?", "coin", "Money."),
    ("I’m tall when I’m young, and short when I’m old. What am I?", "candle", "Wax."),
    ("What has hands but can’t clap?", "clock", "On the wall."),
    ("What has a heart that doesn’t beat?", "artichoke", "A vegetable."),
    ("What has one eye but cannot see?", "needle", "Sewing tool."),
    ("What has words but never speaks?", "book", "Paper pages."),
    ("What runs but never walks?", "water", "A river."),
    ("What has a neck but no head?", "bottle", "Glass or plastic."),
    ("What has an eye but cannot see and is in a storm?", "hurricane", "Weather."),
    ("What can fill a room but takes up no space?", "light", "Switch it on."),
    ("What gets sharper the more you use it?", "brain", "You’re using it now."),
    ("What has keys, space, and no locks or rooms?", "keyboard", "Typing."),
    ("What begins with T, ends with T, and has T in it?", "teapot", "Tea time."),
    ("What gets bigger the more you take away?", "hole", "Dig it."),
    ("What has a thumb and four fingers but isn’t alive?", "glove", "Wear it."),
    ("What belongs to you but others use it more?", "name", "People call you."),
    ("What can you break, even if you never pick it up?", "promise", "Keep it!"),
    ("What kind of room has no doors or windows?", "mushroom", "Not a building."),
    ("Where does today come before yesterday?", "dictionary", "Alphabetical."),
    ("What has many teeth but can’t bite?", "comb", "Hair."),
    ("What can travel around the world while staying in a corner?", "stamp", "On letters."),
    ("What can you keep after giving to someone?", "word", "Or 'your word'."),
    ("What has four wheels and flies?", "garbagetruck", "Sanitation vehicle."),
    ("I shave every day, but my beard stays the same. Who am I?", "barber", "A job."),
]

THOUGHT_DETECTIVE_CARDS = [
    {
        "scenario": "Sleep: 5h (late bedtime) • Caffeine: 3 cups • Steps: 2k • Screen: doomscrolled 1h • Social: argued with friend • Self-talk: harsh",
        "choices": ["irritability","calm","energized","content"],
        "answer": "irritability",
        "why": "Short sleep + high caffeine + conflict + harsh self-talk → irritability likely.",
        "tactic": "Try 4-7-8 breathing + short walk + kind self-talk reframe."
    },
    {
        "scenario": "Sleep: 8h (regular) • Caffeine: 1 cup • Steps: 9k • Screen: minimal • Social: lunch with friend • Self-talk: appreciative",
        "choices": ["calm","stressed","lonely","overwhelmed"],
        "answer": "calm",
        "why": "Steady sleep, movement, positive social contact, and kind self-talk support calm.",
        "tactic": "Keep a gratitude note + gentle stretch in the evening."
    },
    {
        "scenario": "Sleep: 6h (woke twice) • Caffeine: 0 • Steps: 1.5k • Screen: news binge • Social: none • Self-talk: worried",
        "choices": ["low energy","energized","content","irritability"],
        "answer": "low energy",
        "why": "Poor sleep + low movement + anxious media + isolation → low energy.",
        "tactic": "10-minute sunlight walk + box breathing (4-4-4-4)."
    },
    {
        "scenario": "Sleep: 7.5h • Caffeine: 2 cups late afternoon • Steps: 7k • Screen: gaming late • Social: fine • Self-talk: neutral",
        "choices": ["restless","calm","sad","angry"],
        "answer": "restless",
        "why": "Late caffeine + late screens can create restlessness despite okay sleep length.",
        "tactic": "Swap late caffeine for water; screen-off 60m before bed."
    },
    {
        "scenario": "Sleep: 4.5h (deadline) • Caffeine: 4 cups • Steps: 3k • Screen: work late • Social: skipped messages • Self-talk: 'not enough'",
        "choices": ["overwhelmed","content","relaxed","energized"],
        "answer": "overwhelmed",
        "why": "Sleep debt + pressure + isolation + self-criticism → overwhelmed.",
        "tactic": "1-page brain dump + prioritize top 1–2 tasks + 5 slow breaths."
    },
    {
        "scenario": "Sleep: 8h • Caffeine: 1 cup • Steps: 11k (nature walk) • Screen: light • Social: helped a neighbor • Self-talk: proud",
        "choices": ["content","guilty","jealous","irritable"],
        "answer": "content",
        "why": "Good sleep + movement + prosocial act + proud self-talk → contentment.",
        "tactic": "Savor win: write 3 sentences about the moment."
    },
    {
        "scenario": "Sleep: 6h • Caffeine: 2 cups • Steps: 5k • Screen: social media comparisons • Social: saw party pics w/o invite • Self-talk: 'I’m not fun'",
        "choices": ["lonely","relaxed","energized","calm"],
        "answer": "lonely",
        "why": "Social exclusion cues + negative comparison → feeling lonely.",
        "tactic": "Send 1 honest text to a friend + plan a small hangout."
    },
    {
        "scenario": "Sleep: 7h • Caffeine: 1 cup • Steps: 4k • Screen: constant notifications • Social: group chat heated • Self-talk: 'I must fix this'",
        "choices": ["stressed","calm","sleepy","content"],
        "answer": "stressed",
        "why": "Alert overload + conflict + fixing pressure → stress.",
        "tactic": "Silence alerts for 30m + 5-minute body scan."
    },
    {
        "scenario": "Sleep: 9h • Caffeine: 0 • Steps: 10k • Screen: creative video • Social: compliments received • Self-talk: grateful",
        "choices": ["energized","sad","angry","jealous"],
        "answer": "energized",
        "why": "Rested body + movement + positive input + gratitude → energy.",
        "tactic": "Channel energy: 10-minute stretch or mini-project sprint."
    },
    {
        "scenario": "Sleep: 6.5h • Caffeine: 2 cups • Steps: 3k • Screen: news late night • Social: cancelled plan • Self-talk: 'people don’t like me'",
        "choices": ["anxious","calm","content","energized"],
        "answer": "anxious",
        "why": "Late news + cancellation + negative story → anxiety.",
        "tactic": "Reframe story + 4-7-8 breathing + set a tiny social plan."
    },
    {
        "scenario": "Sleep: 7h • Caffeine: 3 cups • Steps: 2k • Screen: morning to night • Social: corrected harshly online • Self-talk: defensive",
        "choices": ["irritability","calm","content","sleepy"],
        "answer": "irritability",
        "why": "High caffeine + low steps + online conflict → irritability spikes.",
        "tactic": "Water + 8-minute walk + pause-before-reply rule."
    },
    {
        "scenario": "Sleep: 8h • Caffeine: 1 cup • Steps: 6k • Screen: music & art • Social: kind call with parent • Self-talk: encouraging",
        "choices": ["calm","overwhelmed","jealous","sad"],
        "answer": "calm",
        "why": "Balanced day + soothing media + supportive call → calm baseline.",
        "tactic": "Anchor this: 2-minute mindful breath to store the feeling."
    }
]

MANDALA_GRID_SIZES = [3,4,5,6,7]  # NxN

# ====================== HELPERS ======================
def _affirmation(success=True):
    import random
    pos = [
        "Beautifully done 🌟",
        "Nice work — breathe and smile 🌿",
        "You’ve got this 💪",
        "Great focus ✨",
        "Lovely progress 💛",
    ]
    neg = [
        "Oops — that’s okay, you’re learning 🌱",
        "Not quite. Try the next one with a fresh breath 💫",
        "Every try counts. Keep going 💖",
    ]
    return random.choice(pos if success else neg)

# ====================== PUZZLES ======================
def game_puzzles():
    ss = st.session_state
    st.subheader("Puzzle Challenge 🧩")
    if "pz" not in ss:
        ss.pz = {"order": [], "idx": 0, "score": 0, "show_hint": False,
                 "last_checked": False, "last_correct": None}

    top = st.columns(3)
    with top[0]:
        rounds = st.selectbox("Questions this round", [5, 10, 15], index=0, key="pz_rounds")
    with top[1]:
        if st.button("New Round"):
            order = np.random.default_rng().choice(len(PUZZLES_BANK), size=rounds, replace=False).tolist()
            ss.pz.update({"order": order, "idx": 0, "score": 0,
                          "show_hint": False, "last_checked": False, "last_correct": None})

    if not ss.pz["order"]:
        st.info("Click **New Round** to begin.")
        return
    if ss.pz["idx"] >= len(ss.pz["order"]):
        st.success(f"Round complete! Score: **{ss.pz['score']}/{len(ss.pz['order'])}** 🥳")
        return

    q_i = ss.pz["order"][ss.pz["idx"]]
    q, ans, hint = PUZZLES_BANK[q_i]
    st.write(f"**Q{ss.pz['idx']+1} of {len(ss.pz['order'])}**")
    st.markdown(f"**{q}**")

    c1, c2 = st.columns([3,1])
    with c1:
        user_ans = st.text_input("Your answer", key=f"pz_ans_{ss.pz['idx']}")
    with c2:
        if st.button("Hint"):
            ss.pz["show_hint"] = True
    if ss.pz["show_hint"]:
        st.caption(f"Hint: {hint}")

    b1, b2 = st.columns([1,1])
    if b1.button("Check Answer"):
        ok = (user_ans.strip().lower().replace(" ", "") == ans.replace(" ", ""))
        ss.pz["last_checked"] = True
        ss.pz["last_correct"] = ok
        if ok:
            ss.pz["score"] += 1
            st.success(f"Correct! ✅ {_affirmation(True)}")
        else:
            st.error(f"Oops! Incorrect. The answer is **{ans}**. {_affirmation(False)}")

    if b2.button("Next Question ➜") and ss.pz["last_checked"]:
        ss.pz["idx"] += 1
        ss.pz["show_hint"] = False
        ss.pz["last_checked"] = False
        ss.pz["last_correct"] = None
        st.rerun()

# ====================== NEW GAME: MEMORY FLIP ======================
import streamlit as st

def game_memory_flip():
    import time as _t
    import random as _rand

    ss = st.session_state
    st.subheader("Memory Flip 🧠✨")

    # ---------- init ----------
    if "mem" not in ss:
        ss.mem = {
            "rows": 4, "cols": 4,
            "cards": [],                 # list[str] emojis, length rows*cols
            "revealed": [],              # list[bool], same length as cards
            "matched": set(),            # set[int] of matched indices
            "first": None, "second": None,
            "moves": 0,
            "running": False, "won": False,
            "seed": int(_t.time()),      # unique keys per round
            "lock": False,               # block clicks during pending resolve
            "pending": False,            # True when 2nd flip is up & we need to resolve
            "pending_set_at": 0.0,       # time when second was flipped
        }

    # ---------- controls ----------
    size_map = {"4 × 4": (4,4), "4 × 5": (4,5), "5 × 6": (5,6)}
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        # Disable size change while running to avoid state mismatches
        size_label = st.selectbox("Grid", list(size_map.keys()),
                                  index=[*size_map].index(f"{ss.mem['rows']} × {ss.mem['cols']}")
                                  if (ss.mem["rows"], ss.mem["cols"]) in size_map.values() else 0,
                                  disabled=ss.mem["running"],
                                  key="mem_size_label")
    with c2:
        started = st.button("Start ▶", use_container_width=True,
                            disabled=ss.mem["running"], key="mem_start_btn")
    with c3:
        stopped = st.button("Stop ⏹", use_container_width=True,
                            disabled=not ss.mem["running"], key="mem_stop_btn")

    # Shuffle only during a round
    shuffle = st.button("Shuffle 🔀", use_container_width=True,
                        disabled=not ss.mem["running"], key="mem_shuffle_btn")

    def _new_deck(r, c):
        emojis = ["🌸","🐼","🦋","🌈","🍀","🌟","🍉","🎈","🐝","🧩","🧸","🍩",
                  "🌻","🫧","🍪","🐞","🍎","⚽","🎵","🎮","🍓","🌙","🍔","🧃"]
        pairs_needed = (r * c) // 2
        deck = emojis[:pairs_needed] * 2
        _rand.shuffle(deck)
        return deck

    # Stop: end round & clear board so we never render stale sizes
    if stopped:
        ss.mem.update({
            "running": False, "won": False,
            "cards": [], "revealed": [], "matched": set(),
            "first": None, "second": None, "moves": 0,
            "lock": False, "pending": False,
        })
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    # Start: deal for selected size
    if started:
        r, c = size_map[size_label]
        ss.mem.update({
            "rows": r, "cols": c,
            "cards": _new_deck(r, c),
            "revealed": [False] * (r*c),
            "matched": set(),
            "first": None, "second": None,
            "moves": 0, "running": True, "won": False,
            "seed": int(_t.time()), "lock": False,
            "pending": False, "pending_set_at": 0.0,
        })
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    # Shuffle during a round
    if shuffle and ss.mem["running"]:
        r, c = ss.mem["rows"], ss.mem["cols"]
        ss.mem.update({
            "cards": _new_deck(r, c),
            "revealed": [False] * (r*c),
            "matched": set(),
            "first": None, "second": None,
            "moves": 0, "won": False,
            "seed": int(_t.time()), "lock": False,
            "pending": False, "pending_set_at": 0.0,
        })
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    # ---------- guards ----------
    if not ss.mem["running"]:
        st.info("Pick a grid and press **Start** to begin.")
        return

    r, c = ss.mem["rows"], ss.mem["cols"]
    cards = ss.mem.get("cards", [])
    if len(cards) != r*c:
        # If something got out of sync, rebuild deck safely
        ss.mem.update({
            "cards": _new_deck(r, c),
            "revealed": [False] * (r*c),
            "matched": set(),
            "first": None, "second": None,
            "moves": 0, "won": False, "lock": False,
            "pending": False, "pending_set_at": 0.0,
            "seed": int(_t.time()),
        })
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    revealed = ss.mem["revealed"]
    if len(revealed) != r*c:
        # Keep lengths aligned to avoid IndexError
        ss.mem["revealed"] = [False] * (r*c)
        revealed = ss.mem["revealed"]

    matched = ss.mem["matched"]

    # ---------- flip helpers ----------
    def _start_pending():
        ss.mem["pending"] = True
        ss.mem["pending_set_at"] = _t.time()

    def _resolve_pending():
        """Resolve current pair after short display."""
        i1, i2 = ss.mem["first"], ss.mem["second"]
        if i1 is None or i2 is None:
            ss.mem["pending"] = False
            return
        if cards[i1] == cards[i2]:
            matched.update([i1, i2])
            ss.mem["first"] = None
            ss.mem["second"] = None
            ss.mem["pending"] = False
            if len(matched) == r*c:
                ss.mem["won"] = True
        else:
            # auto-hide
            revealed[i1] = False
            revealed[i2] = False
            ss.mem["first"] = None
            ss.mem["second"] = None
            ss.mem["pending"] = False

    def _flip(i: int):
        if ss.mem["lock"] or ss.mem["won"] or i in matched or revealed[i]:
            return
        revealed[i] = True
        if ss.mem["first"] is None:
            ss.mem["first"] = i
            # show first immediately
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
        elif ss.mem["second"] is None and i != ss.mem["first"]:
            ss.mem["second"] = i
            ss.mem["moves"] += 1
            # do NOT resolve yet — we want the user to *see* the second card first
            _start_pending()
            # show both cards first
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    # ---------- GRID RENDER ----------
    grid_cols = st.columns(c)
    for i in range(r*c):
        with grid_cols[i % c]:
            is_matched = (i in matched)
            is_up = revealed[i] or is_matched
            label = cards[i] if is_up else "⬜"
            disabled = is_matched or ss.mem["lock"] or ss.mem["pending"]  # lock clicks during pending
            if st.button(label, key=f"mem_card_{ss.mem['seed']}_{r}_{c}_{i}",
                         use_container_width=True, disabled=disabled):
                _flip(i)

    # ---------- PENDING RESOLUTION (after drawing so user sees the 2nd card) ----------
    if ss.mem["pending"]:
        elapsed = _t.time() - ss.mem["pending_set_at"]
        wait = 0.6 - elapsed
        if wait > 0:
            _t.sleep(wait)
        _resolve_pending()
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    # Win banner (non-blocking)
    if ss.mem["won"]:
        st.success(f"All pairs matched in **{ss.mem['moves']}** moves! 🌿 Proud of you.")

    # footer info
    f1, f2 = st.columns(2)
    f1.metric("Moves", ss.mem["moves"])
    f2.metric("Found", f"{len(matched)//2} pairs")




# ====================== NEW GAME: CATCHING LEAVES ======================
# ===== Catching Leaves 🍂 (auto-fall, speed control, low parallel) =====
import time
import random
import streamlit as st

def game_catching_leaves():
    st.subheader("Catching Leaves 🍃 — Move the basket and catch them!")

    ss = st.session_state
    if "leaves" not in ss:
        ss.leaves = {
            "running": False,
            "ended": False,
            "score": 0,
            "cols": 6,          # 6 horizontal lanes
            "rows": 8,          # 8 vertical rows (falling area)
            "basket": 3,        # start center-ish (0..cols-1)
            "leaves": [],       # list of {id, col, y(float), vy}
            "next_id": 0,
            "tick_ms": 110,
            "spawn_p": 0.30,    # spawn probability per tick (max one leaf per tick)
            "base_vy": 0.22,    # rows per tick
            "ends_at": None,
            "total_secs": 120,  # default 2 min
            "seed": int(time.time()),
            "move_dir": 0,      # -1 hold left, 0 none, +1 hold right
        }

    COLS = ss.leaves["cols"]
    ROWS = ss.leaves["rows"]
    GROUND_ROW = ROWS

    # ---------- Controls (single row) ----------
    c1, c2, c3 = st.columns(3)
    with c1:
        speed = st.selectbox("Speed", ["Calm", "Medium", "Swift"], index=1,
                             key=f"lv_speed_{ss.leaves['seed']}")
    with c2:
        density = st.selectbox("Spawn", ["Low", "Med", "High"], index=1,
                               key=f"lv_density_{ss.leaves['seed']}")
    with c3:
        timer_label = st.selectbox("Timer", ["60s", "2 min", "3 min", "4 min", "5 min"], index=1,
                                   key=f"lv_timer_{ss.leaves['seed']}")

    # Apply settings when not running
    if not ss.leaves["running"]:
        ss.leaves["tick_ms"] = {"Calm": 130, "Medium": 110, "Swift": 90}[speed]
        ss.leaves["spawn_p"] = {"Low": 0.22, "Med": 0.30, "High": 0.38}[density]
        ss.leaves["total_secs"] = {
            "60s": 60, "2 min": 120, "3 min": 180, "4 min": 240, "5 min": 300
        }[timer_label]
        steps = max(1, int((ss.leaves["total_secs"] * 1000) / ss.leaves["tick_ms"]))
        ss.leaves["base_vy"] = float(max(0.12, min(0.45, (ROWS * 0.95) / steps)))

    # ---------- Start / Pause / Reset ----------
    b1, b2, b3 = st.columns(3)
    if b1.button("Start ▶", use_container_width=True, key=f"lv_start_{ss.leaves['seed']}",
                 disabled=ss.leaves["running"]):
        ss.leaves.update({
            "running": True,
            "ended": False,
            "score": 0,
            "leaves": [],
            "next_id": 0,
            "basket": COLS // 2,
            "ends_at": time.time() + ss.leaves["total_secs"],
            "move_dir": 0,
        })
        st.rerun()

    if b2.button("Pause ⏸", use_container_width=True, key=f"lv_pause_{ss.leaves['seed']}"):
        ss.leaves["running"] = False
        ss.leaves["move_dir"] = 0  # stop holding on pause

    if b3.button("Reset 🔄", use_container_width=True, key=f"lv_reset_{ss.leaves['seed']}"):
        seed = int(time.time())
        ss.leaves.update({
            "running": False, "ended": False, "score": 0,
            "leaves": [], "next_id": 0, "basket": COLS // 2,
            "ends_at": None, "seed": seed, "move_dir": 0
        })
        st.rerun()

    # ---------- Update loop ----------
    now = time.time()
    if ss.leaves["running"] and ss.leaves["ends_at"] and now >= ss.leaves["ends_at"]:
        ss.leaves["running"] = False
        ss.leaves["ended"] = True

    if ss.leaves["running"]:
        # continuous basket move (one lane per tick while holding)
        if ss.leaves["move_dir"] != 0:
            ss.leaves["basket"] = int(min(COLS - 1, max(0, ss.leaves["basket"] + ss.leaves["move_dir"])))

        # fall
        for leaf in ss.leaves["leaves"]:
            leaf["y"] += leaf["vy"]

        # catch or miss
        remove_ids = set()
        for leaf in ss.leaves["leaves"]:
            if leaf["y"] >= GROUND_ROW:
                if leaf["col"] == ss.leaves["basket"]:
                    ss.leaves["score"] += 1
                remove_ids.add(leaf["id"])
        if remove_ids:
            ss.leaves["leaves"] = [l for l in ss.leaves["leaves"] if l["id"] not in remove_ids]

        # spawn at most ONE new leaf per tick
        if random.random() < ss.leaves["spawn_p"]:
            ss.leaves["leaves"].append({
                "id": ss.leaves["next_id"],
                "col": random.randrange(COLS),
                "y": 0.0,
                "vy": random.uniform(ss.leaves["base_vy"] * 0.9, ss.leaves["base_vy"] * 1.2),
            })
            ss.leaves["next_id"] += 1

    # ---------- Styles ----------
    st.markdown(
        """
        <style>
          .lv-wrap { border: 2px solid #ddd; border-radius: 14px; padding: 8px; background: #fff; }
          .lv-cell { height: 44px; display:flex; align-items:center; justify-content:center; }
          .lv-ground { height: 18px; background:#e8f6e8; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------- Render grid ----------
    buckets = {r: {} for r in range(ROWS)}
    for leaf in ss.leaves["leaves"]:
        r = min(ROWS - 1, max(0, int(leaf["y"])))
        c = min(COLS - 1, max(0, int(leaf["col"])))
        buckets[r][c] = "🍃"

    st.markdown('<div class="lv-wrap">', unsafe_allow_html=True)
    for r in range(ROWS):
        row_cols = st.columns(COLS)
        for c in range(COLS):
            with row_cols[c]:
                label = buckets.get(r, {}).get(c, " ")
                st.button(label, key=f"lv_cell_{ss.leaves['seed']}{r}{c}",
                          use_container_width=True, disabled=True)

    bcols = st.columns(COLS)
    for c in range(COLS):
        with bcols[c]:
            if c == ss.leaves["basket"]:
                st.button("🧺", key=f"lv_basket_{ss.leaves['seed']}_{c}",
                          use_container_width=True, disabled=True)
            else:
                st.markdown('<div class="lv-ground"></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Bottom controls (step move) ----------
    STEP = 1
    sp1, leftc, rightc, sp2 = st.columns([3, 1, 1, 3])
    with leftc:
        if st.button("⬅", key=f"lv_left_{ss.leaves['seed']}", use_container_width=True):
            ss.leaves["basket"] = max(0, ss.leaves["basket"] - STEP)
            st.rerun()
    with rightc:
        if st.button("➡", key=f"lv_right_{ss.leaves['seed']}", use_container_width=True):
            ss.leaves["basket"] = min(COLS - 1, ss.leaves["basket"] + STEP)
            st.rerun()

    # ---------- Hold / Release controls (continuous move) ----------
    h1, h2, h3 = st.columns([1.3, 1.3, 1.3])
    with h1:
        if st.button("Hold ◀", key=f"lv_hold_left_{ss.leaves['seed']}", use_container_width=True):
            ss.leaves["move_dir"] = -1
            if not ss.leaves["running"]:
                ss.leaves["basket"] = max(0, ss.leaves["basket"] - 1)
            st.rerun()
    with h2:
        if st.button("Release ⏹", key=f"lv_hold_stop_{ss.leaves['seed']}", use_container_width=True):
            ss.leaves["move_dir"] = 0
            st.rerun()
    with h3:
        if st.button("Hold ▶", key=f"lv_hold_right_{ss.leaves['seed']}", use_container_width=True):
            ss.leaves["move_dir"] = 1
            if not ss.leaves["running"]:
                ss.leaves["basket"] = min(COLS - 1, ss.leaves["basket"] + 1)
            st.rerun()

    # ---------- Score + time ----------
    remaining = max(0, int((ss.leaves["ends_at"] or now) - time.time())) if ss.leaves["ends_at"] else 0
    st.metric("Score", ss.leaves["score"])
    if ss.leaves["ends_at"]:
        elapsed = ss.leaves["total_secs"] - remaining
        st.progress(min(1.0, elapsed / max(1, ss.leaves["total_secs"])))
    else:
        st.progress(0.0)

    # ---------- Tick ----------
    if ss.leaves["running"]:
        time.sleep(ss.leaves["tick_ms"] / 1000.0)
        st.rerun()

    # ---------- End banner ----------
    if ss.leaves["ended"]:
        st.success(f"Nice! You caught {ss.leaves['score']} leaves. 🍃✨")


# ====================== MAZE HELPERS & GAME (unchanged logic) ======================
def _maze_carve(n: int):
    import random
    L = n if n % 2 == 1 else n + 1
    grid = [[1] * L for _ in range(L)]
    stack = [(0, 0)]
    grid[0][0] = 0

    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in ((2,0),(-2,0),(0,2),(0,-2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < L and 0 <= nc < L and grid[nr][nc] == 1:
                nbrs.append((nr, nc, dr, dc))
        if nbrs:
            nr, nc, dr, dc = random.choice(nbrs)
            grid[r + dr//2][c + dc//2] = 0
            grid[nr][nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()

    grid[L-1][L-1] = 0
    if L != n:
        start = 0
        grid = [row[start:start+n] for row in grid[start:start+n]]
    grid[0][0] = 0
    grid[n-1][n-1] = 0
    return grid

def _maze_has_path(grid, start=(0,0), goal=None):
    from collections import deque
    n = len(grid)
    if goal is None:
        goal = (n-1, n-1)
    sr, sc = start
    gr, gc = goal
    if grid[sr][sc] == 1 or grid[gr][gc] == 1:
        return False
    q = deque([(sr, sc)])
    seen = {(sr, sc)}
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            return True
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False

def _maze_force_connect(grid):
    n = len(grid)
    r, c = 0, 0
    grid[r][c] = 0
    while r < n - 1:
        r += 1
        grid[r][c] = 0
    while c < n - 1:
        c += 1
        grid[r][c] = 0
    return grid

def _new_solvable_maze(n: int, max_tries: int = 30):
    for _ in range(max_tries):
        g = _maze_carve(n)
        if _maze_has_path(g, (0,0), (n-1, n-1)):
            return g
    g = _maze_carve(n)
    return _maze_force_connect(g)

def _maze_win_banner(moves: int):
    st.markdown(
        f"""
        <div style="
            margin: 10px auto 14px auto;
            text-align:center;
            padding: 14px 16px;
            border: 1px solid #333333;
            border-radius: 10px;
            background: #0f0f0f;      /* black-ish background */
            color: #ffffff;            /* white text */
            max-width: 640px;
            font-size: 1.05rem;
        ">
          🎉 Nice pathfinding! You reached the cheese in <b>{moves}</b> moves. <br/>
          Take a breath and enjoy that little win. 🌿
        </div>
        """,
        unsafe_allow_html=True
    )


def game_mindful_maze():
    ss = st.session_state
    st.subheader("Mindful Maze 🧭 — Help 🐭 reach 🧀")

    if "maze2" not in ss:
        size_default = 8
        ss.maze2 = {
            "size": size_default,
            "grid": _new_solvable_maze(size_default),
            "pos": (0, 0),
            "goal": (size_default - 1, size_default - 1),
            "won": False,
            "moves": 0,
            "seed": int(time.time()),
        }

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        sizes = [6, 7, 8, 9, 10]
        size_pick = st.selectbox("Size", sizes, index=sizes.index(ss.maze2["size"]))
    with c2:
        if st.button("New (selected size)", key=f"mz_new_size_{size_pick}"):
            ss.maze2["size"] = size_pick
            ss.maze2["grid"] = _new_solvable_maze(size_pick)
            ss.maze2["pos"] = (0, 0)
            ss.maze2["goal"] = (size_pick - 1, size_pick - 1)
            ss.maze2["won"] = False
            ss.maze2["moves"] = 0
            ss.maze2["seed"] = int(time.time())
            st.rerun()
    with c3:
        if st.button("New Maze", key=f"mz_new_rand_{ss.maze2['seed']}"):
            n = ss.maze2["size"]
            ss.maze2["grid"] = _new_solvable_maze(n)
            ss.maze2["pos"] = (0, 0)
            ss.maze2["goal"] = (n - 1, n - 1)
            ss.maze2["won"] = False
            ss.maze2["moves"] = 0
            ss.maze2["seed"] = int(time.time())
            st.rerun()

    grid = ss.maze2["grid"]
    n = len(grid)
    mouse  = ss.maze2["pos"]
    cheese = ss.maze2["goal"]

    if ss.maze2.get("won"):
        _maze_win_banner(ss.maze2["moves"])

    st.markdown("""
    <style>
      .maze-cell { height: 34px; display:flex; align-items:center; justify-content:center; }
      @media (min-width: 768px){ .maze-cell { height: 38px; } }
    </style>
    """, unsafe_allow_html=True)

    gcols = st.columns(n)
    for r in range(n):
        for c in range(n):
            with gcols[c]:
                if (r, c) == mouse:
                    label = "🐭"
                elif (r, c) == cheese:
                    label = "🧀"
                else:
                    label = "◻️" if grid[r][c] == 0 else "🧱"
                st.button(label, key=f"mz_cell_{n}_{r}_{c}", use_container_width=True, disabled=True)

    def _move(dr, dc):
        if ss.maze2["won"]:
            return
        r, c = ss.maze2["pos"]
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
            ss.maze2["pos"] = (nr, nc)
            ss.maze2["moves"] += 1
            if ss.maze2["pos"] == cheese:
                ss.maze2["won"] = True
                st.success(f"You found the cheese in {ss.maze2['moves']} moves! Great navigation 🌿")
                try:
                    st.balloons()
                except Exception:
                    pass
                _maze_win_banner(ss.maze2["moves"])
        else:
            st.error("🧱 Wall ahead — try another calm step.")

    b = st.columns(4)
    if b[0].button("⬆️ UP", key=f"mz_up_{n}"):    _move(-1, 0); st.rerun()
    if b[1].button("⬇️ DOWN", key=f"mz_down_{n}"):  _move( 1, 0); st.rerun()
    if b[2].button("⬅️ LEFT", key=f"mz_left_{n}"):  _move( 0,-1); st.rerun()
    if b[3].button("➡️ RIGHT", key=f"mz_right_{n}"): _move( 0, 1); st.rerun()

# ====================== THOUGHT DETECTIVE ======================
def game_thought_detective():
    ss = st.session_state
    st.subheader("Mood Mystery 🕵️ – Trigger & Tactic Detective")

    def _td_new_order(round_len: int):
        rng = np.random.default_rng()
        total = len(THOUGHT_DETECTIVE_CARDS)
        size = min(round_len, total)
        return rng.choice(total, size=size, replace=False).tolist()

    current_ver = len(THOUGHT_DETECTIVE_CARDS)
    if "td2" not in ss:
        ss.td2 = {"order": [], "i": 0, "score": 0, "checked": False, "last_choice": None, "ver": current_ver}
    if ss.td2.get("ver") != current_ver:
        ss.td2 = {"order": [], "i": 0, "score": 0, "checked": False, "last_choice": None, "ver": current_ver}

    top = st.columns([1,1,2])
    with top[0]:
        round_len = st.selectbox("Cases this round", [5, 10], index=0, key="td_rounds")
    with top[1]:
        if st.button("New Case", key="td_new_case_btn"):
            ss.td2 = {"order": _td_new_order(round_len), "i": 0, "score": 0, "checked": False,
                      "last_choice": None, "ver": current_ver}
            st.rerun()

    if not ss.td2["order"]:
        st.info("Click **New Case** to start investigating.")
        return
    if any((x is None) or (x >= len(THOUGHT_DETECTIVE_CARDS)) for x in ss.td2["order"]):
        ss.td2["order"] = _td_new_order(round_len)
        ss.td2["i"] = 0

    if ss.td2["i"] >= len(ss.td2["order"]):
        st.success(f"Case closed! Score: **{ss.td2['score']}/{len(ss.td2['order'])}** 🥳")
        return

    idx = ss.td2["order"][ss.td2["i"]]
    card = THOUGHT_DETECTIVE_CARDS[idx]
    st.write(f"🧩 **Case {ss.td2['i']+1} of {len(ss.td2['order'])}:**")
    st.info(card["scenario"])

    if ss.td2["last_choice"] in card["choices"]:
        default_idx = card["choices"].index(ss.td2["last_choice"])
    else:
        default_idx = 0

    choice = st.radio(
        "Most likely mood/trigger:",
        card["choices"],
        index=default_idx,
        key=f"tdq_{ss.td2['i']}"
    )
    ss.td2["last_choice"] = choice

    bb = st.columns([1,1])
    if bb[0].button("Check", key=f"td_check_{ss.td2['i']}"):
        ss.td2["checked"] = True
        if choice == card["answer"]:
            ss.td2["score"] += 1
            st.success(f"Correct! 🔍 {_affirmation(True)}")
        else:
            st.error(f"Nope — the correct answer is **{card['answer']}**.")
        st.info(f"Why: {card['why']}")
        if "tactic" in card and card["tactic"]:
            st.caption(f"Try this tactic: {card['tactic']}")

    if bb[1].button("Next Question ➜", key=f"td_next_{ss.td2['i']}") and ss.td2["checked"]:
        ss.td2["i"] += 1
        ss.td2["checked"] = False
        ss.td2["last_choice"] = None
        st.rerun()

# ====================== MANDALA COLOR FLOW (grid) ======================
def game_mandala_colorflow():
    ss = st.session_state
    st.subheader("Mandala Color Flow 🎨 (Grid Mode)")

    if "mf2" not in ss:
        ss.mf2 = {
            "size": 4,
            "palette": ["🟦","🟩","🟨","🟧","🟥","🟪"],
            "sel": "🟦",
            "grid": [[None for _ in range(4)] for _ in range(4)],
        }

    st.caption("Pick a color and tap the tiles to paint.")
    pc = st.columns(len(ss.mf2["palette"]))
    for i, c in enumerate(ss.mf2["palette"]):
        if pc[i].button(c, key=f"mf2_palette_grid_{i}"):
            ss.mf2["sel"] = c
    st.write("Selected:", ss.mf2["sel"])

    sizes = [3,4,5,6,7]
    size = st.selectbox(
        "Grid size",
        sizes,
        index=sizes.index(ss.mf2.get("size", 4)),
        key="mf2_grid_only_size"
    )

    if ss.mf2.get("grid") is None or ss.mf2["size"] != size:
        ss.mf2["size"] = size
        ss.mf2["grid"] = [[None for _ in range(size)] for _ in range(size)]

    n = ss.mf2["size"]
    gcols = st.columns(n)
    for r in range(n):
        for c in range(n):
            with gcols[c]:
                label = ss.mf2["grid"][r][c] or "◻️"
                if st.button(label, key=f"mf2_grid_only_btn_{n}_{r}_{c}", use_container_width=True):
                    if ss.mf2["grid"][r][c] is None:
                        ss.mf2["grid"][r][c] = ss.mf2["sel"]
                        st.rerun()

    filled = sum(1 for r in range(n) for c in range(n) if ss.mf2["grid"][r][c] is not None)
    st.caption(f"Segments filled: {filled}/{n*n}")

    cclear, _ = st.columns([1,6])
    if cclear.button("Clear", key=f"mf2_grid_only_clear_{n}"):
        ss.mf2["grid"] = [[None for _ in range(n)] for _ in range(n)]
        st.rerun()

# ====================== PAGE: TABS ======================
def page_games():
    st.markdown("## Games 🎮")
    tabs = st.tabs([
        "Puzzles",
        "Memory Flip",        # NEW
        "Catching Leaves",    # NEW
        "Mindful Maze",
        "Thought Detective",
        "Mandala Color Flow",
    ])

    with tabs[0]:
        game_puzzles()
    with tabs[1]:
        game_memory_flip()
    with tabs[2]:
        game_catching_leaves()
    with tabs[3]:
        game_mindful_maze()
    with tabs[4]:
        game_thought_detective()
    with tabs[5]:
        game_mandala_colorflow()



# ---------------------- DIARY (WHO-5 + Gratitude + Growing Tree) ----------------------
from datetime import date
import time
import os
import re
import streamlit as st
import pandas as pd
from data.db_setup import get_connection, init_db

# Ensure database exists
# Ensure database exists
if "db_initialized" not in st.session_state:
    from data.db_setup import init_db
    init_db()
    st.session_state["db_initialized"] = True



# ========= WHO-5 emoji selector =========
def emoji_scale(label, key):
    options = ["😞", "😕", "😐", "🙂", "😄"]
    cols = st.columns(len(options))
    if key not in st.session_state:
        st.session_state[key] = 3
    for i, emo in enumerate(options, start=1):
        with cols[i - 1]:
            if st.button(emo, key=f"{key}_{i}", use_container_width=True):
                st.session_state[key] = i
    st.caption(label)
    return st.session_state[key]


# ========= Detect crisis language =========
_CRISIS_PATTERNS = [
    r"\bi\s*(want|wish|wanna)\s*(to\s*)?(die|disappear|end\s*(it|my\s*life))\b",
    r"\bkill\s*myself\b",
    r"\bsuicide\b",
    r"\bself[-\s]*harm\b",
    r"\bno\s*reason\s*to\s*live\b",
]
def reflections_have_crisis(reflections):
    text = " ".join([(r or "") for r in reflections]).lower()
    return any(re.search(p, text) for p in _CRISIS_PATTERNS)


# ========= Map WHO-5 score to 7-stage tree =========
def stage_from_score(score: int) -> int:
    s = max(0, min(25, int(score)))
    if s <= 3:   return 0
    if s <= 6:   return 1
    if s <= 9:   return 2
    if s <= 13:  return 3
    if s <= 17:  return 4
    if s <= 21:  return 5
    return 6


# ========= Classify day (more expressive) =========
def classify_day(score: int) -> str:
    s = max(0, min(25, int(score)))
    if s <= 4:
        return "🆘 Very low mood — It’s okay to ask for help today."
    elif s <= 8:
        return "😣 Tough day — Take some rest and self-kindness."
    elif s <= 12:
        return "😕 A low/flat day — Small steps forward still count."
    elif s <= 16:
        return "🙂 A steady day — You’re doing fine."
    elif s <= 20:
        return "😊 A good day — Keep that positive momentum."
    elif s <= 24:
        return "🌿 A bright day — You’re growing well."
    else:
        return "🌳 A great day — You’re thriving today!"


# ========= Tree image handling =========
def get_tree_image_path(stage: int) -> str:
    """Return correct image path for given stage (0–6)."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "assets", f"tree_stage_{stage}.png")


def render_tree(stage: int, placeholder):
    """Display a centered image of the given tree growth stage."""
    img_path = get_tree_image_path(stage)
    if not os.path.exists(img_path):
        placeholder.error(f"⚠ Missing image for stage {stage}: {img_path}")
        return

    with placeholder.container():
        # Center the image cleanly
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img_path, width=380)



def animate_stages(prev_stage: int, target_stage: int, placeholder, delay: float = 0.35):
    """Smooth transition between tree stages."""
    if prev_stage is None:
        prev_stage = 0
    step = 1 if target_stage >= prev_stage else -1
    for s in range(prev_stage, target_stage + step, step):
        render_tree(s, placeholder)
        time.sleep(delay)


# ========= Main Diary Page =========
from badges_logs import log_activity  # keep this import at top with your others

def page_diary():
    conn = get_connection()
    conn.execute("PRAGMA busy_timeout=30000;")
    today = date.today().isoformat()
    if "username" not in st.session_state:
        st.session_state["username"] = "guest"
    user = st.session_state["username"]

    # Reset daily session
    if st.session_state.get("last_date") != today:
        st.session_state["saved_who_score"] = None
        st.session_state["last_stage"] = None
        st.session_state["animated_today"] = False
        st.session_state["last_date"] = today

    st.title("🪴 Daily Reflection")
    st.caption("Your mood shapes your tree — nurture it every day 🌿")

    tab_who, tab_reflect = st.tabs(["WHO-5", "Reflection Journal"])

    # ======================= TAB 1: WHO-5 =======================
    with tab_who:
        st.subheader("WHO-5 Well-being Check-in")
        qs = [
            "I felt cheerful and optimistic 🌞",
            "I felt calm and peaceful 🌿",
            "I felt energetic and active ⚡",
            "I woke up feeling rested ☀",
            "My day was interesting and meaningful 🌸",
        ]
        who_scores = [emoji_scale(q, f"who_{i}") for i, q in enumerate(qs)]
        who_total = int(sum(who_scores))  # 5–25

        # Save WHO-5
        saving = st.session_state.get("who_saving", False)
        save_btn = st.button("💾 Save WHO-5 Response", disabled=saving)

        if save_btn and not saving:
            st.session_state["who_saving"] = True
            try:
                import sqlite3, time
                for attempt in range(3):
                    try:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO user_mood
                                (username, date, who5_score, reflection1, reflection2, reflection3,
                                 inferred_mood, growth_score)
                            VALUES (
                                ?, ?, ?,
                                COALESCE((SELECT reflection1 FROM user_mood WHERE username=? AND date=?), ''),
                                COALESCE((SELECT reflection2 FROM user_mood WHERE username=? AND date=?), ''),
                                COALESCE((SELECT reflection3 FROM user_mood WHERE username=? AND date=?), ''),
                                COALESCE((SELECT inferred_mood FROM user_mood WHERE username=? AND date=?), ''),
                                ?
                            )
                            """,
                            (
                                user, today, who_total,
                                user, today, user, today, user, today, user, today,
                                who_total / 25.0,
                            ),
                        )
                        conn.commit()
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e).lower() and attempt < 2:
                            time.sleep(1.5) 
                            continue
                        else:
                            raise

                st.session_state["saved_who_score"] = who_total
                st.session_state["animated_today"] = False

                # Log WHO-5 to Activities
                log_activity(
                    st.session_state.get("user_id", "demo-user"),
                    "mood_entry",
                    {"who5": who_total, "mood": classify_day(who_total)}
                )

                st.success(f"Saved! Score: {who_total}/25 — {classify_day(who_total)}")
            finally:
                st.session_state["who_saving"] = False

        # Tree visualization
        st.subheader("🌳 Your Growth Tree")
        ph = st.empty()
        saved = st.session_state.get("saved_who_score")

        if saved is None:
            render_tree(0, ph)
            st.info("➡ Save WHO-5 to make your tree start growing.")
        else:
            target_stage = stage_from_score(saved)
            if not st.session_state.get("animated_today", False):
                prev_stage = st.session_state.get("last_stage", 0)
                animate_stages(prev_stage, target_stage, ph)
                st.session_state["last_stage"] = target_stage
                st.session_state["animated_today"] = True
            else:
                render_tree(target_stage, ph)

        # 30-day chart
        st.subheader("📊 WHO-5 Over the Last 30 Days")
        df = pd.read_sql_query(
            "SELECT date, who5_score FROM user_mood WHERE username=? ORDER BY date DESC LIMIT 30",
            conn, params=(user,),
        )
        if not df.empty:
            st.line_chart(df.sort_values("date").set_index("date"))
        else:
            st.caption("No entries yet — your chart will appear after saving.")

    # =================== TAB 2: Reflection Journal ===================
    with tab_reflect:
        st.subheader("🌼 Reflection Journal")
        r1 = st.text_area("1️⃣ What made you smile today?")
        r2 = st.text_area("2️⃣ What challenged you today?")
        r3 = st.text_area("3️⃣ What are you looking forward to tomorrow?")

        if st.button("📝 Save My Reflections"):
            missing = [i for i, r in enumerate([r1, r2, r3], start=1) if not (r or "").strip()]
            if missing:
                st.warning("You left " + ", ".join(f"Q{m}" for m in missing) + " empty — please fill all.")
            else:
                crisis = reflections_have_crisis([r1, r2, r3])
                conn.execute(
                    """
                    INSERT OR REPLACE INTO user_mood
                        (username, date, who5_score, reflection1, reflection2, reflection3,
                         inferred_mood, growth_score)
                    VALUES (
                        ?, ?, COALESCE((SELECT who5_score FROM user_mood WHERE username=? AND date=?), 0),
                        ?, ?, ?,
                        COALESCE((SELECT inferred_mood FROM user_mood WHERE username=? AND date=?), ''),
                        COALESCE((SELECT growth_score FROM user_mood WHERE username=? AND date=?), 0.0)
                    )
                    """,
                    (user, today, user, today, r1, r2, r3, user, today, user, today),
                )
                conn.commit()

                # Log all three answers for Activities (multiline rendering)
                log_activity(
                    st.session_state.get("user_id", "demo-user"),
                    "journal_entry",
                    {"notes": [r1, r2, r3]}
                )

                if crisis:
                    st.error(
                        "🚨 It sounds like you might be going through something hard. "
                        "Please reach out to someone you trust or local help lines. 💛"
                    )
                else:
                    st.success("🌟 Reflections saved successfully.")
                    st.caption("💚 Thank you for sharing — your reflections water your growth.")

    


# --- Sync today's diary/WHO-5 into Badges & Logs activity (no changes to page_diary) ---
def _sync_diary_activity_for_today():
    import sqlite3, datetime
    from badges_logs import log_activity, DB_PATH  # uses the same DB and logger

    # Make sure the same ID is used across the app:
    # If you already set ss.user_id elsewhere, keep it. Otherwise mirror username.
    ss = st.session_state
    username = ss.get("username", "guest")
    if "user_id" not in ss:
        ss["user_id"] = username  # align identity without touching page_diary()

    user_id = ss["user_id"]
    today = datetime.date.today().isoformat()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Pull today's saved reflections + WHO-5 (page_diary already writes user_mood)
    row = cur.execute(
        """
        SELECT reflection1, reflection2, reflection3, who5_score
        FROM user_mood
        WHERE username=? AND date=?
        """,
        (username, today)
    ).fetchone()

    if row:
        # 1) Log a journal entry (once/day) if any reflection text exists
        note = " ".join([
            (row["reflection1"] or "").strip(),
            (row["reflection2"] or "").strip(),
            (row["reflection3"] or "").strip(),
        ]).strip()

        already_journal = cur.execute(
            """
            SELECT 1 FROM user_activity_log
            WHERE user_id=? AND kind='journal_entry' AND date=?
            LIMIT 1
            """,
            (user_id, today)
        ).fetchone()

        if note and not already_journal:
            log_activity(user_id, "journal_entry", {"note": note})

        # 2) Log a mood/WHO-5 entry (once/day) if a score exists
        score = int(row["who5_score"] or 0)
        already_mood = cur.execute(
            """
            SELECT 1 FROM user_activity_log
            WHERE user_id=? AND kind='mood_entry' AND date=?
            LIMIT 1
            """,
            (user_id, today)
        ).fetchone()

        if score > 0 and not already_mood:
            # (Optional label from score—kept simple here to avoid importing page_diary logic)
            log_activity(user_id, "mood_entry", {"who5": score})

    con.close()





# ---------------------- Helpline ----------------------
import re
import streamlit as st

def _tel(num: str) -> str:
    num = str(num or "").strip()
    clean = re.sub(r"[^\d+]", "", num)
    return f"[{num}](tel:{clean})" if num else ""

# Static, verified national helplines
NATIONAL_SECTIONS = {
    "🚨 Emergency": {
        "tip": "Use these for *immediate danger* or *urgent medical help*.",
        "items": [
            {"label": "Single Emergency (Police/Fire/Health)", "number": "112", "note": "Pan-India ERSS, 24×7"},
            {"label": "Ambulance (Emergency)", "number": "108"},
            {"label": "Ambulance (Patient Transport)", "number": "102"},
        ],
    },
    "🧠 Mental Wellness (National)": {
        "tip": "Use for *counselling, feeling overwhelmed, anxiety, low mood*.",
        "items": [
            {"label": "Tele-MANAS — 24×7 Counselling", "number": "14416", "alt": "1800-891-4416", "note": "Govt. of India; multilingual"},
            {"label": "NIMHANS Mental Health Helpline", "number": "080-4611-0007"},
            {"label": "CHILDLINE (for children)", "number": "1098"},
        ],
    },
    "🛡 Protection & Legal": {
        "tip": "Use for *violence/abuse, **elder support, **online fraud, or **legal aid*.",
        "items": [
            {"label": "Women Helpline", "number": "181"},
            {"label": "Elderline (Senior Citizens)", "number": "14567"},
            {"label": "Cybercrime Helpline", "number": "1930", "note": "Financial/online fraud"},
            {"label": "NALSA — National Legal Aid", "number": "15100"},
        ],
    },
}

def page_helpline():
    st.markdown("## 📞 Helplines — MannMitra")
    st.info("If you’re in *immediate danger, call **112* now.")

    # Simple category switcher (no state/search)
    category = st.radio(
        "Choose a section",
        list(NATIONAL_SECTIONS.keys()),
        index=0,
        horizontal=True,
        key="helpline_category",
    )

    sec = NATIONAL_SECTIONS[category]
    st.caption(sec["tip"])

    st.subheader(category)
    for it in sec["items"]:
        parts = [f"{it['label']}"]
        if it.get("number"):
            parts.append(_tel(it["number"]))
        if it.get("alt"):
            parts.append(f"(Alt: {_tel(it['alt'])})")
        if it.get("note"):
            parts.append(f"— {it['note']}")
        st.write(" — ".join(parts))

    st.markdown("---")
    st.caption("MannMitra is not a substitute for professional care. In emergencies, dial *112*.")




# =================== CommuniGrow ===================
# Imports
import os, sqlite3, datetime, json, uuid, random
import streamlit as st

# Globals / paths
ss = st.session_state
DB_PATH = os.path.join("data", "mannmitra.db")
USER_STORE = os.path.join("data", "users.json")

# ---------- basics ----------
def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def _today():
    return datetime.date.today()

def _iso_year_week():
    y, w, _ = _today().isocalendar()
    return y, w

def get_user_id():
    """Persist a local UUID so each device/user has a stable id."""
    if ss.get("user_id"):
        return ss.user_id
    os.makedirs("data", exist_ok=True)
    uid = None
    try:
        if os.path.exists(USER_STORE):
            with open(USER_STORE, "r", encoding="utf-8") as f:
                data = json.load(f)
                uid = data.get("default_user_id")
    except Exception:
        uid = None
    if not uid:
        uid = "user-" + uuid.uuid4().hex[:12]
        try:
            with open(USER_STORE, "w", encoding="utf-8") as f:
                json.dump({"default_user_id": uid}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    ss.user_id = uid
    return uid

# ---------- schema + seed ----------
def init_communigrow():
    """Safe to call repeatedly. Ensures schema + seeds content."""
    os.makedirs("data", exist_ok=True)
    con = _db(); cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS daily_quotes (
      id INTEGER PRIMARY KEY,
      quote TEXT NOT NULL,
      author TEXT
    );
    CREATE TABLE IF NOT EXISTS weekly_goals (
      id INTEGER PRIMARY KEY,
      title TEXT NOT NULL,
      description TEXT,
      xp INTEGER DEFAULT 10
    );
    CREATE TABLE IF NOT EXISTS mini_lessons (
      id INTEGER PRIMARY KEY,
      title TEXT NOT NULL,
      content TEXT NOT NULL,
      tags TEXT
    );
    CREATE TABLE IF NOT EXISTS user_sprout_log (
      id INTEGER PRIMARY KEY,
      user_id TEXT NOT NULL,
      quote_id INTEGER NOT NULL,
      shown_on DATE NOT NULL
    );
    -- v2: multiple goals per week via unique(user,year,week,goal)
    CREATE TABLE IF NOT EXISTS user_week_progress_v2 (
      id INTEGER PRIMARY KEY,
      user_id TEXT NOT NULL,
      iso_year INTEGER NOT NULL,
      iso_week INTEGER NOT NULL,
      goal_id INTEGER NOT NULL,
      completed INTEGER DEFAULT 0,
      UNIQUE(user_id, iso_year, iso_week, goal_id)
    );
    -- affirmations master
    CREATE TABLE IF NOT EXISTS affirmations (
      id INTEGER PRIMARY KEY,
      text TEXT NOT NULL
    );
    -- DAILY affirmations log (no repeat in 30 days)
    CREATE TABLE IF NOT EXISTS user_daily_affirmations (
      id INTEGER PRIMARY KEY,
      user_id TEXT NOT NULL,
      date TEXT NOT NULL,                -- YYYY-MM-DD
      affirmation_id INTEGER NOT NULL,
      UNIQUE(user_id, date)
    );
    """)
    # ---- seed quotes ----
    if cur.execute("SELECT COUNT(*) FROM daily_quotes").fetchone()[0] < 25:
        existing = set(q["quote"] for q in cur.execute("SELECT quote FROM daily_quotes").fetchall())
        add_quotes = [
            ("Connection is courage in disguise.", "Unknown"),
            ("Listening is an act of love.", "Unknown"),
            ("Small steps, big change.", "Unknown"),
            ("Be the friend you wish you had.", "Unknown"),
            ("You don’t need perfect words. You need a kind start.", "Mitra"),
            ("Confidence grows from tiny reps.", "Mitra"),
            ("Your voice matters—even when it shakes.", "Mitra"),
            ("Every hello plants a seed.", "Mitra"),
            ("Brave is just shy plus action.", "Mitra"),
            ("Silence can be gentle. Fill it with care, not panic.", "Mitra"),
            ("You can restart any moment with kindness.", "Mitra"),
            ("Curiosity is the friend-maker.", "Mitra"),
            ("Soft is strong. Gentle is brave.", "Mitra"),
            ("Practice makes natural.", "Mitra"),
            ("One message can reopen a door.", "Mitra"),
            ("The goal isn’t perfect chat. It’s warm presence.", "Mitra"),
            ("Ask once, don’t chase. Self-respect is magnetic.", "Mitra"),
            ("You’re allowed to try again tomorrow.", "Mitra"),
            ("Vulnerability invites real connection.", "Mitra"),
            ("It’s okay to go slow. Speed isn’t kindness.", "Mitra"),
            ("A smile is a bridge you can build in one second.", "Mitra"),
            ("Your people are out there—go find one small moment.", "Mitra"),
            ("Awkward is normal. Keep going.", "Mitra"),
            ("Kindness is never cringe.", "Mitra"),
            ("Effort is attractive. Be the one who tries.", "Mitra"),
            ("Boundaries protect your warmth.", "Mitra"),
            ("You can be shy and social.", "Mitra"),
        ]
        to_insert = [(q,a) for (q,a) in add_quotes if q not in existing]
        if to_insert:
            cur.executemany("INSERT INTO daily_quotes(quote, author) VALUES(?,?)", to_insert)

    # ---- seed weekly goals ----
    if cur.execute("SELECT COUNT(*) FROM weekly_goals").fetchone()[0] < 20:
        existing = set(g["title"] for g in cur.execute("SELECT title FROM weekly_goals").fetchall())
        add_goals = [
            ("Compliment 2 people genuinely","Notice something real and kind about two people and tell them.", 15),
            ("Phone-free 5-minute chat","Talk to someone for five minutes without checking your phone.", 20),
            ("Reach out to someone you miss","Send a warm message to someone you haven’t talked to in a while.", 15),
            ("Ask someone a curious question","Try: ‘What’s something fun you’re looking forward to?’", 15),
            ("Join a conversation for 1 minute","Stand nearby, smile, add one sentence. That’s it.", 15),
            ("Invite someone to sit with you","Offer a warm invite to lunch/break/short walk.", 20),
            ("Share a small win out loud","Tell a friend one tiny thing you did well this week.", 10),
            ("Give a gratitude text","Thank someone for something specific they did.", 15),
            ("Start a group chat for a plan","Propose a mini plan: ‘Tea break Wed?’", 20),
            ("Ask for advice on something tiny","People like to help. Keep it small and easy.", 15),
            ("Encourage someone publicly","Back someone up in class or a meeting.", 20),
            ("Hold eye contact for 2 seconds & smile","Practice warmth in passing moments.", 10),
            ("Learn 2 new names","Remember them and use them once.", 15),
            ("Give someone the floor","Ask a follow-up and let them talk for a minute.", 15),
            ("Say hi to staff (security/cafeteria/etc.)","Build everyday kindness with people who help you daily.", 10),
            ("Offer help once","‘I’m heading there anyway, want me to bring one?’", 15),
            ("Start a positive rumor","Spread praise about someone’s good work.", 10),
            ("Phone stack challenge with friends","Put phones down for 10 minutes and chat.", 20),
            ("Plan a tiny event","2–3 people, 15 minutes. Keep it low-pressure.", 25),
            ("Rejection rep","Ask a harmless question you might hear ‘no’ to. Learn it’s okay.", 20),
        ]
        to_insert = [g for g in add_goals if g[0] not in existing]
        if to_insert:
            cur.executemany("INSERT INTO weekly_goals(title, description, xp) VALUES(?,?,?)", to_insert)

    # ---- seed Mitra lessons ----
    if cur.execute("SELECT COUNT(*) FROM mini_lessons").fetchone()[0] < 20:
        existing = set(m["title"] for m in cur.execute("SELECT title FROM mini_lessons").fetchall())
        lessons = [
            ("Start a conversation when you’re nervous",
             "Why this works:\n- Tiny starts lower pressure.\n- Curiosity beats performance.\n\nTry this today:\n1) Warm-up line: “Hey! How’s your day going?”\n2) Follow-up: “What was the best part?” or “How did that go?”\n3) Exit kindly: “Good to see you—catch you later!”\n\nPro tips:\n- Smile + name + short question = natural.\n- If your mind goes blank, reflect: “That sounds fun.”",
             "conversation,confidence,anxiety"),
            ("Handle awkward silences (without panicking)",
             "Reframe: silence = thinking time, not failure.\n3 tools:\n1) Reflect: “Makes sense.” / “True.”\n2) Pivot small: “Anyway, did you check the new canteen menu?”\n3) Share a mini-story (<10 sec) to restart flow.\nPractice goal: allow 2 seconds of silence before reacting.",
             "awkwardness,conversation,confidence"),
            ("Text someone first (without cringe)",
             "Simple openers:\n- ‘Saw this and thought of you → [pic/link]’\n- ‘Hey! How’s your week going?’\n- ‘I liked your [thing].’\nKeep it light, specific, and end with an easy question.\nBoundary tip: if they reply slow, don’t triple-text. Self-respect keeps convo comfy.",
             "texting,overthinking,confidence"),
            ("Join a new group when you feel shy",
             "Micro-join method:\n1) Stand near edge, smile.\n2) After a beat, add one sentence agreeing or asking.\n3) Stay for 60 seconds; leave kindly if you want.\nThis counts as a rep—confidence grows from small reps.",
             "groups,shyness,exposure"),
            ("Speak up in class or a meeting (starter lines)",
             "Use a pre-built opener:\n- “I’m curious about…”\n- “To build on that…”\n- “Can I check I got this right?”\nKeep it under 20 seconds. One clear point > long speech.\nTip: jot your opener on your phone before the session.",
             "speaking,school,confidence"),
            ("Handle embarrassment in the moment",
             "Normalize + move: “Oops, that was awkward.” *smile*\nThen pivot: “Anyway…”\nYour brain learns: embarrassment isn’t dangerous. That’s social resilience training.",
             "embarrassment,resilience"),
            ("Fear of rejection—do 3 tiny reps",
             "Rejection = data, not doom.\nThis week’s reps:\n1) Say hi to a stranger.\n2) Ask a tiny favor (hold door, quick opinion).\n3) Invite someone to a 2-minute chat.\nEach ‘no’ proves you can survive and move on.",
             "rejection,anxiety,confidence"),
            ("Be interesting without trying hard",
             "Be specific. Swap ‘I like music’ for ‘I’m into old-school Hindi lofi right now.’\nAsk back: “What’s your current song?”\nSpecific → memorable → closer.",
             "conversation,personality"),
            ("Make eye contact feel natural",
             "Aim for 2–3 seconds, then glance away briefly.\nThink ‘I’m happy to see you’ instead of ‘Don’t mess up’. Your face will soften naturally.",
             "confidence,nonverbal"),
            ("Restart with someone after distance",
             "Use the bridge: “I’ve been meaning to say hi—how have you been?”\nOffer context if needed, don’t over-apologize, and propose a tiny plan.",
             "friendship,reconnect"),
        ]
        to_insert = [l for l in lessons if l[0] not in existing]
        if to_insert:
            cur.executemany("INSERT INTO mini_lessons(title, content, tags) VALUES(?,?,?)", to_insert)

    # ---- seed affirmations ----
    if cur.execute("SELECT COUNT(*) FROM affirmations").fetchone()[0] < 60:
        existing = set(a["text"] for a in cur.execute("SELECT text FROM affirmations").fetchall())
        add_affirmations = [
            "I am worthy of calm connections and kind people.",
            "My effort counts more than perfection.",
            "I can be shy and still be brave.",
            "Every small step I take builds confidence.",
            "I bring warmth into rooms just by being me.",
            "I am allowed to learn at my own pace.",
            "I don’t need to be loud to be heard.",
            "My boundaries protect my kindness.",
            "Today, I choose curiosity over fear.",
            "Awkward moments don’t define me.",
            "I am becoming the friend I want to have.",
            "It’s safe to be seen as I am.",
            "I can try again with a softer heart.",
            "I honor my energy and say no when needed.",
            "My voice matters, even in short sentences.",
            "I’m building a life full of gentle courage.",
            "I celebrate tiny wins—they grow big roots.",
            "I can start small and still go far.",
            "I am lovable as I learn.",
            "I give and receive kindness with ease.",
            "I can handle a ‘no’ and keep my glow.",
            "I am growing stronger and softer at once.",
            "I can make space for others and myself.",
            "I show up imperfectly and that’s enough.",
            "Connection gets easier with practice.",
            "I deserve friendships that feel safe.",
            "I trust myself to take the next step.",
            "My presence is a gift, not a problem.",
            "I can create the moments I wish I had.",
            "I am allowed to take up space kindly.",
            "I am patient with my progress.",
            "My heart is learning new ways to be brave.",
            "I can meet today with gentle confidence.",
            "I am the author of my social story.",
            "I choose kindness, especially for myself.",
            "I can be nervous and still say hello.",
            "I let go of perfect; I choose real.",
            "I grow through what I practice.",
            "I am safe to try, learn, and try again.",
            "I am blooming in my own season.",
            "People respond to my genuine warmth.",
            "I find friends by being a friend.",
            "I can rest without losing progress.",
            "I lead with respect for myself and others.",
            "I am enough—quiet or loud.",
            "I can repair after a rough moment.",
            "I notice good things and say them out loud.",
            "My curiosity opens doors.",
            "I am learning to enjoy social moments.",
            "I’m courageous for trying again.",
            "I belong in the rooms I enter.",
            "I can be soft and powerful together.",
            "My kindness is never wasted.",
            "I am allowed to learn in public.",
            "I start small and stay consistent.",
            "I accept myself as I grow.",
            "I can turn awkward into honest.",
            "I am building a kinder inner voice.",
            "I choose progress over perfection today.",
        ]
        to_insert = [ (t,) for t in add_affirmations if t not in existing ]
        if to_insert:
            cur.executemany("INSERT INTO affirmations(text) VALUES(?)", to_insert)

    con.commit(); con.close()

from streamlit.components.v1 import html as st_html
import base64, random, uuid

# === UI helpers ===
from textwrap import dedent
from streamlit.components.v1 import html as st_html
import base64, random, uuid
import streamlit as st

def _inject_communi_styles():
    """Global CSS for the Sprout card. Safe to call once."""
    st.markdown(dedent("""
    <style>
      .mm-card{background:linear-gradient(180deg,#111418 0%,#0c0f13 100%);
               border:1px solid #2b2f36;border-radius:18px;padding:18px 20px;margin:12px 0;}
      .mm-card h3{margin:0 0 8px 0;font-size:1.15rem}
      .mm-quote{font-size:1.05rem;line-height:1.6}
      .mm-author{opacity:.75;font-size:.9rem;margin-bottom:10px}
      .mm-tip{background:#0f1420;border:1px dashed #3a4251;border-radius:12px;padding:10px 12px;margin-top:10px}
    </style>
    """), unsafe_allow_html=True)

def render_sprout_card(sprout: dict):
    quote = sprout.get("quote","")
    author = sprout.get("author") or "Unknown"
    tip = sprout.get("tip","")
    st.markdown(dedent(f"""
<div class="mm-card">
  <h3>🌱 Today’s Sprout</h3>
  <div class="mm-quote">“{quote}”</div>
  <div class="mm-author">— {author}</div>
  <div class="mm-tip"><strong>Try this:</strong> {tip}</div>
</div>
"""), unsafe_allow_html=True)

from streamlit.components.v1 import html as st_html
import base64, random, uuid

def render_affirmation_section_component(
    affirm: dict,
    gift_image_path: str | None = "/mnt/data/fdf7e42e-0424-42e6-bb7f-7c0df1ecc3df.png",
    max_width: int = 720,        # width of the dark card container
    card_width: int = 420,       # width of the beige flip card inside
    height: int = 420,           # total component height (adjust if needed)
):
    """One boxed section: heading + subtitle + flip gift card INSIDE the same container."""
    date = affirm.get("date", "")
    text = affirm.get("text", "")
    cid  = f"gift_{uuid.uuid4().hex[:8]}"

    # inline the front image (or fall back to CSS ribbon)
    front_bg = ""
    include_fallback_ribbon = True
    if gift_image_path:
        try:
            with open(gift_image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            front_bg = f'background-image:url("data:image/png;base64,{b64}");background-size:cover;background-position:center;'
            include_fallback_ribbon = False
        except Exception:
            include_fallback_ribbon = True

    conf = "".join(
        f'<span style="left:{random.randint(3,97)}%;background:hsl({random.randint(0,360)},80%,70%);'
        f'animation-delay:{random.uniform(0,.6):.2f}s;"></span>'
        for _ in range(22)
    )

    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8" />
<style>
  :root {{ --bg1:#111418; --bg2:#0c0f13; --stroke:#2b2f36; --txt:#eaeef6;
           --mut:#a7b0c0; --cream:#fffdf9; --cream2:#f8f3ee; --gold:#c7a874; --shadow:rgba(0,0,0,.22); }}
  html,body {{ margin:0; background:transparent; font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif; color:var(--txt); }}

  /* outer dark card (matches your Sprout box) */
  .mm-card {{
    width:100%;
    max-width:{max_width}px;
    margin:0 auto;
    background:linear-gradient(180deg,var(--bg1),var(--bg2));
    border:1px solid var(--stroke);
    border-radius:18px;
    padding:18px 20px 22px 20px;
    box-shadow:0 10px 26px rgba(0,0,0,.35);
  }}
  .mm-title {{ margin:0 0 6px 0; font-size:1.15rem; font-weight:700; }}
  .mm-sub   {{ margin:0 0 12px 0; font-size:.95rem; color:var(--mut); }}

  /* flip card area (centered) */
  .wrap {{ display:flex; justify-content:center; }}
  .card3d {{ position:relative; width: min({card_width}px, 92vw); height: 210px; perspective: 1200px; }}
  .face {{ position:absolute; inset:0; border-radius:16px; backface-visibility:hidden; overflow:hidden; }}
  .front {{ background:#fff; border:1px solid #eee; box-shadow:0 10px 24px var(--shadow); {front_bg} }}
  .back  {{
    transform: rotateY(180deg);
    background:linear-gradient(145deg,var(--cream),var(--cream2));
    color:#333; border:1px solid rgba(0,0,0,.06);
    display:flex; align-items:center; justify-content:center;
  }}

  .flipbox {{ position:relative; width:100%; height:100%; transform-style:preserve-3d;
             transition: transform .8s cubic-bezier(.2,.7,.2,1); }}
  input[type=checkbox] {{ display:none; }}
  input:checked ~ .flipbox {{ transform: rotateY(180deg); }}
  .hit {{ position:absolute; inset:0; cursor:pointer; z-index:3; }}
  .tap {{ position:absolute; bottom:10px; width:100%; text-align:center; font-size:.85rem; color:#333; font-weight:600; }}

  /* fallback ribbon if no image is provided */
  .ribbon {{ position:absolute; top:50%; left:0; right:0; height:56px; transform:translateY(-50%);
            background:linear-gradient(90deg,#d6b98b,#e6caa0,#d6b98b); }}
  .bow {{ position:absolute; top:50%; left:18%; transform:translate(-50%,-50%); font-size:56px; }}

  /* back content */
  .glass {{
    position:relative; max-width:86%; padding:18px 20px; text-align:center; border-radius:14px;
    background:rgba(255,255,255,.55); backdrop-filter: blur(8px);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.35), 0 8px 20px var(--shadow);
  }}
  .a-text {{ font-size:1.08rem; line-height:1.55; color:#2f2a24; }}
  .a-date {{ margin-top:8px; font-size:.85rem; color:var(--gold); display:block; }}

  /* confetti pops only when opened */
  .confetti {{ position:absolute; inset:0; pointer-events:none; overflow:hidden; }}
  .confetti span {{ position:absolute; top:110%; width:6px; height:6px; border-radius:50%; opacity:.9;
                    animation: pop 1.3s ease-out forwards; }}
  input:not(:checked) ~ .flipbox .confetti span {{ animation-play-state:paused; }}
  @keyframes pop {{ 0% {{ transform:translateY(0) rotate(0) }} 100% {{ transform:translateY(-150%) rotate(300deg); opacity:0 }} }}
</style>
</head>
<body>
  <section class="mm-card">
    <h3 class="mm-title">🎁 Affirmation</h3>
    <p class="mm-sub">A gentle reminder to stay kind and grounded today.</p>

    <div class="wrap">
      <div class="card3d">
        <input id="{cid}" type="checkbox" />
        <label class="hit" for="{cid}" title="Tap to open/close"></label>

        <div class="flipbox">
          <div class="face front">
            {('<div class="ribbon"></div><div class="bow">🎀</div>') if include_fallback_ribbon else ''}
            <div class="tap">Tap to open</div>
          </div>

          <div class="face back">
            <div class="glass">
              <div class="a-text">🌟 {text}</div>
              <span class="a-date">{date}</span>
            </div>
            <div class="confetti">{conf}</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</body></html>
"""
    st_html(doc, height=height, width=max_width)



# ---------- Mini Lessons: "Book" UI helpers ----------
import hashlib
from textwrap import wrap

def _inject_book_styles():
    st.markdown("""
    <style>
      /* ---------- Shelf cards ---------- */
      .shelf{
        display:grid;
        grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
        gap:16px;margin-top:10px;
      }
      .book{
        background:linear-gradient(145deg,#1a1f28,#12161c);
        border:1px solid #2b2f36;
        border-radius:16px;
        padding:14px 14px 16px 14px;
        position:relative;
        cursor:pointer;
        transition:.25s transform, .25s box-shadow;
      }
      .book:hover{
        transform:translateY(-4px);
        box-shadow:0 10px 28px rgba(0,0,0,.45);
      }
      .cover{
        height:160px;border-radius:12px;overflow:hidden;
        border:1px solid rgba(255,255,255,.08);
        display:flex;align-items:center;justify-content:center;
        text-align:center;padding:12px;
      }
      .cover h4{margin:0;font-size:1.05rem;line-height:1.35}
      .spine{position:absolute;left:0;top:0;bottom:0;width:12px;
             background:linear-gradient(180deg,rgba(255,255,255,.1),rgba(255,255,255,0));}
      .tagline{margin-top:10px;font-size:.86rem;opacity:.8}

      /* ---------- Open Book View ---------- */
      .reader{
        background:linear-gradient(180deg,#101418,#0c0f13);
        border:1px solid #2b2f36;
        border-radius:18px;
        padding:24px 26px;
        box-shadow:0 10px 30px rgba(0,0,0,.4);
        margin-top:10px;
      }
      .reader h3{
        margin:0 0 6px 0;
        font-size:1.4rem;
        font-weight:700;
        letter-spacing:0.3px;
      }
      .reader .meta{
        opacity:.7;
        font-size:.9rem;
        margin-bottom:16px;
      }

      /* ---------- The inner "page" ---------- */
      .paper{
        background:radial-gradient(circle at 30% 20%, #fff8e7 0%, #fff4dd 40%, #ffe7b9 100%);
        border:1px solid rgba(255,255,255,.15);
        border-radius:16px;
        padding:28px 26px;
        box-shadow:0 12px 28px rgba(255,230,180,.15),
                   inset 0 0 25px rgba(255,255,255,.1);
        color:#2a1e0a;
        font-family:"Georgia",serif;
        line-height:1.65;
        font-size:1.06rem;
        text-shadow:0 0 1px rgba(255,255,255,.3);
      }
      .paper p{margin:0 0 10px 0;}
      .paper ul{margin:8px 0 10px 1.2rem;}
      .paper li{margin:6px 0;}

      /* ---------- Buttons + nav ---------- */
      .toprow{
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom:10px;
      }
      .navrow{
        display:flex;
        justify-content:flex-end;
        align-items:center;
        margin-top:14px;
      }
      .pill{
        padding:6px 10px;
        border:1px solid #3a3f45;
        border-radius:999px;
        font-size:.85rem;
        color:#ccc;
        opacity:.85;
      }

      /* subtle glow hover for button */
      button[kind="secondary"]{
        background:linear-gradient(145deg,#242830,#191d24);
        border:1px solid #353a45;
        color:#eee;
      }
      button[kind="secondary"]:hover{
        background:linear-gradient(145deg,#2c323d,#1e222a);
      }
    </style>
    """, unsafe_allow_html=True)


def _title_color(title:str)->str:
    # deterministic pleasant pastel by hashing title
    h = int(hashlib.sha256(title.encode()).hexdigest(), 16)
    hue = h % 360
    return f"hsl({hue}, 55%, 70%)"

def _split_pages(text:str, max_chars:int=900)->list[str]:
    # Keep bullets / paragraphs together; fall back to length-chunks
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    pages, cur = [], ""
    for p in paragraphs:
        # soft preference for paragraph boundaries
        if len(cur) + len(p) + 2 <= max_chars:
            cur += (("\n\n" if cur else "") + p)
        else:
            if cur: pages.append(cur); cur = ""
            if len(p) <= max_chars:
                pages.append(p)
            else:
                # hard wrap very long paragraph
                chunks = wrap(p, width=max_chars, break_long_words=False, replace_whitespace=False)
                pages.extend(chunks)
    if cur: pages.append(cur)
    return pages or ["(no content)"]

def render_book_shelf(lessons:list[dict]):
    """Grid of books. Clicking sets ss.book_open to the lesson id."""
    _inject_book_styles()
    st.write("")  # small spacer
    with st.container():
        st.markdown('<div class="shelf">', unsafe_allow_html=True)
        for L in lessons:
            color = _title_color(L["title"])
            subtitle = (L["tags"] or "").replace(",", " · ")
            # Use a form per card so the button posts only this "book"
            with st.form(key=f"book_{L['id']}", clear_on_submit=False):
                st.markdown(
                    f"""
                    <div class="book">
                      <div class="cover" style="background:linear-gradient(145deg,{color}, rgba(255,255,255,.15));">
                        <div class="spine"></div>
                        <h4>{L['title']}</h4>
                      </div>
                      <div class="tagline">{subtitle}</div>
                    </div>
                    """, unsafe_allow_html=True
                )
                open_it = st.form_submit_button("Open book", use_container_width=True)
                if open_it:
                    ss.book_open = L["id"]
                    ss.book_page = 0
        st.markdown('</div>', unsafe_allow_html=True)

def render_book_reader(lesson:dict):
    """Elegant single-page reader with cozy paper theme."""
    _inject_book_styles()
    pages = _split_pages(lesson["content"], max_chars=1200)
    total = len(pages)
    p = int(ss.get("book_page", 0))
    p = max(0, min(p, total-1))
    ss.book_page = p

    st.markdown('<div class="reader">', unsafe_allow_html=True)

    # top bar: back + title
    top = st.columns([1,5])
    with top[0]:
        if st.button("← Back to shelf", key="back_btn", type="secondary"):
            ss.book_open = None
            ss.book_page = 0
            st.rerun()
    with top[1]:
        st.markdown(f"<h3>📖 {lesson['title']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='meta'>{(lesson['tags'] or '').replace(',', ' · ')}</div>", unsafe_allow_html=True)

    # main content page
    html_body = "".join(f"<p>{para.strip().replace('<','&lt;').replace('>','&gt;')}</p>" for para in pages[p].split("\\n\\n"))
    st.markdown(f"<div class='paper'>{html_body}</div>", unsafe_allow_html=True)

    # nav row
    if total > 1:
        colA, colB, colC = st.columns([2,4,2])
        with colA:
            if st.button("◀ Prev", disabled=(p==0)):
                ss.book_page = max(0,p-1); st.rerun()
        with colB:
            st.caption(f"Page {p+1} of {total}")
        with colC:
            if st.button("Next ▶", disabled=(p>=total-1)):
                ss.book_page = min(total-1,p+1); st.rerun()
    else:
        st.markdown("<div class='navrow'><span class='pill'>✨ Tiny, kind, doable ✨</span></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)



# ---------- logic ----------
def _quote_to_tip(quote: str) -> str:
    q = quote.lower()
    if "courage" in q: return "Say hi to one new person today."
    if "listen" in q:  return "Ask how someone’s day went—then reflect one detail."
    if "small" in q:   return "Share one tiny win with a friend."
    if "friend" in q:  return "Send a kind check-in to someone you care about."
    return "Start a short conversation with someone nearby today."

def get_daily_affirmation(user_id: str):
    """Return today's affirmation; ensure no repeats within last 30 days."""
    con = _db(); cur = con.cursor()
    today = _today().isoformat()

    row = cur.execute("""
        SELECT a.id, a.text FROM user_daily_affirmations uda
        JOIN affirmations a ON a.id = uda.affirmation_id
        WHERE uda.user_id=? AND uda.date=? LIMIT 1
    """, (user_id, today)).fetchone()
    if row:
        con.close()
        return {"id": row["id"], "text": row["text"], "date": today}

    recent_ids = {
        r["affirmation_id"]
        for r in cur.execute("""
            SELECT affirmation_id FROM user_daily_affirmations
            WHERE user_id=? AND date >= date('now','-30 day')
        """, (user_id,)).fetchall()
    }
    candidates = cur.execute("SELECT id, text FROM affirmations").fetchall()
    pool = [c for c in candidates if c["id"] not in recent_ids] or candidates
    choice = random.choice(pool)

    cur.execute("INSERT OR REPLACE INTO user_daily_affirmations(user_id, date, affirmation_id) VALUES(?,?,?)",
                (user_id, today, choice["id"]))
    con.commit(); con.close()
    return {"id": choice["id"], "text": choice["text"], "date": today}

def get_daily_sprout(user_id: str):
    """Daily quote + action tip (quote rotates; tip derived)."""
    con = _db(); cur = con.cursor()
    row = cur.execute("""
        SELECT dq.id, dq.quote, dq.author
        FROM user_sprout_log usl
        JOIN daily_quotes dq ON dq.id = usl.quote_id
        WHERE usl.user_id=? AND usl.shown_on=? ORDER BY usl.id DESC LIMIT 1
    """, (user_id, _today())).fetchone()
    if row:
        con.close()
        return {"id": row["id"], "quote": row["quote"], "author": row["author"],
                "tip": _quote_to_tip(row["quote"])}

    row = cur.execute("""
        SELECT dq.id, dq.quote, dq.author
        FROM daily_quotes dq
        WHERE dq.id NOT IN (
            SELECT quote_id FROM user_sprout_log
            WHERE user_id=? AND shown_on >= date('now','-14 day')
        )
        ORDER BY RANDOM() LIMIT 1
    """, (user_id,)).fetchone()
    if not row:
        row = cur.execute("SELECT id, quote, author FROM daily_quotes ORDER BY RANDOM() LIMIT 1").fetchone()

    cur.execute("INSERT INTO user_sprout_log(user_id, quote_id, shown_on) VALUES(?,?,?)",
                (user_id, row["id"], _today()))
    con.commit(); con.close()
    return {"id": row["id"], "quote": row["quote"], "author": row["author"],
            "tip": _quote_to_tip(row["quote"])}

def _pick_two_distinct(all_ids):
    if len(all_ids) < 2:
        return all_ids, []
    i1 = random.randrange(len(all_ids))
    i2 = (i1 + random.randrange(1, len(all_ids))) % len(all_ids)
    if i1 == i2:
        i2 = (i1 + 1) % len(all_ids)
    return [all_ids[i1], all_ids[i2]]

def _assign_weekly_pair(user_id: str, iso_year: int, iso_week: int, cur):
    cur.execute("DELETE FROM user_week_progress_v2 WHERE user_id=? AND iso_year=? AND iso_week=?",
                (user_id, iso_year, iso_week))
    all_goals = cur.execute("SELECT id FROM weekly_goals ORDER BY id").fetchall()
    all_ids = [r["id"] for r in all_goals]
    if not all_ids:
        return
    ids = _pick_two_distinct(all_ids)
    for gid in ids:
        cur.execute("""
            INSERT OR IGNORE INTO user_week_progress_v2(user_id, iso_year, iso_week, goal_id, completed)
            VALUES (?,?,?,?,0)
        """, (user_id, iso_year, iso_week, gid))

def get_weekly_quests(user_id: str):
    con = _db(); cur = con.cursor()
    iso_year, iso_week = _iso_year_week()

    rows = cur.execute("""
        SELECT wg.id, wg.title, wg.description, wg.xp, uwp.completed
        FROM user_week_progress_v2 uwp
        JOIN weekly_goals wg ON wg.id = uwp.goal_id
        WHERE uwp.user_id=? AND uwp.iso_year=? AND uwp.iso_week=?
        ORDER BY wg.id
    """, (user_id, iso_year, iso_week)).fetchall()

    if len(rows) < 2:
        _assign_weekly_pair(user_id, iso_year, iso_week, cur)
        con.commit()
        rows = cur.execute("""
            SELECT wg.id, wg.title, wg.description, wg.xp, uwp.completed
            FROM user_week_progress_v2 uwp
            JOIN weekly_goals wg ON wg.id = uwp.goal_id
            WHERE uwp.user_id=? AND uwp.iso_year=? AND uwp.iso_week=?
            ORDER BY wg.id
        """, (user_id, iso_year, iso_week)).fetchall()

    if len(rows) == 2 and all(r["completed"] for r in rows):
        _assign_weekly_pair(user_id, iso_year, iso_week, cur)
        con.commit()
        rows = cur.execute("""
            SELECT wg.id, wg.title, wg.description, wg.xp, uwp.completed
            FROM user_week_progress_v2 uwp
            JOIN weekly_goals wg ON wg.id = uwp.goal_id
            WHERE uwp.user_id=? AND uwp.iso_year=? AND uwp.iso_week=?
            ORDER BY wg.id
        """, (user_id, iso_year, iso_week)).fetchall()

    con.close()
    return [{"id": r["id"], "title": r["title"], "description": r["description"],
             "xp": r["xp"], "completed": bool(r["completed"])} for r in rows]

def complete_weekly_quest(user_id: str, goal_id: int):
    con = _db(); cur = con.cursor()
    iso_year, iso_week = _iso_year_week()
    cur.execute("""
        UPDATE user_week_progress_v2 SET completed=1
        WHERE user_id=? AND iso_year=? AND iso_week=? AND goal_id=?
    """, (user_id, iso_year, iso_week, goal_id))
    con.commit(); con.close()
    return True

def regenerate_weekly_quests(user_id: str):
    con = _db(); cur = con.cursor()
    iso_year, iso_week = _iso_year_week()
    _assign_weekly_pair(user_id, iso_year, iso_week, cur)
    con.commit(); con.close()
    return True

def list_lessons(tag: str | None = None, limit: int = 50):
    con = _db(); cur = con.cursor()
    if tag:
        rows = cur.execute("""
            SELECT id, title, content, tags FROM mini_lessons
            WHERE tags LIKE ? ORDER BY id DESC LIMIT ?
        """, (f"%{tag}%", limit)).fetchall()
    else:
        rows = cur.execute("SELECT id, title, content, tags FROM mini_lessons ORDER BY id DESC LIMIT ?",
                           (limit,)).fetchall()
    con.close()
    return [{"id": r["id"], "title": r["title"], "content": r["content"], "tags": r["tags"]} for r in rows]

# ---------- page (UI: simple, Streamlit-native) ----------
from badges_logs import log_sprout_view, log_affirmation_open, log_activity
def page_communi():
    init_communigrow()
    user_id = get_user_id()
    _inject_communi_styles()
    log_daily_visit(user_id)
    # Header
    st.title("🌻 CommuniGrow ")
    st.caption("Lightweight daily nudges for connection and confidence 🌱")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Daily Sprout", "Weekly Quests", "Mini Lessons"])

    # DAILY SPROUT
    with tab1:
        sprout = get_daily_sprout(user_id)
        affirm = get_daily_affirmation(user_id)

    # logs (unchanged)
        log_sprout_view(user_id, quote=sprout.get("quote"), author=sprout.get("author"), tip=sprout.get("tip"))
        log_affirmation_open(user_id, text=affirm.get("text"), affirmation_id=affirm.get("id"))

    # sprout card stays as-is
        _inject_communi_styles()
        render_sprout_card(sprout)

    # affirmation section: heading + flip card INSIDE one box
        render_affirmation_section_component(
          affirm,
          gift_image_path="assets\affirmation.jpg",  # or your own path
          max_width=720,
          card_width=420,
          height=420,
        )  





    # WEEKLY QUESTS
    with tab2:
        st.subheader("Your Weekly Quests")
        quests = get_weekly_quests(user_id)

        if not quests:
            st.info("No challenges yet — they’ll generate automatically.")
        else:
            iso_year, iso_week = _iso_year_week()
            for q in quests:
                with st.container(border=True):
                    top = st.columns([1, 5, 2])
                    with top[0]:
                        st.markdown("#### ✅" if q["completed"] else "#### ⭕")
                    with top[1]:
                        st.markdown(f"**{q['title']}**")
                        if q["description"]:
                            st.write(q["description"])
                    with top[2]:
                        st.metric(label="XP", value=q["xp"])

                    if q["completed"]:
                        st.success("Completed")
                    else:
                        btn_key = f"wkq_{iso_year}_{iso_week}_{q['id']}"
                        if st.button("Mark as done", key=btn_key):
                            # mark complete in weekly progress
                            complete_weekly_quest(user_id, q["id"])
                            # 🔹 log quest completion (deduped by quest_id/title per day) + credit XP
                            log_activity(
                                user_id,
                                "quest_complete",
                                {"quest_id": q["id"], "title": q["title"], "xp": q["xp"]},
                            )
                            st.rerun()

        st.divider()
        if st.button("Get different challenges"):
            regenerate_weekly_quests(user_id)
            st.rerun()

    # MINI LESSONS
        # MINI LESSONS (Book Shelf)
    with tab3:
        st.subheader("Grow with Mitra")
        tag = st.selectbox(
          "Pick a topic",
          ["all","embarrassment","confidence","conversation","texting","groups",
           "speaking","rejection","anxiety","friendship","nonverbal","personality"],
          index=0, key="mitra_tag_book"
        )

        lessons = list_lessons(None if tag == "all" else tag)

        if "book_open" not in ss:
          ss.book_open = None
        if "book_page" not in ss:
          ss.book_page = 0

        if not lessons:
          st.info("Lessons will appear here soon.")
        else:
        # If a book is open, show the reader; otherwise show the shelf
          open_id = ss.get("book_open")
          if open_id:
            lesson = next((L for L in lessons if L["id"] == open_id), None)
            # If the open book isn't in the current filter, fall back to shelf
            if lesson is None:
                ss.book_open = None
                render_book_shelf(lessons)
            else:
                render_book_reader(lesson)
          else:
            render_book_shelf(lessons)




# ===================== BADGES & LOGS PAGE (v2 with Journal & Mood) =====================
import streamlit as st
from badges_logs import (
    log_sprout_view,
    log_affirmation_open,
    log_activity,
    log_daily_visit,
    init_badges_logs_schema,
    page_badges_logs,   # ✅ add this line
)


ss = st.session_state
if "user_id" not in ss:
    ss.user_id = "demo-user"  # replace with your stable id generator

# Ensure tables exist (do this once during app boot)
init_badges_logs_schema()



from badges_logs import log_activity




# ---------------------- ROUTING ----------------------
if ss.active_tab == "Chat":
    if ss.intro_step != "done":
        if ss.intro_step == "welcome":
            welcome_card()
        elif ss.intro_step == "confetti":
            confetti_screen()
        elif ss.intro_step == "caution":
            caution_screen()
    else:
        # Language selector visible only inside Chat (not in sidebar)
        chat_language_picker()
        L = LANG[ss.lang]
        page_chat()

elif ss.active_tab == "Exercises":
    page_exercises()

elif ss.active_tab == "Diary":
    page_diary()
    _sync_diary_activity_for_today()

elif ss.active_tab == "Games":
    page_games()

elif ss.active_tab == "CommuniGrow":
    page_communi()

elif ss.active_tab == "Badges & Logs":
    page_badges_logs()

elif ss.active_tab == "Helpline":
    page_helpline()
