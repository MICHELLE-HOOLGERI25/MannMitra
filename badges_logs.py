# badges_logs.py
import os, json, sqlite3, csv, datetime, hashlib
import streamlit as st

# DB_PATH = os.path.join("data", "mannmitra.db")
# os.makedirs("data", exist_ok=True)

# NEW:
from data.db_setup import DB_PATH  # ← use the shared path
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

USER_STORE = os.path.join("data", "users.json")

def _get_user_id_fallback():
    try:
        if os.path.exists(USER_STORE):
            with open(USER_STORE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("default_user_id"):
                    return data["default_user_id"]
    except Exception:
        pass
    return "demo-user"


# ---------- utils ----------
def _today(): return datetime.date.today()
def _iso(d): return d.isoformat()
def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def _inject_styles():
    st.markdown("""
    <style>
      .mm-h1 { font-size: 1.9rem; font-weight: 800; margin-bottom: .5rem; }
      .badge-pill {
        display:flex; flex-direction:column; gap:.3rem;
        padding:12px 14px; border-radius:12px; margin-bottom:10px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
      }
      .badge-owned { border-color:#6BE675; }
      .mm-subtle { color:#aaa; font-size:0.9rem; }
    </style>
    """, unsafe_allow_html=True)

# ---------- schema & seeds ----------
def init_badges_logs_schema():
    con = _db(); cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS user_xp (
      user_id TEXT PRIMARY KEY,
      total_xp INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS badges (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL UNIQUE,
      description TEXT,
      icon TEXT,
      criteria TEXT NOT NULL,   -- 'streak' or 'xp'
      threshold INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS user_badges (
      id INTEGER PRIMARY KEY,
      user_id TEXT NOT NULL,
      badge_id INTEGER NOT NULL,
      awarded_on DATE NOT NULL,
      UNIQUE(user_id, badge_id)
    );
    CREATE TABLE IF NOT EXISTS user_activity_log (
      id INTEGER PRIMARY KEY,
      user_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      date TEXT NOT NULL,
      payload TEXT,
      payload_hash TEXT
    );
    """)
    # migrate: add payload_hash if missing
    try:
        cur.execute("SELECT payload_hash FROM user_activity_log LIMIT 1")
    except Exception:
        cur.execute("ALTER TABLE user_activity_log ADD COLUMN payload_hash TEXT")
    # idempotency index
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_user_kind_date_hash
        ON user_activity_log(user_id, kind, date, payload_hash)
    """)

    # Desired catalog with CORRECT icon counts
    seeds = [
        # Streaks
        ("7-Day Streak",  "Opened MannMitra 7 days in a row",  "🔥",            "streak", 7),
        ("14-Day Streak", "Showed up 14 days in a row",        "🔥🔥",          "streak", 14),
        ("30-Day Streak", "Consistency hero: 30 days straight","🔥🔥🔥",        "streak", 30),
        ("60-Day Streak", "Unstoppable: 60 days streak",       "🔥🔥🔥🔥",      "streak", 60),
        ("90-Day Streak", "Showed up 90 days in a row",        "🔥🔥🔥🔥🔥",    "streak", 90),
        # XP
        ("Level 1: 100 XP",  "Earned 100 XP",   "⭐",            "xp", 100),
        ("Level 2: 250 XP",  "Earned 250 XP",   "⭐⭐",           "xp", 250),
        ("Level 3: 500 XP",  "Earned 500 XP",   "⭐⭐⭐",         "xp", 500),
        ("Level 4: 1000 XP", "Earned 1000 XP",  "⭐⭐⭐⭐",       "xp", 1000),
        ("Level 5: 1500 XP", "Earned 1500 XP",  "⭐⭐⭐⭐⭐",     "xp", 1500),
    ]

    # Insert any missing
    names = {r["name"] for r in cur.execute("SELECT name FROM badges").fetchall()}
    to_add = [s for s in seeds if s[0] not in names]
    if to_add:
        cur.executemany(
            "INSERT INTO badges(name,description,icon,criteria,threshold) VALUES(?,?,?,?,?)",
            to_add
        )

    # **Update existing** badges to correct icons (so your current DB fixes itself)
    for name, _, icon, _, _ in seeds:
        cur.execute("UPDATE badges SET icon=? WHERE name=?", (icon, name))

    con.commit(); con.close()

# ---------- XP / streak / badges ----------
_XP_DEFAULTS = {
    "sprout_view": 1,
    "affirmation_open": 1,
    "quest_complete": 10,
    "journal_entry": 5,
    "mood_entry": 2,
    "game_play": 3,
    "exercise_complete": 6,
    "daily_visit": 1,          # <-- ADD THIS
}

_ONCE_PER_DAY = {"sprout_view", "affirmation_open", "journal_entry", "daily_visit"}  # <-- ADD daily_visit

def _add_xp(user_id, xp):
    con = _db(); cur = con.cursor()
    row = cur.execute("SELECT total_xp FROM user_xp WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        cur.execute("INSERT INTO user_xp(user_id,total_xp) VALUES(?,?)", (user_id, max(0,int(xp))))
    else:
        cur.execute("UPDATE user_xp SET total_xp=? WHERE user_id=?", ((row["total_xp"] or 0)+int(xp), user_id))
    con.commit(); con.close()

def get_total_xp(user_id) -> int:
    con = _db(); cur = con.cursor()
    row = cur.execute("SELECT total_xp FROM user_xp WHERE user_id=?", (user_id,)).fetchone()
    con.close()
    return int(row["total_xp"]) if row and row["total_xp"] is not None else 0

def get_current_streak(user_id) -> int:
    con = _db(); cur = con.cursor()
    rows = cur.execute(
        "SELECT DISTINCT date FROM user_activity_log WHERE user_id=?",
        (user_id,)
    ).fetchall()
    con.close()
    days = {r["date"] for r in rows}
    if not days: return 0
    streak, d = 0, _today()
    while _iso(d) in days:
        streak += 1
        d = d - datetime.timedelta(days=1)
    return streak

def _maybe_award_badges(user_id):
    total_xp, streak = get_total_xp(user_id), get_current_streak(user_id)
    con = _db(); cur = con.cursor()
    owned = {r["badge_id"] for r in cur.execute("SELECT badge_id FROM user_badges WHERE user_id=?", (user_id,)).fetchall()}
    for b in cur.execute("SELECT id,criteria,threshold FROM badges").fetchall():
        meets = (streak >= b["threshold"]) if b["criteria"]=="streak" else (total_xp >= b["threshold"])
        if meets and b["id"] not in owned:
            cur.execute("INSERT OR IGNORE INTO user_badges(user_id,badge_id,awarded_on) VALUES(?,?,?)",
                        (user_id, b["id"], _iso(_today())))
    con.commit(); con.close()

def prune_old_activity(days=30):
    cutoff = _iso(_today() - datetime.timedelta(days=days))
    con = _db(); cur = con.cursor()
    cur.execute("DELETE FROM user_activity_log WHERE date < ?", (cutoff,))
    con.commit(); con.close()

# ---------- logging (idempotent) ----------
def _payload_fingerprint(kind: str, payload: dict | None) -> str:
    if kind in _ONCE_PER_DAY:
        key = {"once": True}
    else:
        p = payload or {}
        key = {
            "title": p.get("title"),
            "quest_id": p.get("quest_id"),
            "exercise": p.get("exercise"),
            "game": p.get("game"),
            "xp": p.get("xp"),
            "who5": p.get("who5"),     # <-- Add this
            "mood": p.get("mood"),   
        }
    blob = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def log_activity(user_id: str, kind: str, payload: dict | None = None):
    payload = payload or {}
    today = _iso(_today())
    fp = _payload_fingerprint(kind, payload)

    con = _db(); cur = con.cursor()
    inserted = False
    try:
        cur.execute("DELETE FROM user_activity_log WHERE user_id=? AND kind=? AND date=?", (user_id, kind, today))
        cur.execute(
            "INSERT INTO user_activity_log(user_id,kind,date,payload,payload_hash) VALUES(?,?,?,?,?)",
            (user_id, kind, today, json.dumps(payload, ensure_ascii=False), fp)
        )
        inserted = (cur.rowcount == 1)
        con.commit()
    except sqlite3.IntegrityError:
        inserted = False
    finally:
        con.close()

    if inserted:
        xp = payload.get("xp", _XP_DEFAULTS.get(kind, 0))
        if xp:
            _add_xp(user_id, int(xp))
        _maybe_award_badges(user_id)
        prune_old_activity(30)

# ---------- optional helpers for richer logs ----------
def log_sprout_view(user_id: str, quote: str | None = None, author: str | None = None, tip: str | None = None):
    payload = {}
    if quote: payload["quote"] = str(quote)
    if author: payload["author"] = str(author)
    if tip: payload["tip"] = str(tip)
    log_activity(user_id, "sprout_view", payload)

def log_affirmation_open(user_id: str, text: str | None = None, affirmation_id: int | None = None):
    payload = {}
    if text: payload["affirmation"] = str(text)
    if affirmation_id is not None: payload["affirmation_id"] = int(affirmation_id)
    log_activity(user_id, "affirmation_open", payload)

def log_daily_visit(user_id: str):
    # Counts once per day due to ONCE_PER_DAY + payload_hash
    log_activity(user_id, "daily_visit", {})

def log_affirmation_open(user_id: str, text: str | None = None, affirmation_id: int | None = None):
    payload = {}
    if text: payload["affirmation"] = str(text)
    if affirmation_id is not None: payload["affirmation_id"] = int(affirmation_id)
    log_activity(user_id, "affirmation_open", payload)



# ---------- queries ----------
def get_badges(user_id):
    con = _db(); cur = con.cursor()
    all_badges = cur.execute("SELECT * FROM badges ORDER BY id").fetchall()
    owned_rows = cur.execute("""
        SELECT b.*, ub.awarded_on FROM user_badges ub
        JOIN badges b ON b.id=ub.badge_id
        WHERE ub.user_id=? ORDER BY ub.awarded_on DESC
    """, (user_id,)).fetchall()
    con.close()
    owned_ids = {r["id"] for r in owned_rows}
    locked = [b for b in all_badges if b["id"] not in owned_ids]
    return owned_rows, locked

def _read_today_mood_from_csv():
    path = os.path.join("data","mood_log.csv"); today = _iso(_today())
    if not os.path.exists(path): return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("date") or "").strip()==today: return row.get("mood")
    except Exception: return None
    return None

def _read_reflection_today():
    path = os.path.join("data","reflection_log.csv"); today = _iso(_today())
    if not os.path.exists(path): return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("date") or "").strip()==today:
                    return row.get("reflection") or row.get("entry")
    except Exception: return None
    return None

def who5_label(score: int) -> str:
    if score <= 9: return "Low"
    if score <= 14: return "Fair"
    if score <= 19: return "Good"
    return "Excellent"

# ---------- daily summary builder ----------
def _summary_rows_for_date(user_id: str, date_iso: str, include_sprout: bool):
    con = _db(); cur = con.cursor()
    rows = cur.execute("""
    SELECT kind, payload FROM user_activity_log
    WHERE user_id=? AND date=?
    ORDER BY id ASC
    """, (user_id, date_iso)).fetchall()

    con.close()

    seen_sprout = False
    sprout_quote = sprout_author = sprout_tip = None
    seen_affirm = False
    affirm_text = None
    who5 = None
    mood_name = None
    diary_notes = []
    xp_total = 0

    for r in rows:
        kind = r["kind"]
        try:
            p = json.loads(r["payload"]) if r["payload"] else {}
        except Exception:
            p = {}
        xp_total += int(p.get("xp", _XP_DEFAULTS.get(kind, 0)) or 0)

        if kind == "sprout_view":
            seen_sprout = True
            sprout_quote = p.get("quote", sprout_quote)
            sprout_author = p.get("author", sprout_author)
            sprout_tip = p.get("tip", sprout_tip)
        elif kind == "affirmation_open":
            seen_affirm = True
            affirm_text = p.get("affirmation", affirm_text)
        elif kind == "mood_entry":
            if "who5" in p: who5 = int(p["who5"])
            if "mood" in p and str(p["mood"]).strip(): mood_name = str(p["mood"]).strip()
        elif kind == "journal_entry":
            if isinstance(p.get("notes"), list) and p["notes"]:
                diary_notes = []
                for i, ans in enumerate(p["notes"], start=1):
                    ans = (str(ans) or "").strip()
                    if ans: diary_notes.append(f"Q{i}: {ans}")
            else:
                q_parts = []
                for key in ("q1","q2","q3"):
                    if key in p and (str(p[key]) or "").strip():
                        q_parts.append(f"{key.upper()}: {str(p[key]).strip()}")
                if q_parts: diary_notes = q_parts
                elif (p.get("note") or "").strip():
                    diary_notes = [str(p["note"]).strip()]

    data = []
    if include_sprout:
        if seen_sprout:
            if sprout_quote:
                det = f"“{sprout_quote}”"
                if sprout_author: det += f" — {sprout_author}"
                if sprout_tip: det += f"\n tip: {sprout_tip}"
                data.append({"Activity": "🌼 Daily Sprout", "Details": det})
            else:
                data.append({"Activity": "🌼 Daily Sprout", "Details": "Shown"})
        else:
            data.append({"Activity": "🌼 Daily Sprout", "Details": "—"})

    data.append({"Activity": "🎁 Affirmation", "Details": affirm_text if seen_affirm else "—"})

    diary_lines = []
    if who5 is not None: diary_lines.append(f"WHO-5: {who5}/25 ({who5_label(who5)})")
    if mood_name: diary_lines.append(f"mood: {mood_name}")
    diary_lines.extend(diary_notes)
    data.append({"Activity": "📔 Diary", "Details": "\n".join(diary_lines) if diary_lines else "—"})

    data.append({"Activity": "✨ XP earned", "Details": f"+{xp_total} XP" if xp_total > 0 else "No XP"})
    return data

# ---------- UI ----------
def page_badges_logs():
    init_badges_logs_schema()
    _inject_styles()

    ss = st.session_state
    user_id = ss.get("user_id") or _get_user_id_fallback()

    st.markdown('<div class="mm-h1">Badges & Logs</div>', unsafe_allow_html=True)

    tab_badges, tab_activity = st.tabs(["🏅 Badges", "📔 Activities & Reflections"])

    # TAB 1
    with tab_badges:
        c1, c2 = st.columns(2)
        c1.metric("Total XP", get_total_xp(user_id))
        c2.metric("Current Streak (days)", get_current_streak(user_id))

        owned, locked = get_badges(user_id)

        st.subheader("Your Badges")
        if not owned:
            st.caption("No badges yet — complete quests, keep a streak, write journals, or log moods to earn some!")
        else:
            cols = st.columns(3)
            for i, b in enumerate(owned):
                with cols[i % 3]:
                    st.markdown(
                        f"<div class='badge-pill badge-owned'>{b['icon']} <b>{b['name']}</b>"
                        f"<span class='mm-subtle'>{b['description']}</span>"
                        f"<span class='mm-subtle'>Earned on {b['awarded_on']}</span></div>",
                        unsafe_allow_html=True
                    )

        st.subheader("Locked Badges")
        if not locked:
            st.caption("🎉 You’ve unlocked all available badges!")
        else:
            cols2 = st.columns(3)
            for i, b in enumerate(locked):
                with cols2[i % 3]:
                    tip = f"Reach {b['threshold']} days streak" if b["criteria"]=="streak" else f"Reach {b['threshold']} XP"
                    st.markdown(
                        f"<div class='badge-pill'>{b['icon']}<br/>"
                        f"<b>{b['name']}</b>"
                        f"<span class='mm-subtle'>{b['description']}</span>"
                        f"<span class='mm-subtle'>Unlock: {tip}</span></div>",
                        unsafe_allow_html=True
                    )

    # TAB 2
    with tab_activity:
        con = _db(); cur = con.cursor()
        date_rows = cur.execute("""
            SELECT DISTINCT date FROM user_activity_log
            WHERE user_id=? AND date >= date('now','-30 day')
            ORDER BY date DESC
        """, (user_id,)).fetchall()
        con.close()

        if not date_rows:
            st.caption("No activity yet. Your last 30 days will appear here as you use the app.")
        else:
            today_iso = _iso(_today())
            for drow in date_rows:
                d = drow["date"]
                with st.expander(d if d != today_iso else f"{d} (today)", expanded=(d == today_iso)):
                    rows = _summary_rows_for_date(user_id, d, include_sprout=True)

                    def _escape(s: str) -> str:
                        return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

                    html = [
                        "<table style='width:100%; border-collapse:collapse;'>",
                        "<thead><tr>",
                        "<th style='text-align:left; padding:6px; border-bottom:1px solid rgba(255,255,255,0.15)'>Activity</th>",
                        "<th style='text-align:left; padding:6px; border-bottom:1px solid rgba(255,255,255,0.15)'>Details</th>",
                        "</tr></thead><tbody>",
                    ]
                    for r in rows:
                        act = _escape(str(r["Activity"]))
                        det_html = _escape(str(r["Details"] or "")).replace("\n", "<br/>")
                        html.append(
                            f"<tr><td style='padding:6px 6px 10px 6px; vertical-align:top;'>{act}</td>"
                            f"<td style='padding:6px 6px 10px 6px; white-space:normal;'>{det_html}</td></tr>"
                        )
                    html.append("</tbody></table>")
                    st.markdown("\n".join(html), unsafe_allow_html=True)

# alias for older code
page_badges_tree = page_badges_logs
