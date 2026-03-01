
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo

from streamlit_calendar import calendar
import holidays

# Google Calendar API (optional)
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
except Exception:
    Credentials = None  # type: ignore
    Flow = None  # type: ignore
    build = None  # type: ignore


APP_TAG = "[LeavePlannerApp]"

# -----------------------------------------------------------------------------
# Storage paths
# -----------------------------------------------------------------------------
# Streamlit Community Cloud runs your code from a read-only repo checkout.
# Writing beside app.py can raise "Read-only file system" and crash the app.
#
# We therefore:
# - Prefer the app directory if it is writable (typical local dev)
# - Otherwise fall back to a writable user/home directory (cloud)
# - Otherwise fall back to /tmp

APP_DIR = Path(__file__).resolve().parent


def _pick_writable_dir() -> Path:
    # Local dev: repo/app folder is normally writable
    try:
        if os.access(str(APP_DIR), os.W_OK):
            return APP_DIR
    except Exception:
        pass

    # Cloud: use a user-writable directory
    candidates = [Path.home() / ".leave_planner", Path("/tmp") / "leave_planner"]
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test = d / ".write_test"
            test.write_text("ok", encoding="utf-8")
            try:
                test.unlink()
            except Exception:
                pass
            return d
        except Exception:
            continue
    return Path("/tmp")


DATA_DIR = _pick_writable_dir()
DATA_FILE = str(DATA_DIR / "planner_data.json")
GCAL_TOKEN_FILE = str(DATA_DIR / "gcal_token.json")

SECRETS_TOML_EXAMPLE = """gcal_client_id = \"YOUR_CLIENT_ID.apps.googleusercontent.com\"
gcal_client_secret = \"YOUR_CLIENT_SECRET\"
gcal_redirect_uri = \"https://YOUR-APP-NAME.streamlit.app/\"
"""


@dataclass
class PlannerEvent:
    id: str
    title: str
    kind: str  # AL, AL_MAND, PL, NOTE, UNPAID
    start_iso: str  # timezone-aware ISO
    end_iso: str    # timezone-aware ISO (EXCLUSIVE for all-day)
    all_day: bool = False
    meta: Optional[Dict] = None
    gcal_event_id: Optional[str] = None

    def start_dt(self) -> datetime:
        return datetime.fromisoformat(self.start_iso)

    def end_dt(self) -> datetime:
        return datetime.fromisoformat(self.end_iso)


@dataclass
class Settings:
    tz_name: str = "Australia/Sydney"
    region: str = "NSW"  # NSW.. or NZ

    # Employment baseline
    employment_start: str = ""  # YYYY-MM-DD

    # Work day definition
    hours_per_workday: float = 7.0
    work_start: str = "09:00"
    work_end: str = "17:00"
    unpaid_break_minutes: int = 60

    # Weekly roster pattern (Mon..Sun): OFF / OFFICE / REMOTE
    roster: Dict[int, str] = None  # type: ignore

    # Accrual config
    pay_frequency: str = "fortnightly"  # weekly/fortnightly/monthly
    pay_anchor_end: str = ""            # pay period END date (date picker)

    al_method: str = "pay_period"       # pay_period or weeks_per_year
    al_weeks_per_year: float = 6.0
    al_accrued_per_pay_hours: float = 4.31
    pl_accrued_per_pay_hours: float = 2.15

    # Opening balances as-of employment_start
    al_open_hours: float = 0.0
    pl_open_hours: float = 0.0

    # Leave year
    leave_year_start: str = ""
    leave_year_end: str = ""

    # Payslip / earnings (optional)
    hourly_rate: float = 56.51
    super_rate: float = 0.12
    payg_withheld_per_pay: float = 0.0
    other_deductions_per_pay: float = 0.0

    def __post_init__(self):
        if self.roster is None:
            self.roster = {1: "OFFICE", 2: "OFFICE", 3: "OFFICE", 4: "OFF", 5: "REMOTE", 6: "OFF", 7: "OFF"}

    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.tz_name)

    def work_window(self) -> Tuple[time, time]:
        return time.fromisoformat(self.work_start), time.fromisoformat(self.work_end)

    def work_window_minutes(self) -> int:
        ws, we = self.work_window()
        dt0 = datetime.combine(date(2000, 1, 1), ws)
        dt1 = datetime.combine(date(2000, 1, 1), we)
        return max(int((dt1 - dt0).total_seconds() // 60), 0)

    def paid_window_minutes(self) -> int:
        return max(self.work_window_minutes() - int(self.unpaid_break_minutes), 0)

    def paid_minutes_ratio(self) -> float:
        ww = self.work_window_minutes()
        return 1.0 if ww <= 0 else self.paid_window_minutes() / ww


@dataclass
class PlannerState:
    settings: Settings
    events: List[PlannerEvent]
    action_stack: List[Dict]


# -------------------------
# Persistence
# -------------------------
def _file_exists(path: str) -> bool:
    return os.path.exists(path)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, s: str) -> None:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we can't create parents (very locked-down FS), let open() raise a useful error.
        pass
    with p.open("w", encoding="utf-8") as f:
        f.write(s)


def load_state() -> PlannerState:
    if st.session_state.get("planner_loaded"):
        return st.session_state["planner_state"]

    if _file_exists(DATA_FILE):
        raw = json.loads(_read_text(DATA_FILE))
        settings = Settings(**raw["settings"])
        events = [PlannerEvent(**e) for e in raw.get("events", [])]
        action_stack = raw.get("action_stack", [])
    else:
        settings = Settings()
        today = date.today()
        settings.leave_year_start = date(today.year, 1, 1).isoformat()
        settings.leave_year_end = date(today.year, 12, 31).isoformat()
        settings.employment_start = date(today.year, 1, 19).isoformat()
        settings.pay_anchor_end = date(today.year, 2, 19).isoformat()
        events = []
        action_stack = []

    state = PlannerState(settings=settings, events=events, action_stack=action_stack)
    st.session_state["planner_state"] = state
    st.session_state["planner_loaded"] = True
    return state


def save_state(state: PlannerState) -> None:
    payload = {
        "settings": asdict(state.settings),
        "events": [asdict(e) for e in state.events],
        "action_stack": state.action_stack,
    }
    try:
        _write_text(DATA_FILE, json.dumps(payload, indent=2, ensure_ascii=False))
        st.session_state["planner_last_save_error"] = ""
    except Exception as e:
        # Don't crash the whole app if persistence isn't available.
        st.session_state["planner_last_save_error"] = str(e)


# -------------------------
# Holidays
# -------------------------
def build_holidays(settings: Settings, start: date, end: date) -> Dict[date, str]:
    hols: Dict[date, str] = {}
    if settings.region == "NZ":
        nz = holidays.NewZealand(years=range(start.year, end.year + 1))
        for d, name in nz.items():
            if start <= d <= end:
                hols[d] = name
        return hols

    au = holidays.Australia(subdiv=settings.region, years=range(start.year, end.year + 1))
    for d, name in au.items():
        if start <= d <= end:
            hols[d] = name
    return hols


# -------------------------
# Date helpers
# -------------------------
def iter_dates(d0: date, d1: date):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def is_workday(settings: Settings, d: date) -> bool:
    return settings.roster.get(d.isoweekday(), "OFF") != "OFF"


def employment_active(settings: Settings, d: date) -> bool:
    if not settings.employment_start:
        return True
    return d >= date.fromisoformat(settings.employment_start)


# -------------------------
# Pay periods
# -------------------------
def period_start_from_end(settings: Settings, period_end: date) -> date:
    if settings.pay_frequency == "weekly":
        return period_end - timedelta(days=6)
    if settings.pay_frequency == "fortnightly":
        return period_end - timedelta(days=13)
    prev_end = (period_end - relativedelta(months=1))
    return prev_end + timedelta(days=1)


def generate_period_ends(settings: Settings, start: date, end: date) -> List[date]:
    if not settings.pay_anchor_end:
        return []
    anchor = date.fromisoformat(settings.pay_anchor_end)
    ends = set()

    if settings.pay_frequency in ("weekly", "fortnightly"):
        step = timedelta(days=7 if settings.pay_frequency == "weekly" else 14)

        cur = anchor
        while cur > end:
            cur -= step
        while cur >= start:
            ends.add(cur)
            cur -= step

        cur = anchor
        while cur < start:
            cur += step
        while cur <= end:
            ends.add(cur)
            cur += step
    else:
        cur = anchor
        while cur > end:
            cur = (cur - relativedelta(months=1))
        while cur >= start:
            ends.add(cur)
            cur = (cur - relativedelta(months=1))
        cur = anchor
        while cur < start:
            cur = (cur + relativedelta(months=1))
        while cur <= end:
            ends.add(cur)
            cur = (cur + relativedelta(months=1))

    return sorted(list(ends))


# -------------------------
# Accrual (pay-period based)
# -------------------------
def spread_accrual_over_workdays(settings: Settings, d0: date, d1: date, accrued_per_pay_hours: float) -> Dict[date, float]:
    accrual = {d: 0.0 for d in iter_dates(d0, d1)}
    period_ends = generate_period_ends(settings, d0, d1)
    if not period_ends:
        return accrual

    for pe in period_ends:
        ps = period_start_from_end(settings, pe)
        span_start = max(ps, d0)
        span_end = min(pe, d1)
        wds = [d for d in iter_dates(span_start, span_end) if is_workday(settings, d) and employment_active(settings, d)]
        if not wds:
            continue
        per = accrued_per_pay_hours / len(wds)
        for d in wds:
            accrual[d] += per

    return accrual


def annual_daily_accrual(settings: Settings, d0: date, d1: date) -> Dict[date, float]:
    if settings.al_method == "weeks_per_year":
        total_workdays = sum(1 for d in iter_dates(d0, d1) if is_workday(settings, d) and employment_active(settings, d))
        if total_workdays <= 0:
            return {d: 0.0 for d in iter_dates(d0, d1)}
        total_hours = settings.al_weeks_per_year * 4.0 * settings.hours_per_workday
        per = total_hours / total_workdays
        return {d: (per if is_workday(settings, d) and employment_active(settings, d) else 0.0) for d in iter_dates(d0, d1)}

    return spread_accrual_over_workdays(settings, d0, d1, settings.al_accrued_per_pay_hours)


def personal_daily_accrual(settings: Settings, d0: date, d1: date) -> Dict[date, float]:
    return spread_accrual_over_workdays(settings, d0, d1, settings.pl_accrued_per_pay_hours)


# -------------------------
# Leave hours by overlap with work window
# -------------------------
def overlap_minutes(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> int:
    start = max(a0, b0)
    end = min(a1, b1)
    if end <= start:
        return 0
    return int((end - start).total_seconds() // 60)


def work_window_for_day(settings: Settings, d: date) -> Tuple[datetime, datetime]:
    tz = settings.tz()
    ws, we = settings.work_window()
    start = datetime.combine(d, ws).replace(tzinfo=tz)
    end = datetime.combine(d, we).replace(tzinfo=tz)
    return start, end


def event_leave_hours_on_day(settings: Settings, ev: PlannerEvent, d: date) -> float:
    if not is_workday(settings, d) or not employment_active(settings, d):
        return 0.0

    tz = settings.tz()
    day_ws, day_we = work_window_for_day(settings, d)

    ev0 = ev.start_dt().astimezone(tz)
    ev1 = ev.end_dt().astimezone(tz)

    mins = overlap_minutes(ev0, ev1, day_ws, day_we)
    if mins <= 0:
        return 0.0

    paid_ratio = settings.paid_minutes_ratio()
    hrs = (mins * paid_ratio) / 60.0
    return min(hrs, settings.hours_per_workday)


def event_span_dates(settings: Settings, ev: PlannerEvent, d0: date, d1: date) -> Optional[Tuple[date, date]]:
    tz = settings.tz()
    ev0 = ev.start_dt().astimezone(tz)
    ev1 = ev.end_dt().astimezone(tz)
    d_start = ev0.date()

    if ev.all_day:
        last_day = ev1.date() - timedelta(days=1)
    else:
        if ev1.time() == time(0, 0) and ev1 > ev0:
            last_day = ev1.date() - timedelta(days=1)
        else:
            last_day = ev1.date()

    span_start = max(d_start, d0)
    span_end = min(last_day, d1)
    if span_end < span_start:
        return None
    return span_start, span_end


def build_daily_ledger(state: PlannerState) -> pd.DataFrame:
    s = state.settings
    d0 = date.fromisoformat(s.leave_year_start)
    d1 = date.fromisoformat(s.leave_year_end)
    tz = s.tz()

    hols = build_holidays(s, d0, d1)
    al_acc = annual_daily_accrual(s, d0, d1)
    pl_acc = personal_daily_accrual(s, d0, d1)

    al_taken = {d: 0.0 for d in iter_dates(d0, d1)}
    pl_taken = {d: 0.0 for d in iter_dates(d0, d1)}
    al_mand_taken = {d: 0.0 for d in iter_dates(d0, d1)}
    unpaid_taken = {d: 0.0 for d in iter_dates(d0, d1)}
    notes = {d: [] for d in iter_dates(d0, d1)}

    for ev in state.events:
        if ev.kind == "NOTE":
            dd = ev.start_dt().astimezone(tz).date()
            if d0 <= dd <= d1:
                notes.setdefault(dd, []).append(ev.title)
            continue

        if ev.kind not in {"AL", "AL_MAND", "PL", "UNPAID"}:
            continue

        span = event_span_dates(s, ev, d0, d1)
        if not span:
            continue
        span_start, span_end = span

        for d in iter_dates(span_start, span_end):
            if d in hols:
                continue
            h = event_leave_hours_on_day(s, ev, d)
            if ev.kind == "PL":
                pl_taken[d] += h
            elif ev.kind == "AL_MAND":
                al_taken[d] += h
                al_mand_taken[d] += h
            elif ev.kind == "UNPAID":
                unpaid_taken[d] += h
            else:
                al_taken[d] += h

    rows = []
    al_bal = s.al_open_hours
    pl_bal = s.pl_open_hours

    for d in iter_dates(d0, d1):
        al_bal += al_acc.get(d, 0.0) - al_taken.get(d, 0.0)
        pl_bal += pl_acc.get(d, 0.0) - pl_taken.get(d, 0.0)

        rows.append({
            "date": d.isoformat(),
            "weekday": d.strftime("%a"),
            "roster": s.roster.get(d.isoweekday(), "OFF"),
            "is_workday": is_workday(s, d),
            "employment_active": employment_active(s, d),
            "holiday": hols.get(d, ""),
            "al_accrue_h": round(al_acc.get(d, 0.0), 4),
            "pl_accrue_h": round(pl_acc.get(d, 0.0), 4),
            "al_taken_h": round(al_taken.get(d, 0.0), 4),
            "al_mand_taken_h": round(al_mand_taken.get(d, 0.0), 4),
            "pl_taken_h": round(pl_taken.get(d, 0.0), 4),
            "unpaid_h": round(unpaid_taken.get(d, 0.0), 4),
            "al_balance_h": round(al_bal, 4),
            "pl_balance_h": round(pl_bal, 4),
            "notes": " | ".join(notes.get(d, [])),
        })

    df = pd.DataFrame(rows)
    total_mand = df["al_mand_taken_h"].sum()
    df["al_mand_remaining_h"] = (total_mand - df["al_mand_taken_h"].cumsum()).round(4)
    df["al_safe_balance_h"] = (df["al_balance_h"] - df["al_mand_remaining_h"]).round(4)
    df["date_dt"] = pd.to_datetime(df["date"])
    return df


KIND_META = {
    "AL": {"icon": "✈️", "label": "Annual Leave"},
    "AL_MAND": {"icon": "🎄", "label": "Annual Leave (Mandatory)"},
    "PL": {"icon": "🤒", "label": "Personal/Carer’s"},
    "UNPAID": {"icon": "⛔", "label": "Unpaid leave"},
    "NOTE": {"icon": "🗒️", "label": "Note (no leave)"},
    "HOLIDAY": {"icon": "🎉", "label": "Public Holiday"},
    "ROSTER": {"icon": "📍", "label": "Roster"},
}

ROSTER_LABEL = {"OFFICE": "Office", "REMOTE": "Remote", "OFF": "Off"}

def new_event_id() -> str:
    return str(uuid.uuid4())

def to_fullcalendar_event(ev: PlannerEvent) -> Dict:
    meta = KIND_META.get(ev.kind, {"icon": "", "label": ev.kind})
    title = f"{meta.get('icon','')} {ev.title}".strip()
    if ev.all_day:
        return {"id": ev.id, "title": title, "start": ev.start_dt().date().isoformat(), "end": ev.end_dt().date().isoformat(), "allDay": True}
    return {"id": ev.id, "title": title, "start": ev.start_iso, "end": ev.end_iso, "allDay": False}

def push_action(state: PlannerState, event_ids: List[str]) -> None:
    state.action_stack.append({"action_id": new_event_id(), "event_ids": event_ids})

def undo_last_action(state: PlannerState) -> bool:
    if not state.action_stack:
        return False
    last = state.action_stack.pop()
    remove_ids = set(last.get("event_ids", []))
    state.events = [e for e in state.events if e.id not in remove_ids]
    return True


# -------------------------
# Google Calendar (optional)
# -------------------------
def gcal_available() -> bool:
    return Credentials is not None and Flow is not None and build is not None

def get_gcal_secrets() -> Optional[Dict[str, str]]:
    try:
        cid = (st.secrets.get("gcal_client_id", "") or "").strip()
        csec = (st.secrets.get("gcal_client_secret", "") or "").strip()
        redir = (st.secrets.get("gcal_redirect_uri", "") or "").strip()
    except Exception:
        return None
    if not cid or not csec or not redir:
        return None
    return {"client_id": cid, "client_secret": csec, "redirect_uri": redir}

def make_flow(secrets: Dict[str, str], scopes: List[str]) -> Flow:
    cfg = {
        "web": {
            "client_id": secrets["client_id"],
            "project_id": "leave-planner",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": secrets["client_secret"],
            "redirect_uris": [secrets["redirect_uri"]],
        }
    }
    return Flow.from_client_config(cfg, scopes=scopes, redirect_uri=secrets["redirect_uri"])

def load_gcal_creds() -> Optional[Credentials]:
    if os.path.exists(GCAL_TOKEN_FILE) and Credentials is not None:
        try:
            raw = json.loads(_read_text(GCAL_TOKEN_FILE))
            return Credentials.from_authorized_user_info(raw)
        except Exception:
            return None
    return None

def save_gcal_creds(creds: Credentials) -> None:
    _write_text(GCAL_TOKEN_FILE, creds.to_json())

def gcal_service(creds: Credentials):
    return build("calendar", "v3", credentials=creds)

def gcal_list_calendars(creds: Credentials) -> List[Dict]:
    svc = gcal_service(creds)
    items = []
    page_token = None
    while True:
        res = svc.calendarList().list(pageToken=page_token).execute()
        items.extend(res.get("items", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return items

def gcal_upsert_events(creds: Credentials, calendar_id: str, events: List[PlannerEvent], tz_name: str) -> Tuple[int, int]:
    svc = gcal_service(creds)
    created = 0
    updated = 0
    for ev in events:
        if ev.kind not in {"AL", "AL_MAND", "PL", "NOTE", "UNPAID"}:
            continue
        icon = KIND_META.get(ev.kind, {}).get("icon", "")
        summary = f"{icon} {ev.title}".strip()
        body = {"summary": summary, "description": f"{APP_TAG}\nkind={ev.kind}\nlocal_id={ev.id}"}
        if ev.all_day:
            body["start"] = {"date": ev.start_dt().date().isoformat()}
            body["end"] = {"date": ev.end_dt().date().isoformat()}
        else:
            body["start"] = {"dateTime": ev.start_iso, "timeZone": tz_name}
            body["end"] = {"dateTime": ev.end_iso, "timeZone": tz_name}

        if ev.gcal_event_id:
            try:
                svc.events().update(calendarId=calendar_id, eventId=ev.gcal_event_id, body=body).execute()
                updated += 1
                continue
            except Exception:
                ev.gcal_event_id = None

        res = svc.events().insert(calendarId=calendar_id, body=body).execute()
        ev.gcal_event_id = res.get("id")
        created += 1
    return created, updated


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Leave Planner", layout="wide")
state = load_state()
s = state.settings

st.title("Leave Planner — rostered workdays, pay-period accrual, hourly leave, public holidays")

with st.sidebar:
    st.header("Core settings")

    region = st.selectbox("Public holiday region", ["NSW","VIC","QLD","SA","WA","TAS","ACT","NT","NZ"],
                          index=["NSW","VIC","QLD","SA","WA","TAS","ACT","NT","NZ"].index(s.region) if s.region in ["NSW","VIC","QLD","SA","WA","TAS","ACT","NT","NZ"] else 0)
    tz_map = {
        "NSW": "Australia/Sydney",
        "VIC": "Australia/Melbourne",
        "QLD": "Australia/Brisbane",
        "SA": "Australia/Adelaide",
        "WA": "Australia/Perth",
        "TAS": "Australia/Hobart",
        "ACT": "Australia/Sydney",
        "NT": "Australia/Darwin",
        "NZ": "Pacific/Auckland",
    }
    tz_default = tz_map.get(region, s.tz_name)
    tz_name = st.selectbox("Time zone", sorted(set(list(tz_map.values()) + [s.tz_name])),
                           index=sorted(set(list(tz_map.values()) + [s.tz_name])).index(s.tz_name) if s.tz_name in sorted(set(list(tz_map.values()) + [s.tz_name])) else 0)

    s.region = region
    s.tz_name = tz_name

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        ly_start = st.date_input("Leave year start", value=date.fromisoformat(s.leave_year_start) if s.leave_year_start else date(date.today().year,1,1))
    with c2:
        ly_end = st.date_input("Leave year end", value=date.fromisoformat(s.leave_year_end) if s.leave_year_end else date(date.today().year,12,31))
    s.leave_year_start = ly_start.isoformat()
    s.leave_year_end = ly_end.isoformat()

    st.divider()

    emp_start = st.date_input("Employment start date", value=date.fromisoformat(s.employment_start) if s.employment_start else ly_start)
    s.employment_start = emp_start.isoformat()
    st.caption("Opening balances are treated as of Employment start date.")

    st.divider()

    st.subheader("Workday definition")
    colw1, colw2 = st.columns(2)
    with colw1:
        s.work_start = st.text_input("Work start (HH:MM)", s.work_start)
    with colw2:
        s.work_end = st.text_input("Work end (HH:MM)", s.work_end)
    s.unpaid_break_minutes = int(st.number_input("Unpaid break (minutes)", min_value=0, max_value=180, value=int(s.unpaid_break_minutes), step=5))
    s.hours_per_workday = float(st.number_input("Paid hours per workday", min_value=0.0, max_value=12.0, value=float(s.hours_per_workday), step=0.25))
    st.caption(f"Work window: {s.work_window_minutes()} mins | Paid: {s.paid_window_minutes()} mins (ratio {s.paid_minutes_ratio():.3f})")

    st.divider()

    st.subheader("Weekly roster pattern")
    opts = ["OFF","OFFICE","REMOTE"]
    labels = {"OFF": "Off", "OFFICE": "Office", "REMOTE": "Remote"}
    weekdays = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    cols = st.columns(7)
    for i, wd in enumerate(weekdays, start=1):
        with cols[i-1]:
            cur = s.roster.get(i, "OFF")
            s.roster[i] = st.selectbox(wd, opts, index=opts.index(cur) if cur in opts else 0, format_func=lambda x: labels[x], key=f"roster_{i}")

    st.divider()

    st.subheader("Accrual")
    s.pay_frequency = st.selectbox("Pay frequency", ["weekly","fortnightly","monthly"],
                                   index=["weekly","fortnightly","monthly"].index(s.pay_frequency) if s.pay_frequency in ["weekly","fortnightly","monthly"] else 1)

    anchor_default = date.fromisoformat(s.pay_anchor_end) if s.pay_anchor_end else emp_start
    pay_anchor = st.date_input("Pay period END (anchor)", value=anchor_default)
    s.pay_anchor_end = pay_anchor.isoformat()

    s.al_method = st.selectbox("Annual leave method", ["pay_period","weeks_per_year"], index=0 if s.al_method=="pay_period" else 1)
    if s.al_method == "pay_period":
        s.al_accrued_per_pay_hours = float(st.number_input("Annual leave accrued per pay (hours)", min_value=0.0, max_value=20.0, value=float(s.al_accrued_per_pay_hours), step=0.01))
    else:
        s.al_weeks_per_year = float(st.number_input("Annual leave weeks/year", min_value=0.0, max_value=12.0, value=float(s.al_weeks_per_year), step=0.25))
    s.pl_accrued_per_pay_hours = float(st.number_input("Personal/Carer’s accrued per pay (hours)", min_value=0.0, max_value=20.0, value=float(s.pl_accrued_per_pay_hours), step=0.01))

    st.divider()

    st.subheader("Opening balances")
    s.al_open_hours = float(st.number_input("Annual leave opening (hours)", value=float(s.al_open_hours), step=0.25))
    s.pl_open_hours = float(st.number_input("Personal/Carer’s opening (hours)", value=float(s.pl_open_hours), step=0.25))

    st.divider()

    st.subheader("Payslip / earnings (optional)")
    s.hourly_rate = float(st.number_input("Hourly rate", value=float(s.hourly_rate), step=0.01))
    s.super_rate = float(st.number_input("Super rate (e.g. 0.12)", value=float(s.super_rate), step=0.005, format="%.3f"))
    s.payg_withheld_per_pay = float(st.number_input("PAYG withheld per pay (optional)", value=float(s.payg_withheld_per_pay), step=1.0))
    s.other_deductions_per_pay = float(st.number_input("Other deductions per pay (optional)", value=float(s.other_deductions_per_pay), step=1.0))

    st.divider()

    if st.button("↩️ Undo last action", use_container_width=True):
        if undo_last_action(state):
            save_state(state)
            st.success("Undid the last action.")
            st.rerun()
        else:
            st.info("Nothing to undo.")

    if st.button("🧹 Clear ALL events", use_container_width=True):
        state.events = []
        state.action_stack = []
        save_state(state)
        st.success("Cleared.")
        st.rerun()

    last_err = (st.session_state.get("planner_last_save_error") or "").strip()
    if last_err:
        st.warning(f"Autosave could not write to disk ({last_err}). You can still export/import JSON below.")
    st.caption(f"Data file: {DATA_FILE}")


ledger = build_daily_ledger(state)

st.subheader("Balance trend")
trend = ledger[["date_dt","al_balance_h","al_safe_balance_h","pl_balance_h"]].set_index("date_dt")
st.line_chart(trend)

today = date.today()
asof_date = st.date_input("As-of date (balances + payslip summary)", value=min(max(today, date.fromisoformat(s.leave_year_start)), date.fromisoformat(s.leave_year_end)))
asof_row = ledger[ledger["date"] == asof_date.isoformat()]
if len(asof_row) == 1:
    al_bal = float(asof_row["al_balance_h"].iloc[0])
    al_safe = float(asof_row["al_safe_balance_h"].iloc[0])
    pl_bal = float(asof_row["pl_balance_h"].iloc[0])
else:
    al_bal = float(ledger["al_balance_h"].iloc[-1])
    al_safe = float(ledger["al_safe_balance_h"].iloc[-1])
    pl_bal = float(ledger["pl_balance_h"].iloc[-1])

m1, m2, m3 = st.columns(3)
m1.metric("Annual leave balance (h)", f"{al_bal:.2f}")
m2.metric("Annual SAFE balance (h)", f"{al_safe:.2f}")
m3.metric("Personal/Carer’s balance (h)", f"{pl_bal:.2f}")

st.divider()

# Calendar
d0 = date.fromisoformat(s.leave_year_start)
d1 = date.fromisoformat(s.leave_year_end)
tz = s.tz()
hols = build_holidays(s, d0, d1)

holiday_events: List[PlannerEvent] = []
for hd, name in hols.items():
    start = datetime.combine(hd, time(0, 0)).replace(tzinfo=tz)
    end = start + timedelta(days=1)
    holiday_events.append(PlannerEvent(id=f"holiday_{hd.isoformat()}", title=name, kind="HOLIDAY", start_iso=start.isoformat(), end_iso=end.isoformat(), all_day=True))

show_roster = st.checkbox("Show roster overlay (Office/Remote labels)", value=True)
roster_events: List[PlannerEvent] = []
if show_roster:
    for d in iter_dates(d0, d1):
        tag = s.roster.get(d.isoweekday(), "OFF")
        if tag == "OFF":
            continue
        start = datetime.combine(d, time(0, 0)).replace(tzinfo=tz)
        end = start + timedelta(days=1)
        roster_events.append(PlannerEvent(id=f"roster_{d.isoformat()}", title=ROSTER_LABEL.get(tag, tag), kind="ROSTER", start_iso=start.isoformat(), end_iso=end.isoformat(), all_day=True))

fc_events = [to_fullcalendar_event(e) for e in (holiday_events + roster_events + state.events)]

cA, cB, cC = st.columns([1.2, 1.2, 1.6])
with cA:
    view = st.selectbox("Calendar view", ["dayGridMonth", "timeGridWeek", "timeGridDay"], index=0)
with cB:
    initial_date = st.date_input("Jump to date", value=asof_date, key="jump_to")
with cC:
    st.caption("Week view: drag-select hours for partial leave / notes.")

cal_options = {
    "initialView": view,
    "initialDate": initial_date.isoformat(),
    "height": 760,
    "selectable": True,
    "editable": False,
    "headerToolbar": {"left": "prev,next today", "center": "title", "right": "dayGridMonth,timeGridWeek,timeGridDay"},
    "businessHours": [{"daysOfWeek": [i-1 for i in range(1,8) if s.roster.get(i, "OFF") != "OFF"], "startTime": s.work_start, "endTime": s.work_end}],
    "slotMinTime": "06:00:00",
    "slotMaxTime": "22:00:00",
    "nowIndicator": True,
}
cal_state = calendar(events=fc_events, options=cal_options, key="calendar")

st.divider()

with st.expander("➕ Quick add (date range)", expanded=True):
    r1, r2, r3, r4 = st.columns([1.3, 1.1, 1.2, 1.2])
    with r1:
        add_title = st.text_input("Title", value="Skiing")
    with r2:
        add_kind = st.selectbox("Type", ["AL","AL_MAND","PL","NOTE","UNPAID"], index=0, format_func=lambda k: KIND_META[k]["label"])
    with r3:
        add_start = st.date_input("Start date", value=initial_date, key="range_start")
    with r4:
        add_end = st.date_input("End date", value=initial_date, key="range_end")

    t1, t2, t3 = st.columns([1.2, 1.1, 1.7])
    with t1:
        is_all_day = st.checkbox("All-day", value=True)
    with t2:
        half_day = st.checkbox("Half-day", value=False)
    with t3:
        custom_hours = st.number_input("Custom hours/day (optional)", min_value=0.0, max_value=12.0, value=0.0, step=0.25)

    if st.button("Add range", type="primary"):
        if add_end < add_start:
            st.error("End date must be on/after start date.")
        else:
            ids = []
            for d in iter_dates(add_start, add_end):
                title = (add_title or "").strip() or KIND_META[add_kind]["label"]
                if is_all_day:
                    ev_start = datetime.combine(d, time(0, 0)).replace(tzinfo=tz)
                    ev_end = ev_start + timedelta(days=1)
                    ev_all_day = True
                else:
                    ws, we = s.work_window()
                    ev_start = datetime.combine(d, ws).replace(tzinfo=tz)
                    ev_end = datetime.combine(d, we).replace(tzinfo=tz)
                    ev_all_day = False

                if add_kind in {"AL","AL_MAND","PL","UNPAID"}:
                    if custom_hours > 0:
                        ws, _ = s.work_window()
                        ev_start = datetime.combine(d, ws).replace(tzinfo=tz)
                        ev_end = ev_start + timedelta(hours=float(custom_hours) / max(s.paid_minutes_ratio(), 1e-6))
                        ev_all_day = False
                    elif half_day:
                        ws = time.fromisoformat(s.work_start)
                        ev_start = datetime.combine(d, ws).replace(tzinfo=tz)
                        ev_end = ev_start + timedelta(hours=(s.hours_per_workday / 2.0) / max(s.paid_minutes_ratio(), 1e-6))
                        ev_all_day = False

                ev = PlannerEvent(id=new_event_id(), title=title, kind=add_kind, start_iso=ev_start.isoformat(), end_iso=ev_end.isoformat(), all_day=ev_all_day, meta={"source": "range"})
                state.events.append(ev)
                ids.append(ev.id)

            push_action(state, ids)
            save_state(state)
            st.success(f"Added {len(ids)} entries. Use Undo if needed.")
            st.rerun()

if cal_state and cal_state.get("select"):
    sel = cal_state["select"]
    start_iso = sel.get("start")
    end_iso = sel.get("end")
    if start_iso and end_iso:
        st.subheader("Selection")
        st.write(f"Selected: `{start_iso}` → `{end_iso}`")
        with st.form("add_from_selection"):
            title = st.text_input("Title", value="Leave / Note")
            kind = st.selectbox("Type", ["AL","AL_MAND","PL","NOTE","UNPAID"], format_func=lambda k: KIND_META[k]["label"])
            submit = st.form_submit_button("Add selected block")
        if submit:
            ev = PlannerEvent(id=new_event_id(), title=(title or "").strip() or KIND_META[kind]["label"], kind=kind, start_iso=start_iso, end_iso=end_iso, all_day=False, meta={"source": "select"})
            state.events.append(ev)
            push_action(state, [ev.id])
            save_state(state)
            st.success("Added.")
            st.rerun()

if cal_state and cal_state.get("eventClick"):
    ev_id = cal_state["eventClick"]["event"]["id"]
    match = next((e for e in state.events if e.id == ev_id), None)
    st.subheader("Clicked event")
    if match:
        st.info(f"Selected: **{KIND_META.get(match.kind,{}).get('label', match.kind)}** — **{match.title}**")
        c1, c2 = st.columns([1.6, 1.2])
        with c1:
            new_title = st.text_input("Edit title", value=match.title, key="edit_title")
        with c2:
            new_kind = st.selectbox("Edit type", ["AL","AL_MAND","PL","NOTE","UNPAID"],
                                    index=["AL","AL_MAND","PL","NOTE","UNPAID"].index(match.kind),
                                    format_func=lambda k: KIND_META[k]["label"], key="edit_kind")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save edits"):
                match.title = new_title
                match.kind = new_kind
                save_state(state)
                st.success("Saved.")
                st.rerun()
        with b2:
            if st.button("🗑️ Delete event"):
                state.events = [e for e in state.events if e.id != match.id]
                save_state(state)
                st.success("Deleted.")
                st.rerun()
    else:
        st.caption("Holiday/roster event (read-only).")

st.divider()

st.header("List view (your entries)")
if state.events:
    rows = []
    for ev in state.events:
        rows.append({
            "title": ev.title,
            "type": ev.kind,
            "start": ev.start_dt().astimezone(tz).isoformat(),
            "end": ev.end_dt().astimezone(tz).isoformat(),
            "all_day": ev.all_day,
        })
    df_events = pd.DataFrame(rows).sort_values("start")
    st.dataframe(df_events, use_container_width=True, hide_index=True)
else:
    st.caption("No entries yet. Use Quick add or select on the calendar.")

st.header("Exports")
x1, x2, x3 = st.columns([1.1, 1.1, 1.8])
with x1:
    st.download_button("⬇️ Download ledger CSV", data=ledger.to_csv(index=False).encode("utf-8"), file_name="leave_ledger.csv", mime="text/csv")
with x2:
    payload = json.dumps({"settings": asdict(s), "events": [asdict(e) for e in state.events]}, indent=2, ensure_ascii=False)
    st.download_button("⬇️ Export planner JSON", data=payload.encode("utf-8"), file_name="planner_export.json", mime="application/json")
with x3:
    uploaded = st.file_uploader("Import planner JSON", type=["json"])
    if uploaded is not None:
        raw = json.loads(uploaded.read().decode("utf-8"))
        s2 = Settings(**raw["settings"])
        events2 = [PlannerEvent(**e) for e in raw.get("events", [])]
        state.settings = s2
        state.events = events2
        state.action_stack = []
        save_state(state)
        st.success("Imported.")
        st.rerun()

st.divider()

st.header("Pay-period summary (payslip style)")
st.caption("Estimate only. Paid leave generally does not change gross pay; it affects leave balances. Use UNPAID leave to reduce pay.")

def find_period_for_date(settings: Settings, target: date) -> Optional[Tuple[date, date]]:
    ends = generate_period_ends(settings, date.fromisoformat(settings.leave_year_start), date.fromisoformat(settings.leave_year_end))
    for pe in ends:
        ps = period_start_from_end(settings, pe)
        if ps <= target <= pe:
            return ps, pe
    return None

period = find_period_for_date(s, asof_date)
if not period:
    st.warning("No pay periods could be generated (check pay anchor end date).")
else:
    ps, pe = period
    hols2 = build_holidays(s, ps, pe)
    workdays = [d for d in iter_dates(ps, pe) if is_workday(s, d) and employment_active(s, d) and d not in hols2]
    unpaid_hours = float(ledger[(ledger["date_dt"] >= pd.to_datetime(ps)) & (ledger["date_dt"] <= pd.to_datetime(pe))]["unpaid_h"].sum())
    paid_hours = max(len(workdays) * s.hours_per_workday - unpaid_hours, 0.0)

    gross = paid_hours * s.hourly_rate
    super_pay = gross * s.super_rate
    net_est = gross - s.payg_withheld_per_pay - s.other_deductions_per_pay

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pay period", f"{ps.isoformat()} → {pe.isoformat()}")
    c2.metric("Paid hours (est.)", f"{paid_hours:.2f}")
    c3.metric("Gross (est.)", f"${gross:,.2f}")
    c4.metric("Super (est.)", f"${super_pay:,.2f}")
    st.metric("Net (est., if you provide PAYG/deductions)", f"${net_est:,.2f}")

st.divider()

st.header("Google Calendar sync (optional)")
if not gcal_available():
    st.warning("Google Calendar dependencies not available. Install requirements.")
else:
    secrets = get_gcal_secrets()
    if not secrets:
        st.info("Not configured. Local: create `.streamlit/secrets.toml`. Cloud: App → Settings → Secrets.")
        st.code(SECRETS_TOML_EXAMPLE, language="toml")
    else:
        qp = st.query_params
        auth_code = qp.get("code")

        if "gcal_creds" not in st.session_state:
            st.session_state["gcal_creds"] = load_gcal_creds()
        creds: Optional[Credentials] = st.session_state.get("gcal_creds")
        scopes = ["https://www.googleapis.com/auth/calendar"]

        if creds is None and auth_code:
            try:
                flow = make_flow(secrets, scopes=scopes)
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                st.session_state["gcal_creds"] = creds
                save_gcal_creds(creds)
                st.query_params.clear()
                st.success("Google Calendar connected.")
            except Exception as e:
                st.error(f"Auth failed: {e}")

        if creds is None:
            flow = make_flow(secrets, scopes=scopes)
            auth_url, _state = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
            st.link_button("🔐 Connect Google Calendar", auth_url)
            st.caption(f"Tokens are stored in {GCAL_TOKEN_FILE} (ignored by Git).")
        else:
            st.success("Connected ✅")
            try:
                cals = gcal_list_calendars(creds)
                cal_map = {f"{c.get('summary','(no name)')} — {c.get('id')}": c.get("id") for c in cals}
                pick = st.selectbox("Choose calendar to sync to", list(cal_map.keys()))
                cal_id = cal_map[pick]
                if st.button("⬆️ Push planner events to Google Calendar", type="primary"):
                    created, updated = gcal_upsert_events(creds, cal_id, state.events, s.tz_name)
                    save_state(state)
                    st.success(f"Synced. Created: {created}, Updated: {updated}")
            except Exception as e:
                st.error(f"Google Calendar error: {e}")

save_state(state)
