# leave_planner (Streamlit)

A visual leave + accrual planner for a 4‑day work week, with:
- Annual leave + personal/carer’s leave accrual (pay-period based or weeks/year)
- Office/Remote/Off roster pattern (Office+Remote count as workdays)
- Hour-by-hour leave blocks (drag-select in week view)
- Public holiday overlay (AU states + NZ)
- Google Calendar sync (optional, via OAuth)
- List view + pay-period “payslip style” summary

## Run locally (no venv)
Windows:
1. Open PowerShell/Command Prompt in this folder
2. Run:
   `py -m pip install --user -r requirements.txt`
3. Run:
   `py -m streamlit run app.py --server.port 8506`

macOS:
1. `python3 -m pip install --user -r requirements.txt`
2. `python3 -m streamlit run app.py --server.port 8506`

Open:
- http://localhost:8506

## Deploy to Streamlit Community Cloud
1. Create a GitHub repo named `leave_planner`
2. Push this folder to GitHub
3. Streamlit Cloud → New app → select repo/branch → main file `app.py`

### Streamlit Cloud secrets
In Streamlit Cloud → App → Settings → **Secrets**, paste:

```toml
gcal_client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
gcal_client_secret = "YOUR_CLIENT_SECRET"
gcal_redirect_uri = "https://YOUR-APP-NAME.streamlit.app/"
```

## Google Calendar OAuth setup (what you need to add in Google Cloud)
In Google Cloud Console:
- Enable **Google Calendar API**
- Configure OAuth consent screen (add yourself as a Test User if in Testing mode)
- Create **OAuth Client ID** (Web application)
- Authorised JavaScript origins:
  - Local: `http://localhost:8506`
  - Cloud: `https://YOUR-APP-NAME.streamlit.app`
- Authorised redirect URIs:
  - Local: `http://localhost:8506/`
  - Cloud: `https://YOUR-APP-NAME.streamlit.app/`

## Data storage
Local mode stores your planner data in `planner_data.json` beside `app.py`.
Streamlit Cloud stores it in the app container; for persistence between redeploys you should export/import JSON.

## Notes
- “All-day” events use EXCLUSIVE end dates (correct for FullCalendar & Google Calendar).
- Events spanning beyond the leave-year end are clamped for calculations (no KeyError).
