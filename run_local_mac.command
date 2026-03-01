#!/bin/bash
cd "$(dirname "$0")"
PORT=8506
python3 -m pip install --user -r requirements.txt
python3 -m streamlit run app.py --server.port ${PORT} --server.headless false
