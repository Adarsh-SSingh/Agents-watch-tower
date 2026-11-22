import collections
import concurrent.futures as futures
import html
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import psutil  # Optional: CPU / RAM gauges (pip install psutil)
except ImportError:
    psutil = None

API_URL = "http://127.0.0.1:8000"

# All timestamps are rendered in this timezone. Change it here to match your
# deployment region; every formatter reads from this single constant.
DISPLAY_TIMEZONE = "Asia/Kolkata"

st.set_page_config(
    page_title="AI Agent Watch Tower",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# STYLING
# Dark AgentGuard-style theme applied entirely through CSS so that no
# config.toml is required. All selectors target Streamlit's stable
# data-testid attributes.
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
    /* ---------- dark base ---------- */
    .stApp {background: #0b1120; color: #e5e7eb;}
    [data-testid="stHeader"] {background: transparent;}
    [data-testid="stSidebar"] {background: #0f172a; border-right: 1px solid rgba(148,163,184,.15);}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label,
    .stApp li, .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stMarkdownContainer"] {color: #e5e7eb;}
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px; border-color: rgba(148,163,184,.18) !important;
        background: rgba(15,23,42,.55);
    }
    .stTabs [data-baseweb="tab"] {color: #cbd5e1;}
    .stTabs [aria-selected="true"] {color: #22d3ee !important;}

    .block-container {padding-top: 3rem; padding-bottom: 3rem;}

    /* ---------- header ---------- */
    .ao-hero {
        padding: .9rem 1.3rem; border-radius: 14px; margin-bottom: 1rem;
        border: 1px solid rgba(148,163,184,.18);
        background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(139,92,246,.14));
    }
    .ao-hero h1 {margin: 0; font-size: 1.4rem; line-height: 1.25;}
    .ao-hero p {margin: .3rem 0 0 0; opacity: .75; font-size: .92rem;}

    .ao-card {
        padding: .9rem 1.1rem; border-radius: 12px; margin-bottom: .8rem;
        border: 1px solid rgba(148,163,184,.22); background: rgba(148,163,184,.07);
    }
    .ao-card .ao-label {opacity: .65; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em;}
    .ao-card .ao-value {font-size: 1.05rem; font-weight: 600; margin-top: .15rem;}

    .ao-pill {
        display: inline-block; padding: .1rem .65rem; border-radius: 999px;
        font-size: .78rem; font-weight: 600; border: 1px solid;
    }
    .ao-pill.success {color:#4ade80; background:rgba(34,197,94,.14);  border-color:rgba(34,197,94,.45);}
    .ao-pill.error   {color:#f87171; background:rgba(239,68,68,.14);  border-color:rgba(239,68,68,.45);}
    .ao-pill.running {color:#fbbf24; background:rgba(245,158,11,.14); border-color:rgba(245,158,11,.45);}
    .ao-pill.warn    {color:#fb923c; background:rgba(249,115,22,.14); border-color:rgba(249,115,22,.45);}
    .ao-pill.neutral {color:#94a3b8; background:rgba(148,163,184,.16);border-color:rgba(148,163,184,.45);}

    .ao-section {margin: 1.4rem 0 .4rem 0; font-size: 1.15rem; font-weight: 700;}
    [data-testid="stSidebar"] .ao-brand {font-size: 1.25rem; font-weight: 800; margin-bottom: .1rem;}

    /* ---------- overview KPI cards ---------- */
    .ag-kpi {
        display: flex; align-items: center; gap: .85rem;
        padding: .85rem 1.1rem; border-radius: 12px; min-height: 78px;
        border: 1px solid rgba(148,163,184,.22); background: rgba(148,163,184,.07);
    }
    .ag-kpi-icon {font-size: 1.6rem; line-height: 1;}
    .ag-kpi-label {font-size: .78rem; opacity: .7;}
    .ag-kpi-value {font-size: 1.45rem; font-weight: 700; line-height: 1.2;}
    .ag-kpi.ok   {border-color: rgba(34,197,94,.6);  background: linear-gradient(135deg, rgba(34,197,94,.22), rgba(34,197,94,.05));}
    .ag-kpi.ok .ag-kpi-value   {color: #4ade80;}
    .ag-kpi.warn {border-color: rgba(245,158,11,.6); background: linear-gradient(135deg, rgba(245,158,11,.22), rgba(245,158,11,.05));}
    .ag-kpi.warn .ag-kpi-value {color: #fbbf24;}
    .ag-kpi.bad  {border-color: rgba(239,68,68,.6);  background: linear-gradient(135deg, rgba(239,68,68,.22), rgba(239,68,68,.05));}
    .ag-kpi.bad .ag-kpi-value  {color: #f87171;}

    .ag-title {font-size: .95rem; font-weight: 700; margin: 0 0 .35rem 0;}
    .ag-sub   {font-size: .78rem; opacity: .6; margin-bottom: .4rem;}
    .ag-head  {font-size: 1.05rem; font-weight: 700; margin: .9rem 0 .4rem 0; opacity: .9;}

    .ag-scroll {overflow-y: auto;}
    .ag-table {width: 100%; border-collapse: collapse; font-size: .8rem;}
    .ag-table th {text-align: left; font-weight: 600; opacity: .6; padding: .35rem .5rem;
                  border-bottom: 1px solid rgba(148,163,184,.25); position: sticky; top: 0;
                  background: #0f172a;}
    .ag-table td {padding: .38rem .5rem; border-bottom: 1px solid rgba(148,163,184,.12); vertical-align: top;}
    .lv {font-weight: 700; font-size: .75rem;}
    .lv.inf {color: #4ade80;} .lv.wrn {color: #fbbf24;} .lv.err {color: #f87171;} .lv.run {color: #22d3ee;}
    .dot {display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; margin-right: .4rem;}
    .dot.crit {background: #ef4444;} .dot.warn {background: #f59e0b;} .dot.info {background: #22d3ee;}
    .st-ok {color: #4ade80;} .st-bad {color: #f87171;} .st-idle {color: #94a3b8;} .st-run {color: #22d3ee;}

    /* ---------- chrome clean-up ---------- */
    [data-testid="stAppDeployButton"], .stDeployButton, #MainMenu {display: none;}
    [data-testid="stBottom"] > div {background: #0b1120;}
    [data-testid="stChatInput"] {
        background: rgba(148,163,184,.08); border: 1px solid rgba(148,163,184,.28); border-radius: 14px;
    }
    [data-testid="stChatInput"] textarea {color: #e5e7eb;}
    [data-testid="stChatMessage"] {
        background: rgba(148,163,184,.05); border: 1px solid rgba(148,163,184,.12);
        border-radius: 12px; margin-bottom: .5rem;
    }
    [data-testid="stExpander"] {border-radius: 12px; border-color: rgba(148,163,184,.2);}
    .stTabs [data-baseweb="tab-highlight"] {background-color: #22d3ee;}

    /* ---------- buttons ---------- */
    [data-testid="stBaseButton-secondary"] {
        background: rgba(148,163,184,.08); color: #e5e7eb;
        border: 1px solid rgba(148,163,184,.28); border-radius: 10px; transition: all .15s;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        border-color: #22d3ee; color: #22d3ee; background: rgba(34,211,238,.08);
    }
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #0891b2, #7c3aed); color: #fff;
        border: none; border-radius: 10px; font-weight: 600;
    }
    [data-testid="stBaseButton-primary"]:hover {filter: brightness(1.12); color: #fff;}

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] > div:first-child {padding-top: .5rem;}
    .sb-brand {display: flex; align-items: center; gap: .75rem; padding: .3rem .2rem .6rem;}
    .sb-logo {
        width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center;
        justify-content: center; font-weight: 800; font-size: 1.3rem; color: #0b1120;
        background: linear-gradient(135deg, #22d3ee, #8b5cf6);
    }
    .sb-name {font-size: 1.25rem; font-weight: 800; line-height: 1.1;}
    .sb-tag {font-size: .74rem; opacity: .6; margin-top: .15rem;}
    .sb-label {
        font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
        opacity: .5; margin: 1.2rem 0 .45rem .2rem; font-weight: 700;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {gap: .35rem;}
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .6rem .85rem; border-radius: 10px; width: 100%;
        border: 1px solid transparent; cursor: pointer; transition: all .15s;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background: rgba(148,163,184,.10);}
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display: none;}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(34,211,238,.18), rgba(139,92,246,.18));
        border-color: rgba(34,211,238,.45);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {font-weight: 700;}
    .sb-status {
        display: flex; align-items: center; gap: .75rem;
        padding: .75rem .9rem; border-radius: 12px; border: 1px solid;
    }
    .sb-status.on  {background: rgba(34,197,94,.10); border-color: rgba(34,197,94,.4);}
    .sb-status.off {background: rgba(239,68,68,.10); border-color: rgba(239,68,68,.4);}
    .sb-dot {width: 10px; height: 10px; border-radius: 50%; flex: 0 0 10px;}
    .sb-status.on .sb-dot  {background: #22c55e; box-shadow: 0 0 0 4px rgba(34,197,94,.22);}
    .sb-status.off .sb-dot {background: #ef4444; box-shadow: 0 0 0 4px rgba(239,68,68,.22);}
    .sb-status-title {font-weight: 700; font-size: .9rem;}
    .sb-status-sub {font-size: .72rem; opacity: .65;}
    .sb-foot {font-size: .72rem; opacity: .5; margin-top: 1.2rem; line-height: 1.5;}

    /* ---------- overview extras ---------- */
    .ag-kpi-sub {font-size: .72rem; opacity: .6; margin-top: .1rem;}

    /* ---------- agent workspace ---------- */
    .ws-banner {
        display: flex; gap: 1rem; align-items: center; padding: 1.1rem 1.3rem;
        border-radius: 16px; margin: .6rem 0 1rem 0;
        border: 1px solid rgba(148,163,184,.2);
        background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(139,92,246,.12));
    }
    .ws-icon {
        flex: 0 0 54px; height: 54px; border-radius: 14px; display: flex;
        align-items: center; justify-content: center; font-size: 1.7rem;
        background: rgba(148,163,184,.14);
    }
    .ws-banner h3 {margin: 0; font-size: 1.25rem;}
    .ws-banner p {margin: .25rem 0 0 0; opacity: .72; font-size: .9rem;}
    .chip {
        display: inline-block; padding: .15rem .65rem; margin: .15rem .3rem .15rem 0;
        border-radius: 999px; font-size: .74rem; border: 1px solid rgba(148,163,184,.3);
        background: rgba(148,163,184,.08);
    }
    .chip.cyan {color: #22d3ee; border-color: rgba(34,211,238,.45); background: rgba(34,211,238,.10);}
    .step-row {display: flex; gap: .75rem; align-items: flex-start; margin: .7rem 0;}
    .step-num {
        flex: 0 0 26px; height: 26px; border-radius: 50%; display: flex; align-items: center;
        justify-content: center; font-size: .78rem; font-weight: 700; color: #0b1120;
        background: linear-gradient(135deg, #22d3ee, #8b5cf6);
    }
    .step-text b {display: block; font-size: .88rem;}
    .step-text span {font-size: .78rem; opacity: .65;}
    .empty-state {text-align: center; padding: 3rem 1rem;}
    .empty-state .big {font-size: 2.4rem;}
    .empty-state h4 {margin: .5rem 0 .25rem 0;}
    .empty-state p {opacity: .65; font-size: .88rem; margin: 0;}

    /* ================= ROUND 2: typography, sidebar, hero, icons ================= */
    html, body, .stApp, .stApp button, .stApp input, .stApp textarea,
    .stApp p, .stApp label, .stApp li, [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"], [data-baseweb="tab"], .ag-table, .ag-kpi-label, .ag-sub {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .ao-hero h1, .sb-name, .ag-kpi-value, .ws-banner h3, .ag-head, .ag-title,
    .ao-section, .ao-eyebrow, .empty-state h4, [data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', 'Inter', system-ui, sans-serif !important;
        letter-spacing: -0.01em;
    }
    .stApp h5 {font-size: 1.02rem; font-weight: 700;}
    code, pre, [data-testid="stCode"] *, .sb-code {
        font-family: 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace !important;
    }

    [data-testid="stDecoration"] {display: none;}
    [data-testid="stHeader"] {border-bottom: none; box-shadow: none;}

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
        background:
            radial-gradient(120% 55% at 0% 0%, rgba(34,211,238,.16), transparent 60%),
            radial-gradient(110% 50% at 100% 100%, rgba(139,92,246,.18), transparent 60%),
            #0c1226;
        border-right: 1px solid rgba(148,163,184,.14);
    }
    [data-testid="stSidebarContent"], [data-testid="stSidebar"] > div:first-child {background: transparent;}
    [data-testid="stSidebar"] .block-container {padding-top: 1rem;}

    .sb-brand {padding: .4rem .1rem .2rem;}
    .sb-logo {
        width: 46px; height: 46px; border-radius: 14px; color: #0b1120;
        background: linear-gradient(135deg, #22d3ee, #8b5cf6);
        box-shadow: 0 8px 22px rgba(34,211,238,.28);
    }
    .sb-logo svg {width: 26px; height: 26px;}
    .sb-name {font-size: 1.3rem; font-weight: 800;}
    .sb-tag {font-size: .72rem; opacity: .6;}

    .sb-label {display: flex; align-items: center; gap: .6rem; margin: 1.4rem 0 .55rem 0;}
    .sb-label::after {content: ""; flex: 1; height: 1px; background: rgba(148,163,184,.18);}

    /* nav menu (radio restyled) */
    [data-testid="stSidebar"] [role="radiogroup"] {gap: .4rem;}
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .7rem .9rem; border-radius: 12px; width: 100%;
        border: 1px solid rgba(148,163,184,.12); background: rgba(148,163,184,.05);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(34,211,238,.08); border-color: rgba(34,211,238,.3);
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"] > div:not(:has([data-testid="stMarkdownContainer"])),
    [data-testid="stSidebar"] [role="radiogroup"] label > div:not(:has([data-testid="stMarkdownContainer"])) {
        display: none !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p {font-size: .95rem; font-weight: 600;}
    [data-testid="stSidebar"] [role="radiogroup"] [data-testid="stIconMaterial"] {
        color: #94a3b8; font-size: 1.3rem; margin-right: .35rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(34,211,238,.24), rgba(139,92,246,.26));
        border-color: rgba(34,211,238,.6); box-shadow: 0 6px 18px rgba(34,211,238,.12);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) [data-testid="stIconMaterial"] {color: #22d3ee;}

    /* status + snapshot */
    @keyframes sbPulse {
        0%   {box-shadow: 0 0 0 0 rgba(34,197,94,.55);}
        70%  {box-shadow: 0 0 0 8px rgba(34,197,94,0);}
        100% {box-shadow: 0 0 0 0 rgba(34,197,94,0);}
    }
    .sb-status.on .sb-dot {animation: sbPulse 2s infinite;}
    .sb-snap {display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem;}
    .sb-tile {
        padding: .6rem .3rem; border-radius: 12px; text-align: center;
        border: 1px solid rgba(148,163,184,.18); background: rgba(148,163,184,.06);
    }
    .sb-tile b {display: block; font-size: 1.1rem; font-family: 'Plus Jakarta Sans', sans-serif;}
    .sb-tile span {font-size: .64rem; opacity: .6; text-transform: uppercase; letter-spacing: .08em;}
    .sb-tile.good b {color: #4ade80;} .sb-tile.bad b {color: #f87171;}
    .sb-code {
        font-size: .72rem; padding: .6rem .7rem; border-radius: 10px; margin-top: .4rem;
        background: #060a16; border: 1px solid rgba(148,163,184,.2); color: #67e8f9; word-break: break-all;
    }
    .sb-tip {
        margin-top: 1.4rem; padding: .85rem .95rem; border-radius: 12px; font-size: .76rem; line-height: 1.55;
        border: 1px dashed rgba(34,211,238,.35); background: rgba(34,211,238,.06);
    }
    .sb-tip b {color: #22d3ee;}

    /* ---------- hero ---------- */
    .ao-hero {
        display: flex; align-items: center; justify-content: space-between; gap: 1rem;
        padding: 1.25rem 1.5rem; border-radius: 18px; margin-bottom: 1.1rem;
        border: 1px solid rgba(148,163,184,.2);
        background:
            radial-gradient(80% 160% at 0% 0%, rgba(34,211,238,.16), transparent 60%),
            radial-gradient(80% 160% at 100% 100%, rgba(139,92,246,.22), transparent 60%),
            rgba(15,23,42,.7);
    }
    .ao-hero-left {display: flex; align-items: center; gap: 1.1rem;}
    .ao-hero-icon {
        flex: 0 0 58px; width: 58px; height: 58px; border-radius: 17px; color: #0b1120;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #22d3ee, #8b5cf6);
        box-shadow: 0 10px 26px rgba(34,211,238,.25);
    }
    .ao-hero-icon svg {width: 28px; height: 28px;}
    .ao-eyebrow {font-size: .7rem; letter-spacing: .2em; text-transform: uppercase; color: #22d3ee; font-weight: 700;}
    .ao-hero h1 {
        margin: .15rem 0 0 0; padding: 0; font-size: 1.75rem; font-weight: 800; line-height: 1.2;
        background: linear-gradient(90deg, #f8fafc, #a5b4fc);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        color: #f8fafc;
    }
    .ao-hero p {margin: .35rem 0 0 0; opacity: .7; font-size: .92rem;}
    .ao-live {
        display: flex; align-items: center; gap: .5rem; padding: .35rem .85rem; border-radius: 999px;
        font-size: .76rem; font-weight: 600; color: #4ade80; white-space: nowrap;
        border: 1px solid rgba(34,197,94,.4); background: rgba(34,197,94,.10);
    }
    .ao-live-dot {width: 8px; height: 8px; border-radius: 50%; background: #22c55e; animation: sbPulse 2s infinite;}

    /* ---------- KPI icon tiles, headings, banners ---------- */
    .ag-kpi-icon {
        flex: 0 0 44px; width: 44px; height: 44px; border-radius: 12px; font-size: 0;
        display: flex; align-items: center; justify-content: center;
        background: rgba(34,211,238,.14); color: #22d3ee;
    }
    .ag-kpi-icon.purple {background: rgba(139,92,246,.2); color: #a78bfa;}
    .ag-kpi-icon svg {width: 22px; height: 22px;}
    .ag-kpi.ok .ag-kpi-icon   {background: rgba(34,197,94,.2);  color: #4ade80;}
    .ag-kpi.warn .ag-kpi-icon {background: rgba(245,158,11,.2); color: #fbbf24;}
    .ag-kpi.bad .ag-kpi-icon  {background: rgba(239,68,68,.2);  color: #f87171;}
    .ag-kpi-value {font-size: 1.55rem; font-weight: 800;}

    .ag-head {display: flex; align-items: center; gap: .55rem; font-size: 1.1rem; font-weight: 700; margin: 1.1rem 0 .5rem 0;}
    .ag-head svg {width: 20px; height: 20px; color: #22d3ee;}

    .ws-icon {
        background: linear-gradient(135deg, rgba(34,211,238,.2), rgba(139,92,246,.24));
        border: 1px solid rgba(34,211,238,.3); color: #22d3ee; font-size: 0;
    }
    .ws-icon svg {width: 28px; height: 28px;}
    .empty-state .big {font-size: 0;}
    .empty-state .big svg {width: 46px; height: 46px; color: #22d3ee; opacity: .85;}

    /* ================= ROUND 3: tabs, inputs, cards, trace, workspace ================= */
    .block-container {padding-bottom: 8rem !important;}
    [data-testid="stHeaderActionElements"] {display: none !important;}
    .stApp h5 {padding: .2rem 0 .45rem 0; margin: 0;}

    /* tabs: no red box / underline */
    .stTabs [data-baseweb="tab-list"] {gap: .3rem;}
    .stTabs [data-baseweb="tab"] {
        border: none !important; outline: none !important; box-shadow: none !important;
        background: transparent !important; border-radius: 10px 10px 0 0; padding: .55rem .9rem;
    }
    .stTabs [data-baseweb="tab"]:hover {background: rgba(34,211,238,.07) !important;}
    .stTabs [data-baseweb="tab"]:focus, .stTabs [data-baseweb="tab"]:focus-visible {outline: none !important; box-shadow: none !important;}
    .stTabs [data-baseweb="tab-highlight"] {background-color: #22d3ee !important; height: 3px; border-radius: 3px;}
    .stTabs [data-baseweb="tab-border"] {background-color: rgba(148,163,184,.15) !important;}
    .stTabs [aria-selected="true"] {color: #22d3ee !important;}

    /* inputs */
    [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"],
    .stTextArea textarea, .stTextInput input {
        background: rgba(148,163,184,.07) !important; border-color: rgba(148,163,184,.25) !important;
        border-radius: 12px !important;
    }
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul {background: #0f172a !important;}
    [data-baseweb="popover"] li:hover {background: rgba(34,211,238,.12) !important;}
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(34,211,238,.05); border: 1.5px dashed rgba(34,211,238,.4); border-radius: 14px;
    }
    [data-testid="stCode"] pre, [data-testid="stCode"] {background: #060a16 !important; border-radius: 12px;}
    [data-testid="stSegmentedControl"] button {border-radius: 10px;}

    /* left-aligned prompt buttons */
    [class*="st-key-tool_example_"] button, [class*="st-key-doc_suggest_"] button {
        justify-content: flex-start; text-align: left;
    }

    /* progress bars + tables */
    .pb-row {display: flex; align-items: center; gap: .55rem; margin: .5rem 0 .2rem;}
    .pb {flex: 1; height: 8px; min-width: 70px; border-radius: 99px; background: rgba(148,163,184,.18); overflow: hidden;}
    .pb-fill {height: 100%; border-radius: 99px;}
    .pb-row span {font-size: .78rem; font-weight: 700; min-width: 3.2rem; text-align: right; font-variant-numeric: tabular-nums;}
    .ag-table td.num, .ag-table th.num {text-align: right; font-variant-numeric: tabular-nums;}

    /* health cards */
    .hc-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: .9rem; margin: .4rem 0 1rem;}
    .hc {
        padding: 1rem 1.1rem; border-radius: 16px; border: 1px solid rgba(148,163,184,.2);
        background: linear-gradient(160deg, rgba(34,211,238,.07), rgba(139,92,246,.07)), rgba(15,23,42,.6);
    }
    .hc.testing {border-style: dashed; opacity: .85;}
    .hc.empty {opacity: .75;}
    .hc-top {display: flex; align-items: center; gap: .65rem;}
    .hc-icon {
        flex: 0 0 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        color: #22d3ee; background: rgba(34,211,238,.12);
    }
    .hc-icon svg {width: 19px; height: 19px;}
    .hc-name {flex: 1; font-weight: 700; font-size: 1rem; font-family: 'Plus Jakarta Sans', sans-serif;}
    .hc-chips {margin: .35rem 0 .2rem;}
    .hc-chips .ao-pill {margin: 0 .3rem .3rem 0; font-size: .7rem;}
    .hc-stats {display: flex; gap: .4rem; margin-top: .7rem; flex-wrap: wrap;}
    .hc-stats > div {flex: 1; min-width: 56px; padding: .45rem .3rem; border-radius: 10px; text-align: center; background: rgba(148,163,184,.07);}
    .hc-stats b {display: block; font-size: .95rem; font-family: 'Plus Jakarta Sans', sans-serif;}
    .hc-stats span {font-size: .62rem; opacity: .6; text-transform: uppercase; letter-spacing: .07em;}
    .hc-note {margin-top: .7rem; font-size: .76rem; opacity: .6;}

    /* trace explorer */
    .tr-card {padding: 1.1rem 1.3rem; border-radius: 16px; margin: .8rem 0; border: 1px solid rgba(148,163,184,.2); background: rgba(15,23,42,.6);}
    .tr-grid {display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;}
    .tr-card .ao-label {opacity: .55; font-size: .68rem; text-transform: uppercase; letter-spacing: .1em; margin-bottom: .25rem;}
    .tr-card .ao-value {font-size: .98rem; font-weight: 600;}
    .tr-quote {margin-top: .35rem; padding: .7rem .9rem; border-left: 3px solid #22d3ee; border-radius: 0 10px 10px 0; background: rgba(34,211,238,.06); font-size: .93rem;}
    .trace-chip {display: inline-block; padding: .15rem .6rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace !important; font-size: .78rem; background: rgba(148,163,184,.12);}
    .callout {
        margin: .5rem 0 .8rem; padding: .65rem .95rem; border-radius: 12px; font-size: .85rem;
        border: 1px solid rgba(245,158,11,.4); background: rgba(245,158,11,.08);
    }
    .mini-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: .5rem; margin: .3rem 0 .7rem;}
    .mini {padding: .55rem .4rem; border-radius: 10px; text-align: center; background: rgba(148,163,184,.07); border: 1px solid rgba(148,163,184,.14);}
    .mini b {display: block; font-size: 1rem; font-family: 'Plus Jakarta Sans', sans-serif;}
    .mini span {font-size: .62rem; opacity: .6; text-transform: uppercase; letter-spacing: .07em;}

    /* workspace alignment fixes */
    .ws-banner {align-items: center;}
    .ws-banner h3 {margin: 0 !important; padding: 0 !important; line-height: 1.25;}
    .ws-banner p {margin: .3rem 0 0 0 !important; line-height: 1.5;}
    .ws-banner .chips {margin-top: .6rem;}
    .steps-panel {padding-left: 1.2rem; border-left: 1px solid rgba(148,163,184,.2);}

    /* info boxes (empty states) stay dark, text readable */
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
        background: rgba(34,211,238,.07) !important;
        border: 1px solid rgba(34,211,238,.25) !important;
        border-radius: 12px !important;
    }
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) * {color: #cbd5e1 !important;}

    /* ================= ROUND 4: perfect alignment ================= */

    /* KPI cards live in one grid row: same width, same height, always aligned */
    .kpi-grid {
        display: grid; gap: .9rem; margin: .3rem 0 1.1rem; align-items: stretch;
        grid-template-columns: repeat(var(--cols, 4), minmax(0, 1fr));
    }
    @media (max-width: 1100px) {.kpi-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
    .kpi-grid .ag-kpi {height: 100%; margin: 0; box-sizing: border-box;}
    .ag-kpi {min-height: 98px; box-sizing: border-box;}
    .ag-kpi > div:last-child {flex: 1; min-width: 0;}
    .ag-kpi-label, .ag-kpi-value {white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
    .ag-kpi-sub {line-height: 1.35; margin-top: .2rem;}

    /* tables: no grid lines, solid header (no see-through text), no wrapped time cells */
    .ag-scroll {overflow: auto; border-radius: 10px;}
    .ag-table {width: 100%; border: none !important; border-collapse: collapse;}
    .ag-table th, .ag-table td {
        border-left: 0 !important; border-right: 0 !important; border-top: 0 !important;
        border-bottom: 1px solid rgba(148,163,184,.14) !important;
    }
    .ag-table th {
        opacity: 1 !important; color: #94a3b8; background: #0f172a; z-index: 2;
        white-space: nowrap; font-size: .72rem; text-transform: uppercase; letter-spacing: .06em;
        padding: .55rem .6rem;
    }
    .ag-table td {padding: .5rem .6rem; vertical-align: middle;}
    .ag-table th:first-child, .ag-table td:first-child {white-space: nowrap;}
    .ag-table tbody tr:hover td {background: rgba(34,211,238,.05);}
    .nw {white-space: nowrap;}

    /* tabs: cyan underline, no red box (works for every Streamlit tab layout) */
    [data-baseweb="tab-highlight"] {background-color: #22d3ee !important; height: 3px !important; border-radius: 3px;}
    [data-baseweb="tab-border"] {background-color: rgba(148,163,184,.15) !important;}
    [data-baseweb="tab"] {
        border: none !important; outline: none !important; box-shadow: none !important;
        background: transparent !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {color: #22d3ee !important;}

    /* segmented control (time window, workspace picker): theme colours instead of brown/red */
    [data-testid="stBaseButton-segmented_control"] {
        background: rgba(148,163,184,.07) !important; border: 1px solid rgba(148,163,184,.22) !important;
        color: #cbd5e1 !important;
    }
    [data-testid="stBaseButton-segmented_control"]:hover {
        background: rgba(34,211,238,.10) !important; color: #22d3ee !important;
    }
    [data-testid="stBaseButton-segmented_controlActive"] {
        background: linear-gradient(135deg, rgba(34,211,238,.28), rgba(139,92,246,.30)) !important;
        border: 1px solid #22d3ee !important; color: #ffffff !important; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# DISPLAY HELPERS
# --------------------------------------------------

DISPLAY_NAMES = {
    "tool_agent": "Tool Agent",
    "document_agent": "Document Agent",
    "research_agent": "Research Agent",
    "legacy": "Legacy",
    "max_rounds_reached": "Max Rounds Reached",
    "total_latency": "Total Latency (s)",
    "average_latency": "Average Latency (s)",
    "latency": "Latency (s)",
    "total_input_tokens": "Input Tokens",
    "total_output_tokens": "Output Tokens",
    "user_input": "User Input",
    "trace_id": "Trace ID",
    "agent_name": "Agent",
    "timestamp": "Timestamp",
    "tool_name": "Tool",
    "event_type": "Event Type",
    "event_name": "Event",
    "input_data": "Input",
    "output_data": "Output",
    "error_type": "Error Type",
    "error_message": "Error Message",
    "successful_calls": "Successful Calls",
    "failed_calls": "Failed Calls",
    "total_calls": "Total Calls",
    "success_rate": "Success Rate (%)",
    "failure_rate": "Failure Rate (%)",
}

SUCCESS_STATES = {"success", "succeeded", "completed"}
FAILED_STATES = {"error", "failed", "failure"}
RUNNING_STATES = {"running", "in_progress", "pending"}
WARN_STATES = {
    "needs_review", "max_rounds_reached", "insufficient_sources", "partial_success",
    "retrieval_miss", "generation_miss", "wrong_answer",
}

# A run that stays "running" longer than this was stopped mid-flight
# (for example with Ctrl+C) and is shown as "Interrupted" instead of "Running".
INCOMPLETE_STATUSES = {"running", "interrupted", "not_judged"}
STALE_RUNNING_HOURS = 1

KIND_COLORS = {
    "success": "#22c55e",
    "error": "#ef4444",
    "running": "#f59e0b",
    "warn": "#f97316",
    "neutral": "#94a3b8",
}

CYAN, PURPLE, AMBER, RED = "#22d3ee", "#8b5cf6", "#f59e0b", "#ef4444"

# One fixed colour per status so that no two statuses look alike in a chart.
STATUS_COLORS = {
    "success": "#22c55e",
    "failed": "#ef4444",
    "error": "#be123c",
    "interrupted": "#94a3b8",
    "running": "#f59e0b",
    "partial_success": "#eab308",
    "needs_review": "#f97316",
    "insufficient_sources": "#a855f7",
    "max_rounds_reached": "#ec4899",
    "max_tool_calls_reached": "#14b8a6",
    "retrieval_miss": "#3b82f6",
    "generation_miss": "#6366f1",
    "wrong_answer": "#d946ef",
    "not_judged": "#64748b",
}

# One fixed colour per agent, reused in every chart so that an agent
# always renders with the same colour.
AGENT_COLORS = {"tool_agent": CYAN, "research_agent": PURPLE, "document_agent": "#f472b6"}
EXTRA_AGENT_COLORS = [AMBER, "#60a5fa", "#a3e635"]

# Icons used across every page. All are inline SVG paths so the
# dashboard renders identically regardless of font availability.
ICONS = {
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "bot": '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "layout-dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "message-square": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "bar-chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "check-circle": '<path d="M21.801 10A10 10 0 1 1 17 3.335"/><path d="m9 11 3 3L22 4"/>',
    "x-circle": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "timer": '<line x1="10" x2="14" y1="2" y2="2"/><line x1="12" x2="15" y1="14" y2="11"/><circle cx="12" cy="14" r="8"/>',
    "list": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    "coins": '<circle cx="8" cy="8" r="6"/><path d="M18.09 10.37A6 6 0 1 1 10.34 18"/><path d="M7 6h1v4"/><path d="m16.71 13.88.7.71-2.82 2.82"/>',
}


def pretty_name(value):
    text = str(value)
    return DISPLAY_NAMES.get(text, text.replace("_", " ").title())


def pretty_status(value):
    """Human-readable status label. None renders as 'Unknown' instead of 'None'."""
    if value is None:
        return "Unknown"
    return pretty_name(value)


def agent_color_map(agent_ids):
    """Return {pretty agent name: colour} for plotly's color_discrete_map."""
    mapping, extra = {}, iter(EXTRA_AGENT_COLORS)
    for agent_id in sorted({a for a in agent_ids if a is not None}):
        mapping[pretty_name(agent_id)] = AGENT_COLORS.get(agent_id) or next(extra, "#94a3b8")
    return mapping


def status_kind(value):
    text = str(value if value is not None else "unknown").strip().lower()
    if text in SUCCESS_STATES:
        return "success"
    if text in FAILED_STATES:
        return "error"
    if text in RUNNING_STATES:
        return "running"
    if text in WARN_STATES:
        return "warn"
    return "neutral"


def pill(value):
    return (
        f'<span class="ao-pill {status_kind(value)}">'
        f"{html.escape(pretty_status(value))}</span>"
    )


def status_color_map(values):
    return {
        pretty_status(value): STATUS_COLORS.get(str(value).strip().lower(), KIND_COLORS[status_kind(value)])
        for value in values
    }


def format_timestamp(value, include_seconds=True, compact=False):
    if value is None or value == "":
        return "N/A"

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return str(value)

    if compact:   # Short form for narrow table columns, e.g. "19 Nov, 05:20:45 PM"
        fmt = "%d %b, %I:%M:%S %p" if include_seconds else "%d %b, %I:%M %p"
    else:
        fmt = "%d %b %Y, %I:%M:%S %p" if include_seconds else "%d %b %Y, %I:%M %p"
    return timestamp.tz_convert(DISPLAY_TIMEZONE).strftime(fmt)


def short_id(value, length=8):
    if not value:
        return "N/A"
    return f"{str(value)[:length]}..."


def display_dataframe(df, columns=None, rename=True, column_config=None):
    if df is None or df.empty:
        st.info("No data available.")
        return

    shown = df.copy()
    if columns:
        available = [column for column in columns if column in shown.columns]
        shown = shown[available]

    if "timestamp" in shown.columns:
        shown["timestamp"] = shown["timestamp"].apply(format_timestamp)

    if "agent_name" in shown.columns:
        shown["agent_name"] = shown["agent_name"].apply(pretty_name)

    if "status" in shown.columns:
        shown["status"] = shown["status"].apply(pretty_status)

    if "trace_id" in shown.columns:
        shown["trace_id"] = shown["trace_id"].apply(short_id)

    if rename:
        shown = shown.rename(columns={col: pretty_name(col) for col in shown.columns})

    st.dataframe(
        shown,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )


def api_get(path, params=None, timeout=15):
    response = requests.get(
        f"{API_URL}{path}",
        params=params,
        timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def api_online():
    try:
        requests.get(f"{API_URL}/", timeout=3).raise_for_status()
        return True
    except requests.RequestException:
        return False


def section(title):
    st.markdown(f'<div class="ao-section">{html.escape(title)}</div>', unsafe_allow_html=True)


def kpi_grid(cards):
    """Render KPI cards in a single CSS grid so that width and height always match."""
    st.markdown(
        f'<div class="kpi-grid" style="--cols:{len(cards)}">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """(label, value) pairs shown with the same card style as every other KPI."""
    icons = ["activity", "gauge", "check-circle", "layers"]
    kpi_grid([
        _kpi(icons[i % len(icons)], label, value, "", "", "purple" if i % 2 else "cyan")
        for i, (label, value) in enumerate(items)
    ])


def panel(height=None):
    """Bordered card. A fixed height keeps the cards of one row perfectly aligned."""
    try:
        return st.container(border=True, height=height) if height else st.container(border=True)
    except TypeError:   # Older Streamlit versions do not accept height
        return st.container(border=True)


def style_fig(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        hovermode="closest",
    )
    return fig


def show_chart(fig, key=None, modebar=True):
    config = {"displaylogo": False, "responsive": True}
    if not modebar:
        config["displayModeBar"] = False
    st.plotly_chart(fig, width="stretch", config=config, key=key)


def json_or_text(value, empty_text):
    if isinstance(value, (dict, list)):
        st.json(value)
    else:
        st.write(value or empty_text)


# --------------------------------------------------
# COMPACT FORMATTERS & TABLE HELPERS
# These helpers are used by multiple sections (including render_time_window),
# so they are defined here rather than next to the section that first uses them.
# --------------------------------------------------

def fmt_compact(value):
    """Compact number formatter: 1500 -> '1.5K', 2_000_000 -> '2.0M'."""
    number = float(value or 0)
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if number >= divisor:
            return f"{number / divisor:.1f}{suffix}"
    return f"{int(number)}"


def rate_color(rate):
    return "#22c55e" if rate >= 95 else ("#f59e0b" if rate >= 80 else "#ef4444")


def rate_cell(rate):
    """Inline progress bar + percentage, used in tables and health cards."""
    rate = max(0.0, min(100.0, float(rate or 0)))
    return (
        '<div class="pb-row"><div class="pb">'
        f'<div class="pb-fill" style="width:{rate:.1f}%;background:{rate_color(rate)}"></div>'
        f"</div><span>{rate:.1f}%</span></div>"
    )


def health_badge(rate, testing=False):
    if testing:
        return '<span class="ao-pill neutral">Testing</span>'
    if rate >= 95:
        return '<span class="ao-pill success">Healthy</span>'
    if rate >= 80:
        return '<span class="ao-pill warn">Needs attention</span>'
    return '<span class="ao-pill error">Unreliable</span>'


def stat_block(items):
    return '<div class="hc-stats">' + "".join(
        f"<div><b>{html.escape(str(value))}</b><span>{html.escape(label)}</span></div>"
        for label, value in items
    ) + "</div>"


def mini_stats(pairs):
    return '<div class="mini-stats">' + "".join(
        f'<div class="mini"><b>{html.escape(str(value))}</b><span>{html.escape(label)}</span></div>'
        for label, value in pairs
    ) + "</div>"


# --------------------------------------------------
# DATA PREPARATION
# Derived columns are computed here once so that every consumer
# (overview, time window, agent figures) reads from a stable contract.
# --------------------------------------------------

def prepare_runs(df):
    """Attach all derived columns used by the dashboard.

    Adds: ts, kind, latency_done, agent_label, timestamp_sort, status_label.
    Idempotent: safe to call on an already prepared frame.
    """
    if df is None or df.empty:
        return df

    runs = df.copy()
    runs["ts"] = pd.to_datetime(runs["timestamp"], utc=True, errors="coerce")
    runs["kind"] = runs["status"].apply(status_kind)
    runs["agent_label"] = runs["agent_name"].apply(pretty_name)
    runs["status_label"] = runs["status"].apply(pretty_status)
    runs["timestamp_sort"] = runs["ts"]

    latency = pd.to_numeric(runs["total_latency"], errors="coerce")
    runs["latency_done"] = latency.where(~runs["status"].isin(INCOMPLETE_STATUSES))
    return runs


def mark_interrupted(df):
    """Flag runs that stayed 'running' for over an hour as 'interrupted'."""
    if df is None or df.empty:
        return df
    df = df.copy()
    stamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=STALE_RUNNING_HOURS)
    df.loc[(df["status"] == "running") & (stamps < cutoff), "status"] = "interrupted"
    return df


DOC_OUTCOME_SLUGS = {
    "Retrieval miss": "retrieval_miss",
    "Generation miss": "generation_miss",
    "Wrong answer (no answer existed)": "wrong_answer",
}


def apply_document_quality(stats, doc):
    """Split the document agent's 'success' runs by answer quality.

    A run that finished without an error but returned a wrong answer is not a
    success. `stats` comes from /agent-stats, `doc` from /labels/document-outcomes.
    """
    if not stats or not stats.get("groups") or not doc or not doc.get("success_total"):
        return stats

    groups = [dict(g) for g in stats["groups"]]
    found = False
    for g in groups:
        if g.get("agent_name") == "document_agent" and g.get("status") == "success":
            # Replace the inflated "success" count with the number of answers
            # that were actually judged correct. Latency and token counts are
            # intentionally left on the original row (they are still valid).
            g["runs"] = int(doc.get("correct", 0))
            found = True
    if not found:
        return stats

    def extra(status, runs):
        return {"agent_name": "document_agent", "status": status, "runs": int(runs),
                "latency_sum": 0.0, "latency_count": 0, "input_tokens": 0, "output_tokens": 0}

    for outcome, count in (doc.get("outcomes") or {}).items():
        if count:
            groups.append(extra(DOC_OUTCOME_SLUGS.get(outcome, "wrong_answer"), count))
    if doc.get("not_judged"):
        groups.append(extra("not_judged", doc["not_judged"]))

    return {**stats, "groups": groups}


def apply_silent_failures(df, failures_data):
    """Show a wrong answer with its real outcome instead of 'success'."""
    if df is None or df.empty or not failures_data:
        return df
    outcome = {
        item["trace_id"]: DOC_OUTCOME_SLUGS.get(item.get("cause"), "wrong_answer")
        for item in failures_data.get("failures", [])
        if item.get("kind") == "Silent failure"
    }
    if not outcome:
        return df
    df = df.copy()
    mask = df["trace_id"].isin(outcome) & (df["status"] == "success")
    df.loc[mask, "status"] = df.loc[mask, "trace_id"].map(outcome)
    return df


def stats_groups(agent_stats, df_runs):
    """Counts per agent and status. Uses /agent-stats (all runs in the
    database); falls back to the loaded runs if the endpoint is missing."""
    if agent_stats and agent_stats.get("groups"):
        return pd.DataFrame(agent_stats["groups"]), True

    columns = ["agent_name", "status", "runs", "latency_sum",
               "latency_count", "input_tokens", "output_tokens"]
    if df_runs is None or df_runs.empty:
        return pd.DataFrame(columns=columns), False

    d = df_runs.copy()
    d["lat"] = pd.to_numeric(d["total_latency"], errors="coerce")
    d["in_t"] = pd.to_numeric(d["total_input_tokens"], errors="coerce").fillna(0)
    d["out_t"] = pd.to_numeric(d["total_output_tokens"], errors="coerce").fillna(0)
    groups = (
        d.groupby(["agent_name", "status"])
        .agg(runs=("trace_id", "count"), latency_sum=("lat", "sum"),
             latency_count=("lat", "count"), input_tokens=("in_t", "sum"),
             output_tokens=("out_t", "sum"))
        .reset_index()
    )
    return groups, False


def run_totals(groups, from_database):
    """Overall and per-agent numbers. Success rate counts finished runs only
    (a run that never finished is neither a success nor a failure)."""
    if groups is None or groups.empty:
        return None

    g = groups.copy()
    g["kind"] = g["status"].apply(status_kind)
    g["incomplete"] = g["status"].isin(INCOMPLETE_STATUSES)

    def count(mask):
        return int(g.loc[mask, "runs"].sum())

    total = int(g["runs"].sum())
    incomplete = count(g["incomplete"])
    finished = total - incomplete
    ok = count(g["kind"] == "success")

    per_agent = []
    for name, grp in g.groupby("agent_name", dropna=False):
        done = grp[~grp["incomplete"]]
        finished_g = int(done["runs"].sum())
        ok_g = int(grp.loc[grp["kind"] == "success", "runs"].sum())
        lat_n = int(done["latency_count"].sum())
        per_agent.append({
            "agent_name": name,
            "total_runs": int(grp["runs"].sum()),
            "successful_runs": ok_g,
            "failed_runs": int(grp.loc[grp["kind"] == "error", "runs"].sum()),
            "average_latency": float(done["latency_sum"].sum()) / lat_n if lat_n else float("nan"),
            "input_tokens": int(grp["input_tokens"].sum()),
            "output_tokens": int(grp["output_tokens"].sum()),
            "success_rate": ok_g / finished_g * 100 if finished_g else 0.0,
        })

    return {
        "from_database": from_database,
        "total": total,
        "ok": ok,
        "bad": count(g["kind"] == "error"),
        "warn": count(g["kind"] == "warn"),
        "running": count(g["status"] == "running"),
        "interrupted": count(g["status"] == "interrupted"),
        "rate": ok / finished * 100 if finished else 0.0,
        "tokens": int(g["input_tokens"].sum() + g["output_tokens"].sum()),
        "agents": int(g["agent_name"].nunique()),
        "per_agent": pd.DataFrame(per_agent),
        "groups": g,
    }


# --------------------------------------------------
# ICON HELPERS & PAGE CHROME
# --------------------------------------------------

def svg_icon(name):
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
        f"{ICONS.get(name, '')}</svg>"
    )


def page_hero(icon, eyebrow, title, subtitle, badge=None):
    badge_html = (
        f'<div class="ao-live"><span class="ao-live-dot"></span>{html.escape(badge)}</div>'
        if badge else ""
    )
    st.markdown(
        f"""
        <div class="ao-hero">
            <div class="ao-hero-left">
                <div class="ao-hero-icon">{svg_icon(icon)}</div>
                <div>
                    <div class="ao-eyebrow">{html.escape(eyebrow)}</div>
                    <h1>{html.escape(title)}</h1>
                    <p>{html.escape(subtitle)}</p>
                </div>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def heading(icon, text):
    st.markdown(
        f'<div class="ag-head">{svg_icon(icon)}<span>{html.escape(text)}</span></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# OVERVIEW DASHBOARD
# --------------------------------------------------

def _kpi(icon, label, value, tone="", sub="", accent="cyan"):
    sub_html = f'<div class="ag-kpi-sub">{html.escape(str(sub))}</div>' if sub else ""
    return (
        f'<div class="ag-kpi {tone}"><div class="ag-kpi-icon {accent}">{svg_icon(icon)}</div>'
        f'<div><div class="ag-kpi-label">{html.escape(str(label))}</div>'
        f'<div class="ag-kpi-value">{html.escape(str(value))}</div>{sub_html}</div></div>'
    )


def _table(headers, rows, max_h=260):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return (
        f'<div class="ag-scroll" style="max-height:{max_h}px">'
        f'<table class="ag-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _title(text, sub=None):
    st.markdown(f'<div class="ag-title">{html.escape(text)}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="ag-sub">{html.escape(sub)}</div>', unsafe_allow_html=True)


def _dark(fig, height):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 34, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1,
                "font": {"color": "#e2e8f0"}, "title": {"font": {"color": "#e2e8f0"}}},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.15)")
    return fig


def _gauge(value, title):
    color = CYAN if value < 60 else AMBER if value < 85 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100],
                     "tickfont": {"size": 10}, "tickcolor": "rgba(148,163,184,.5)"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(148,163,184,.15)",
            "borderwidth": 0,
        },
    ))
    fig.update_layout(
        height=200, margin={"l": 40, "r": 40, "t": 52, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", template="plotly_dark",
    )
    return fig


def _level(kind):
    return {
        "success": ("INF", "inf"),
        "warn": ("WRN", "wrn"),
        "error": ("ERR", "err"),
        "running": ("RUN", "run"),
    }.get(kind, ("INF", "inf"))


def _short(text, n=60):
    text = str(text or "").replace("\n", " ")
    return html.escape(text if len(text) <= n else text[: n - 1] + "…")


def _clean(value, default):
    return value if isinstance(value, str) and value.strip() else default


KNOWN_AGENTS = ["tool_agent", "research_agent", "document_agent"]
WINDOW_GROUP_COLUMNS = ["agent_name", "status", "runs", "latency_sum",
                        "latency_count", "input_tokens", "output_tokens"]


@st.cache_data(ttl=20, show_spinner=False)
def fetch_window_stats(hours):
    """Counts per agent and status for the last N hours, from the whole database.

    Returns the stats payload (or None when the API does not support the
    'hours' option yet). When answer-quality labels are unavailable, the
    document agent's success count stays inflated; a warning flag is added
    to the returned payload so the UI can surface the gap.
    """
    try:
        data = api_get("/agent-stats", params={"hours": hours})
    except requests.RequestException:
        return None
    if data.get("hours") != hours:
        return None

    try:
        labels = api_get("/labels/document-outcomes", params={"hours": hours})
        data = apply_document_quality(data, labels)
    except requests.RequestException:
        data["quality_labels_unavailable"] = True
    return data


@st.cache_data(ttl=20, show_spinner=False)
def fetch_recent_errors(agent_names):
    """Most recent failed runs (up to 200 per agent) fetched in parallel."""
    def _fetch(name):
        try:
            return api_get(
                "/agent-runs",
                params={"agent_name": name, "status": "error", "limit": 200, "offset": 0},
            ).get("runs", [])
        except requests.RequestException:
            return []

    rows = []
    with futures.ThreadPoolExecutor(max_workers=min(8, len(agent_names) or 1)) as pool:
        for chunk in pool.map(_fetch, agent_names):
            rows.extend(chunk)
    return rows


def window_agent_rows(groups, agent_ids):
    """One summary row per agent for the selected window (agents with no runs included)."""
    # Guard against an empty frame that never received any columns.
    if groups is None or groups.empty or "agent_name" not in groups.columns:
        return [{"agent_id": agent_id, "total": 0} for agent_id in agent_ids]

    rows = []
    for agent_id in agent_ids:
        g = groups[groups["agent_name"] == agent_id]
        if g.empty:
            rows.append({"agent_id": agent_id, "total": 0})
            continue
        kinds = g["status"].apply(status_kind)
        incomplete = g["status"].isin(INCOMPLETE_STATUSES)
        total = int(g["runs"].sum())
        ok = int(g.loc[kinds == "success", "runs"].sum())
        bad = int(g.loc[kinds == "error", "runs"].sum())
        unfinished = int(g.loc[incomplete, "runs"].sum())
        finished = total - unfinished
        done = g[~incomplete]
        lat_n = float(done["latency_count"].sum())
        rows.append({
            "agent_id": agent_id,
            "total": total,
            "ok": ok,
            "bad": bad,
            "other": total - ok - bad,
            "finished": finished,
            "rate": ok / finished * 100 if finished else None,
            "latency_sum": float(done["latency_sum"].sum()),
            "latency_n": lat_n,
            "avg_latency": float(done["latency_sum"].sum()) / lat_n if lat_n else None,
            "tokens": int(g["input_tokens"].sum() + g["output_tokens"].sum()),
        })
    return rows


def render_time_window(runs, df_tool_events):
    heading("clock", "Time Window Analysis")
    st.caption(
        "The cards at the top cover the entire recorded period. "
        "Pick an interval here to see what every agent did inside it."
    )

    windows = {
        "Last hour": 1,
        "Last 6 hours": 6,
        "Last 24 hours": 24,
        "Last 7 days": 168,
        "Last 30 days": 720,
    }
    choice = st.segmented_control(
        "Time window",
        list(windows.keys()),
        default="Last 24 hours",
        key="ov_time_window",
        label_visibility="collapsed",
    ) or "Last 24 hours"
    hours = windows[choice]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)

    # ---------- run numbers for the window ----------
    w_runs = runs[runs["ts"] >= cutoff]
    agent_ids = KNOWN_AGENTS + sorted(
        a for a in set(runs["agent_name"].dropna().unique().tolist()) if a not in KNOWN_AGENTS
    )

    stats = fetch_window_stats(hours)
    if stats is not None:
        groups = pd.DataFrame(stats.get("groups", []), columns=WINDOW_GROUP_COLUMNS)
        from_db = True
        agent_ids += sorted(set(groups["agent_name"].dropna()) - set(agent_ids))
    else:
        groups, _ = stats_groups(None, w_runs)
        from_db = False

    agent_rows = window_agent_rows(groups, agent_ids)
    active = [r for r in agent_rows if r["total"] > 0]

    w_total = sum(r["total"] for r in active)
    w_ok = sum(r["ok"] for r in active)
    w_bad = sum(r["bad"] for r in active)
    w_finished = sum(r["finished"] for r in active)
    lat_n = sum(r["latency_n"] for r in active)
    w_avg = sum(r["latency_sum"] for r in active) / lat_n if lat_n else None
    w_rate = w_ok / w_finished * 100 if w_finished else None

    # ---------- tool calls for the window ----------
    tool_errors, w_tool_calls = [], 0
    if df_tool_events is not None and not df_tool_events.empty:
        te = df_tool_events.copy()
        te["ts"] = pd.to_datetime(te["timestamp"], utc=True, errors="coerce")
        te = te[te["ts"] >= cutoff]
        w_tool_calls = len(te)
        tool_errors = te[te["status"].apply(status_kind) == "error"].to_dict("records")

    kpi_grid([
        _kpi("activity", "Runs in window", f"{w_total:,}", "",
             f"{w_ok:,} succeeded · {len(active)} of {len(agent_ids)} agents active"),
        _kpi("x-circle", "Failed Runs", f"{w_bad:,}", "bad" if w_bad else "ok", "across all agents"),
        _kpi("gauge", "Run Success Rate", f"{w_rate:.1f}%" if w_rate is not None else "N/A",
             "" if w_rate is None else ("ok" if w_rate >= 90 else ("warn" if w_rate >= 70 else "bad")),
             "of finished runs"),
        _kpi("wrench", "Failed Tool Calls", f"{len(tool_errors):,}", "bad" if tool_errors else "ok",
             f"of {w_tool_calls:,} recent tool calls"),
        _kpi("timer", "Avg. Run Latency", f"{w_avg:.2f}s" if w_avg is not None else "N/A", "",
             "completed runs only", "purple"),
    ])

    # ---------- per-agent breakdown ----------
    st.markdown(
        f'<div class="ag-title" style="margin-top:.9rem">Every agent in the '
        f'{html.escape(choice.lower())}</div>',
        unsafe_allow_html=True,
    )
    colors = agent_color_map(agent_ids)
    table_rows = []
    for row in agent_rows:
        label = pretty_name(row["agent_id"])
        name_cell = (
            f'<span class="dot" style="background:{colors.get(label, "#94a3b8")}"></span>'
            f"{html.escape(label)}"
        )
        if row["total"] == 0:
            table_rows.append([name_cell, "0", "—", "—", "—",
                               '<span style="opacity:.55">No runs in this window</span>', "—", "—"])
            continue
        table_rows.append([
            name_cell,
            f'{row["total"]:,}',
            f'{row["ok"]:,}',
            f'<span class="st-bad">{row["bad"]:,}</span>' if row["bad"] else "0",
            f'{row["other"]:,}',
            rate_cell(row["rate"]) if row["rate"] is not None else "—",
            f'{row["avg_latency"]:.2f}s' if row["avg_latency"] is not None else "—",
            fmt_compact(row["tokens"]),
        ])
    st.markdown(
        _table(["Agent", "Runs", "Successful", "Failed", "Other", "Success Rate", "Avg Latency", "Tokens"],
               table_rows, 260),
        unsafe_allow_html=True,
    )

    # ---------- errors ----------
    entries = []
    for r in fetch_recent_errors(tuple(agent_ids)):
        ts = pd.to_datetime(r.get("timestamp"), utc=True, errors="coerce")
        if pd.notna(ts) and ts >= cutoff:
            label = pretty_name(r.get("agent_name"))
            entries.append((ts, "Agent run", label, _clean(r.get("error_type"), "Unclassified"),
                            _clean(r.get("error_message"), "No message recorded"), label))
    for e in tool_errors:
        entries.append((e["ts"], "Tool call", pretty_name(e.get("tool_name")),
                        _clean(e.get("error_type"), "Unclassified"),
                        _clean(e.get("error_message"), "No message recorded"), "Tool calls"))
    entries.sort(key=lambda x: x[0] if pd.notna(x[0]) else pd.Timestamp(0, tz="UTC"), reverse=True)

    if w_total == 0 and w_tool_calls == 0:
        st.info(f"No activity was recorded in the {choice.lower()}.")
    elif not entries and not w_bad:
        st.success(f"No errors in the {choice.lower()} — every run and tool call completed cleanly.")
    elif not entries:
        st.warning(
            f"{w_bad:,} failed run(s) were counted in the {choice.lower()}, but their details "
            "are not in the most recent records."
        )
    else:
        group_counts = collections.Counter(e[5] for e in entries)
        st.markdown(
            f'<div class="ag-title" style="margin-top:.9rem">Errors in the '
            f'{html.escape(choice.lower())}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "".join(
                f'<span class="ao-pill error">{html.escape(group)} · {count:,}</span> '
                for group, count in group_counts.most_common()
            ),
            unsafe_allow_html=True,
        )
        source_choice = st.selectbox(
            "Show errors from",
            ["All sources"] + [group for group, _ in group_counts.most_common()],
            key="ov_error_source",
        )
        shown = [e for e in entries if source_choice == "All sources" or e[5] == source_choice]
        rows = [
            [html.escape(format_timestamp(ts, compact=True)), src, html.escape(name),
             html.escape(etype), _short(msg, 120)]
            for ts, src, name, etype, msg, _group in shown[:60]
        ]
        st.markdown(_table(["Time", "Source", "Name", "Error Type", "Message"], rows, 340),
                    unsafe_allow_html=True)
        st.caption(
            f"Showing {min(60, len(shown))} of {len(shown)} errors, newest first. "
            "Error details come from the 200 most recent failed runs of each agent "
            "and the latest 200 tool calls."
        )

    if stats is not None and stats.get("quality_labels_unavailable"):
        st.warning(
            "Answer-quality labels are currently unavailable, so the Document Agent's "
            "success count may include wrong answers. Retry once the labels endpoint "
            "is reachable."
        )

    if not from_db:
        st.caption(
            "Run numbers above use the latest 200 runs of each agent. Update the API's "
            "/agent-stats endpoint (hours option) to count every run in the window."
        )


def ago_text(ts):
    """'12 s ago', '5 min ago', '21 h ago' for a timestamp."""
    if ts is None or pd.isna(ts):
        return "N/A"
    seconds = max(0, int((pd.Timestamp.now(tz="UTC") - ts).total_seconds()))
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86400} d ago"


def smooth_curve(x_times, y_values, points_per_gap=None):
    """Smooth line through the points that never dips below the lowest point or
    above the highest neighbour (monotone cubic, Fritsch-Carlson).

    A plain spline would dip below zero runs, which is misleading. When
    points_per_gap is None an adaptive value is chosen so that dense series
    do not produce thousands of interpolated points.
    """
    x = np.array([int(t.value) / 1e9 for t in x_times], dtype=float)   # seconds
    y = np.array(y_values, dtype=float)
    n = len(x)
    if n < 3:
        return list(x_times), list(y)

    if points_per_gap is None:
        points_per_gap = max(4, min(14, 200 // max(1, n)))

    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(n)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else 2 * d[i - 1] * d[i] / (d[i - 1] + d[i])
    for i in range(n - 1):
        if d[i] == 0:
            m[i] = m[i + 1] = 0.0
        else:
            a, b = m[i] / d[i], m[i + 1] / d[i]
            r = a * a + b * b
            if r > 9:
                t = 3 / np.sqrt(r)
                m[i], m[i + 1] = t * a * d[i], t * b * d[i]
    xs, ys = [], []
    for i in range(n - 1):
        u = np.linspace(0, 1, points_per_gap, endpoint=False)
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        xs += list(x[i] + u * h[i])
        ys += list(h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1])
    xs.append(x[-1]); ys.append(y[-1])
    ys = np.maximum(ys, 0.0)
    tz = x_times[0].tz
    times = [
        pd.Timestamp(int(v * 1e9), tz="UTC").tz_convert(tz) if tz is not None
        else pd.Timestamp(int(v * 1e9))
        for v in xs
    ]
    return times, list(ys)


def quality_numbers(quality):
    """(answers judged, correct, silent failures, accuracy %) for the document agent."""
    doc = (quality or {}).get("document") or {}
    total, correct = int(doc.get("total") or 0), int(doc.get("correct") or 0)
    silent = int(doc.get("silent_failures") or 0)
    return total, correct, silent, (correct / total * 100 if total else None)


def render_failures_section(quality, failures_data, totals=None):
    """Answer quality of the document agent + failures of every agent."""
    heading("triangle-alert", "Failures & Answer Quality")

    total, correct, silent, accuracy = quality_numbers(quality)
    items = (failures_data or {}).get("failures", [])

    left, right = st.columns(2, gap="medium")

    with left, panel(430):
        _title("Success rate by agent",
               "Share of finished runs that ended as a clean success. For the Document "
               "Agent a run only counts when its answer was checked and correct.")
        per_agent = (totals or {}).get("per_agent")
        if per_agent is None or per_agent.empty:
            st.info("No runs yet.")
        else:
            pa = per_agent.copy()
            all_ok, all_total = int(pa["successful_runs"].sum()), int(pa["total_runs"].sum())
            st.markdown(
                mini_stats([
                    ("Runs", f"{all_total:,}"),
                    ("Clean success", f"{(totals or {}).get('rate', 0):.1f}%"),
                    ("Not clean", f"{all_total - all_ok:,}"),
                ]),
                unsafe_allow_html=True,
            )
            pa["agent"] = pa["agent_name"].apply(pretty_name)
            pa["rate"] = pa["success_rate"].round(1)
            pa["label"] = pa.apply(
                lambda r: f"{r['rate']}%  ({int(r['successful_runs']):,}/{int(r['total_runs']):,})",
                axis=1,
            )
            pa = pa.sort_values("rate")
            fig = px.bar(
                pa, x="rate", y="agent", orientation="h", text="label",
                labels={"rate": "Clean success (%)", "agent": ""},
                color="rate", color_continuous_scale=[RED, AMBER, "#22c55e"], range_color=[0, 100],
            )
            fig.update_traces(textposition="outside", cliponaxis=False, textfont={"color": "#e2e8f0"})
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(range=[0, 135])
            show_chart(_dark(fig, 230), key="success_by_agent")
            if total:
                st.caption(f"Document Agent: {total:,} answers checked, {accuracy:.1f}% correct, "
                           f"{total - correct:,} wrong.")

    with right, panel(430):
        _title("Failures by agent",
               "Runs that did not end as a clean success, split by status.")
        groups = (totals or {}).get("groups")
        bad = None
        if groups is not None and not groups.empty:
            bad = groups[groups["status"].apply(status_kind) != "success"]
            bad = bad[~bad["status"].isin(["running", "interrupted", "not_judged"])]
        if bad is None or bad.empty:
            st.success("No failed or flagged runs.")
        else:
            bad = bad.assign(
                agent=bad["agent_name"].apply(pretty_name),
                state=bad["status"].apply(pretty_name),
            )
            colors = {pretty_name(k): v for k, v in STATUS_COLORS.items()}
            fig = px.bar(
                bad, x="runs", y="agent", color="state", orientation="h",
                color_discrete_map=colors,
                labels={"runs": "Runs", "agent": "", "state": "Status"},
            )
            fig.update_layout(barmode="stack", legend_title_text="", legend={"font": {"color": "#e2e8f0"}})
            show_chart(_dark(fig, 340), key="fail_by_agent")

    with panel():
        _title("Recent failures",
               "Failed runs, flagged runs and wrong answers. A wrong answer is a silent "
               "failure: the run finished, the answer was not correct.")
        if failures_data is None:
            st.caption("Failure details need the updated API (labels route).")
        elif not items:
            st.success("No failures recorded.")
        else:
            names = ["All agents"] + sorted({pretty_name(i.get("agent_name")) for i in items})
            chosen = st.radio("Agent", names, horizontal=True, key="fail_agent",
                              label_visibility="collapsed")
            shown = [i for i in items if chosen == "All agents"
                     or pretty_name(i.get("agent_name")) == chosen]
            counts = collections.Counter(i.get("kind") for i in shown)
            st.markdown(
                f'<span class="ao-pill error">Run failed · {counts.get("Run failed", 0):,}</span> '
                f'<span class="ao-pill warn">Flagged · {counts.get("Flagged", 0):,}</span> '
                f'<span class="ao-pill warn">Silent failure · {counts.get("Silent failure", 0):,}</span>',
                unsafe_allow_html=True,
            )
            rows = []
            for item in shown[:60]:
                kind = item.get("kind", "")
                cls = "error" if kind == "Run failed" else "warn"
                rows.append([
                    html.escape(format_timestamp(item.get("timestamp"), include_seconds=False, compact=True)),
                    html.escape(pretty_name(item.get("agent_name"))),
                    f'<span class="ao-pill {cls}" style="white-space:nowrap">{html.escape(kind)}</span>',
                    html.escape(str(item.get("cause") or "")),
                    _short(item.get("error_message") or item.get("user_input"), 90),
                ])
            st.markdown(_table(["Time", "Agent", "Type", "Cause", "Detail"], rows, 340),
                        unsafe_allow_html=True)


ROW1_H = 470   # Runs chart | Latency chart
ROW2_H = 300   # CPU gauge | Memory gauge | Active agents
ROW3_H = 440   # Activity logs | Alerts


def render_overview(df_runs, df_tool_events, tool_analytics=None, totals=None,
                    quality=None, failures_data=None):
    if df_runs is None or df_runs.empty:
        st.info("No agent runs yet. Run an agent from the Agent Lab and it will appear here automatically.")
        return

    # Prepare derived columns once. Stale "running" rows are relabelled
    # as "interrupted" before any count is computed.
    runs = prepare_runs(mark_interrupted(df_runs))
    now = pd.Timestamp.now(tz="UTC")

    # Numbers of the loaded runs (used by the charts and logs below).
    loaded_total = len(runs)
    loaded_ok = int((runs["kind"] == "success").sum())
    loaded_bad = int((runs["kind"] == "error").sum())

    # KPI cards use the true totals of the whole database when available.
    if totals:
        total, ok, bad = totals["total"], totals["ok"], totals["bad"]
        running, interrupted = totals["running"], totals["interrupted"]
        rate, agents_seen, tokens = totals["rate"], totals["agents"], totals["tokens"]
    else:
        total, ok, bad = loaded_total, loaded_ok, loaded_bad
        running = int((runs["kind"] == "running").sum())
        interrupted = int((runs["status"] == "interrupted").sum())
        finished = total - running - interrupted
        rate = ok / finished * 100 if finished else 0.0
        agents_seen = runs["agent_name"].nunique()
        tokens = int(
            pd.to_numeric(runs["total_input_tokens"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(runs["total_output_tokens"], errors="coerce").fillna(0).sum()
        )

    if rate >= 90:
        health, tone = "Optimal", "ok"
    elif rate >= 70:
        health, tone = "Degraded", "warn"
    else:
        health, tone = "Critical", "bad"

    failed_tools = int((tool_analytics or {}).get("failed_calls", 0) or 0)
    _q_total, _q_correct, silent_fail, accuracy = quality_numbers(quality)
    total_errors = bad + failed_tools + silent_fail
    flagged_total = totals["warn"] if totals else 0

    last_ts = runs["ts"].max() if not runs.empty else None
    active_now = runs[runs["ts"] >= now - pd.Timedelta(hours=1)]["agent_name"].nunique()

    # ---------------- KPI row (one grid: equal width, equal height) ----------------
    kpi_grid([
        _kpi("shield-check", "System Health", health, tone,
             f"{rate:.1f}% of finished runs succeeded"),
        _kpi("bot", "Active Agents", f"{active_now} / 3", "",
             (f"{running} running now · " if running else "") + f"Last active {ago_text(last_ts)}",
             "purple"),
        _kpi("activity", "Total Runs", f"{total:,}", "",
             f"{ok:,} succeeded · {bad:,} failed"
             + (f" · {flagged_total:,} flagged" if flagged_total else "")
             + (f" · {interrupted} interrupted" if interrupted else "")),
        _kpi("triangle-alert", "Total Errors", f"{total_errors:,}", "bad" if total_errors else "ok",
             f"{bad:,} failed runs · {failed_tools:,} failed tool calls"
             + (f" · {silent_fail:,} wrong answers" if silent_fail else "")),
    ])

    # ================= ROW 1: runs over time | latency per agent =================
    col_a, col_b = st.columns([3, 2], gap="medium")

    with col_a:
        heading("cpu", "Infrastructure & Run Performance")
        with panel(ROW1_H):
            valid = runs.dropna(subset=["ts"])
            span = (valid["ts"].max() - valid["ts"].min()) if not valid.empty else pd.Timedelta(0)
            freq, unit = ("h", "hour") if span <= pd.Timedelta(days=2) else ("D", "day")
            _title(f"Agent Runs Over Time (per {unit})",
                   "Total, successful and failed runs in each period (latest 200 runs of each agent).")
            loaded_other = loaded_total - loaded_ok - loaded_bad
            st.markdown(
                f'<span class="chip cyan">Total · {loaded_total:,}</span>'
                f'<span class="ao-pill success">Successful · {loaded_ok:,}</span> '
                f'<span class="ao-pill error">Failed · {loaded_bad:,}</span>'
                + (f' <span class="ao-pill warn">Other · {loaded_other:,}</span>' if loaded_other else ""),
                unsafe_allow_html=True,
            )

            if valid.empty:
                st.info("No timestamps available.")
            else:
                valid = valid.assign(bucket=valid["ts"].dt.tz_convert(DISPLAY_TIMEZONE).dt.floor(freq))
                # Periods with no runs count as 0, so the lines do not invent a trend.
                full_range = pd.date_range(valid["bucket"].min(), valid["bucket"].max(), freq=freq)
                series = (
                    pd.DataFrame({
                        "Total Runs": valid.groupby("bucket").size(),
                        "Successful": valid[valid["kind"] == "success"].groupby("bucket").size(),
                        "Failed": valid[valid["kind"] == "error"].groupby("bucket").size(),
                    })
                    .reindex(full_range).fillna(0).rename_axis("bucket").reset_index()
                )

                # (column, colour, line width, marker size, fill). Big marker first,
                # small marker on top: a point with 0 runs stays visible even when
                # all three series share the same value.
                styles = (
                    ("Total Runs", CYAN, 4, 12, "rgba(34,211,238,.10)"),
                    ("Successful", "#22c55e", 2.5, 8, None),
                    ("Failed", RED, 2.5, 5, None),
                )
                fig = go.Figure()
                for column, color, width, _size, fill in styles:
                    xs, ys = smooth_curve(list(series["bucket"]), list(series[column]))
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, name=column, mode="lines", hoverinfo="skip",
                        line={"color": color, "width": width, "shape": "linear"},
                        fill="tozeroy" if fill else None, fillcolor=fill, cliponaxis=False,
                    ))
                for column, color, _width, size, _fill in styles:
                    fig.add_trace(go.Scatter(
                        x=series["bucket"], y=series[column], name=column, mode="markers",
                        showlegend=False, cliponaxis=False,
                        marker={"size": size, "color": color, "line": {"color": "#0b1120", "width": 1.5}},
                    ))
                _dark(fig, 310)
                fig.update_layout(legend={"font": {"color": "#e2e8f0"}})
                fig.update_layout(hovermode="x unified", margin={"l": 16, "r": 22, "t": 10, "b": 10})
                fig.update_xaxes(tickformat="%d %b" if unit == "day" else "%d %b, %H:%M")
                # The axis starts exactly at 0 (the curve never goes below it);
                # markers are not clipped, so points with 0 runs are drawn in full.
                y_max = float(series[["Total Runs", "Successful", "Failed"]].max().max() or 1)
                step = max(1, int(-(-y_max // 4)))
                fig.update_yaxes(
                    range=[0, y_max * 1.12],
                    tickvals=list(range(0, int(y_max * 1.12) + 1, step)),
                    zeroline=True, zerolinecolor="rgba(148,163,184,.45)", zerolinewidth=1,
                )
                show_chart(fig, key="ag_runs_time")

    with col_b:
        heading("bar-chart", "AI Performance & Agent Metrics")
        with panel(ROW1_H):
            _title("Avg. Latency per Agent (s)")
            lat = (
                runs.groupby(["agent_name", "agent_label"])["latency_done"].mean()
                .dropna().reset_index(name="latency")
            )
            if lat.empty:
                st.info("No completed runs with latency yet.")
            else:
                log_scale = st.toggle(
                    "Log scale",
                    value=True,
                    key="ov_latency_log",
                    help="Keeps fast agents visible next to slow ones. Each step up the axis is 10x.",
                )
                fig_lat = px.bar(
                    lat, x="agent_label", y="latency", color="agent_label",
                    color_discrete_map=agent_color_map(lat["agent_name"]),
                    labels={"agent_label": "Agent", "latency": "Seconds (log)" if log_scale else "Seconds"},
                    text=lat["latency"].round(2),
                )
                fig_lat.update_layout(showlegend=False)
                fig_lat.update_traces(textposition="outside", cliponaxis=False)
                if log_scale:
                    fig_lat.update_yaxes(type="log")
                show_chart(_dark(fig_lat, 345), key="ag_latency")

    # ================= ROW 2: CPU | memory | active agents =================
    col_c, col_d = st.columns([3, 2], gap="medium")

    with col_c:
        g1, g2 = st.columns(2, gap="medium")
        with g1, panel(ROW2_H):
            if psutil:
                cpu = psutil.cpu_percent(interval=0.2)
                show_chart(_gauge(cpu, "CPU Load"), key="ag_cpu", modebar=False)
                try:
                    load = f" · load avg {os.getloadavg()[0]:.1f}"
                except (AttributeError, OSError):
                    load = ""
                st.caption(f"{psutil.cpu_count()} cores{load}")
            else:
                st.info("Install psutil (`pip install psutil`) to see CPU & RAM gauges.")
        with g2, panel(ROW2_H):
            if psutil:
                mem = psutil.virtual_memory()
                show_chart(_gauge(mem.percent, "Memory Usage"), key="ag_mem", modebar=False)
                st.caption(f"RAM: {mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB")
            else:
                # Fall back to the run success rate when psutil is not installed.
                show_chart(_gauge(rate, "Run Success Rate"), key="ag_rate", modebar=False)

    with col_d, panel(ROW2_H):
        _title("Active Agents", "Based on the latest 200 runs of each agent")
        rows = []
        for name, grp in runs.sort_values("ts", ascending=False).groupby("agent_name"):
            last_kind = grp.iloc[0]["kind"]
            if (grp["kind"] == "running").any():
                status_html = '<span class="st-run">Running</span>'
            elif last_kind == "error":
                status_html = '<span class="st-bad">Error</span>'
            elif last_kind == "success":
                status_html = '<span class="st-ok">Healthy</span>'
            else:
                status_html = '<span class="st-idle">Idle</span>'
            s_rate = (grp["kind"] == "success").mean() * 100
            avg_lat = grp["latency_done"].mean()
            rows.append([
                html.escape(pretty_name(name)),
                status_html,
                len(grp),
                f"{s_rate:.0f}%",
                f"{avg_lat:.2f}s" if pd.notna(avg_lat) else "N/A",
            ])
        st.markdown(_table(["Agent", "Status", "Runs", "Success", "Latency"], rows, 200),
                    unsafe_allow_html=True)

    # ================= ROW 3: activity logs | alerts =================
    col_e, col_f = st.columns([3, 2], gap="medium")

    with col_e, panel(ROW3_H):
        _title("Real-Time Activity Logs", "Latest agent runs and tool executions")
        logs = []
        for r in runs.sort_values("ts", ascending=False).head(15).to_dict("records"):
            logs.append((r["ts"], short_id(r.get("trace_id")), "RUN", r["kind"],
                         f'{pretty_name(r.get("agent_name"))}: {r.get("user_input") or ""}'))
        if df_tool_events is not None and not df_tool_events.empty:
            te = df_tool_events.copy()
            te["ts"] = pd.to_datetime(te["timestamp"], utc=True, errors="coerce")
            for e in te.sort_values("ts", ascending=False).head(15).to_dict("records"):
                lat = e.get("latency")
                lat_txt = f" ({float(lat):.2f}s)" if lat not in (None, "") else ""
                logs.append((e["ts"], short_id(e.get("trace_id")), "TOOL",
                             status_kind(e.get("status")),
                             f'{pretty_name(e.get("tool_name"))} · {pretty_name(e.get("status"))}{lat_txt}'))
        logs.sort(key=lambda x: x[0] if pd.notna(x[0]) else pd.Timestamp(0, tz="UTC"), reverse=True)

        rows = []
        for ts, tid, typ, kind, msg in logs[:14]:
            label, cls = _level(kind)
            rows.append([
                html.escape(format_timestamp(ts, compact=True)),
                html.escape(tid),
                typ,
                f'<span class="lv {cls}">{label}</span>',
                _short(msg, 70),
            ])
        st.markdown(_table(["Time", "ID", "Source", "Level", "Message"], rows, 335),
                    unsafe_allow_html=True)

    with col_f, panel(ROW3_H):
        _title("Alert Notifications", "Failed and flagged executions")
        alerts = []
        flagged_runs = runs[runs["kind"].isin(["error", "warn"])].sort_values("ts", ascending=False).head(8)
        for r in flagged_runs.to_dict("records"):
            crit = r["kind"] == "error"
            msg = _clean(r.get("error_message"), pretty_name(r.get("status")))
            alerts.append((r["ts"], "crit" if crit else "warn",
                           "Critical" if crit else "Warning",
                           f'{pretty_name(r.get("agent_name"))}: {msg}'))
        if df_tool_events is not None and not df_tool_events.empty:
            for e in df_tool_events.to_dict("records"):
                if status_kind(e.get("status")) == "error":
                    alerts.append((pd.to_datetime(e.get("timestamp"), utc=True, errors="coerce"),
                                   "crit", "Critical",
                                   f'{pretty_name(e.get("tool_name"))}: '
                                   f'{_clean(e.get("error_message"), "Tool failed")}'))
        alerts.sort(key=lambda x: x[0] if pd.notna(x[0]) else pd.Timestamp(0, tz="UTC"), reverse=True)

        if not alerts:
            st.success("All clear — no failures in the loaded records.")
        else:
            rows = [
                [html.escape(format_timestamp(ts, compact=True)),
                 f'<span class="nw"><span class="dot {cls}"></span>{label}</span>',
                 _short(msg, 70)]
                for ts, cls, label, msg in alerts[:8]
            ]
            st.markdown(_table(["Time", "Severity", "Message"], rows, 335),
                        unsafe_allow_html=True)

    render_failures_section(quality, failures_data, totals)

    render_time_window(runs, df_tool_events)

    if totals and totals["from_database"]:
        st.caption(
            "KPI cards count every run in the database. Charts, logs and alerts show "
            f"the latest 200 runs of each agent · {tokens:,} tokens used in total"
        )
    else:
        st.caption(
            f"Based on the latest 200 runs of each agent · {tokens:,} tokens used in total"
        )


# --------------------------------------------------
# CARDS & BADGES (Agents / Tools)
# --------------------------------------------------

TESTING_TOOLS = {"flaky_tool"}
AGENT_ICONS = {
    "tool_agent": "wrench",
    "research_agent": "search",
    "document_agent": "file-text",
}
DOC_STEPS = [
    ("Upload", "Add a PDF, Word, text, CSV, Excel or PowerPoint file."),
    ("Process", "The document is split into chunks and indexed for search."),
    ("Ask", "Answers come from the most relevant passages, with sources."),
]


def tool_card_html(row):
    raw_name = row.get("tool_name")
    testing = raw_name in TESTING_TOOLS
    rate = float(row.get("success_rate") or 0)
    return (
        f'<div class="hc{" testing" if testing else ""}">'
        '<div class="hc-top">'
        f'<div class="hc-icon">{svg_icon("wrench")}</div>'
        f'<div class="hc-name">{html.escape(pretty_name(raw_name))}</div>'
        f"{health_badge(rate, testing)}</div>"
        f"{rate_cell(rate)}"
        + stat_block([
            ("Calls", f"{int(row.get('total_calls') or 0):,}"),
            ("Failed", f"{int(row.get('failed_calls') or 0):,}"),
            ("Avg latency", f"{float(row.get('average_latency') or 0):.2f}s"),
        ])
        + "</div>"
    )


def agent_card_html(agent_id, grp, group_df=None):
    """Health card for one agent.

    group_df: this agent's rows from /agent-stats (all runs in the database).
    grp: this agent's loaded runs (latest 200); used for median, peak and last run.
    """
    name = pretty_name(agent_id)
    icon = svg_icon(AGENT_ICONS.get(agent_id, "bot"))
    has_loaded = grp is not None and not grp.empty
    has_groups = group_df is not None and not group_df.empty

    if not has_loaded and not has_groups:
        return (
            '<div class="hc empty"><div class="hc-top">'
            f'<div class="hc-icon">{icon}</div><div class="hc-name">{html.escape(name)}</div>'
            '<span class="ao-pill neutral">No runs yet</span></div>'
            '<div class="hc-note">This agent has not been run yet. '
            "Open the Agent Lab to try it.</div></div>"
        )

    # ---- counts, average latency and tokens ----
    if has_groups:
        counts = {row.status: int(row.runs) for row in group_df.itertuples()}
        done = group_df[~group_df["status"].isin(INCOMPLETE_STATUSES)]
        lat_n = float(done["latency_count"].sum())
        avg_latency = float(done["latency_sum"].sum()) / lat_n if lat_n else None
        tokens = float(group_df["input_tokens"].sum() + group_df["output_tokens"].sum())
    else:
        counts = {k: int(v) for k, v in grp.groupby("status").size().items()}
        lat_all = grp["latency_done"].dropna()
        avg_latency = float(lat_all.mean()) if not lat_all.empty else None
        tokens = float(
            pd.to_numeric(grp["total_input_tokens"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(grp["total_output_tokens"], errors="coerce").fillna(0).sum()
        )

    total = sum(counts.values())
    ok = sum(n for s, n in counts.items() if status_kind(s) == "success")
    finished = sum(n for s, n in counts.items() if s not in INCOMPLETE_STATUSES)
    rate = ok / finished * 100 if finished else 0.0

    # ---- last run, median and peak come from the loaded runs ----
    last_time, last_kind = "N/A", "neutral"
    median_latency = peak_latency = None
    if has_loaded:
        stamps = pd.to_datetime(grp["timestamp"], utc=True, errors="coerce")
        if stamps.notna().any():
            last_idx = stamps.idxmax()
            last_time = format_timestamp(grp.loc[last_idx, "timestamp"], include_seconds=False)
            last_kind = grp.loc[last_idx, "kind"]
        loaded_lat = grp["latency_done"].dropna()
        if not loaded_lat.empty:
            median_latency, peak_latency = float(loaded_lat.median()), float(loaded_lat.max())

    if counts.get("running", 0) > 0:
        badge = '<span class="ao-pill running">Running</span>'
    elif last_kind == "error":
        badge = '<span class="ao-pill error">Last run failed</span>'
    elif finished == 0:
        badge = '<span class="ao-pill neutral">No finished runs</span>'
    elif rate >= 90:
        badge = '<span class="ao-pill success">Healthy</span>'
    elif rate >= 70:
        badge = '<span class="ao-pill warn">Degraded</span>'
    else:
        badge = '<span class="ao-pill error">Unreliable</span>'

    ordered = sorted(counts.items(), key=lambda item: (status_kind(item[0]) != "success", -item[1]))
    chips = "".join(
        f'<span class="ao-pill {status_kind(status)}">{html.escape(pretty_status(status))} · {count:,}</span>'
        for status, count in ordered
    )

    def seconds(value):
        return f"{value:.1f}s" if value is not None else "N/A"

    note = f"Last run · {last_time}"
    if has_groups and has_loaded:
        note += " · median and peak use the latest 200 runs"

    return (
        '<div class="hc"><div class="hc-top">'
        f'<div class="hc-icon">{icon}</div><div class="hc-name">{html.escape(name)}</div>{badge}</div>'
        f"{rate_cell(rate)}"
        f'<div class="hc-chips">{chips}</div>'
        + stat_block([
            ("Runs", f"{total:,}"),
            ("Avg", seconds(avg_latency)),
            ("Median", seconds(median_latency)),
            ("Peak", seconds(peak_latency)),
            ("Tokens", fmt_compact(tokens)),
        ])
        + f'<div class="hc-note">{html.escape(note)}</div></div>'
    )


def agent_latency_figure(agent_name, source_df):
    agent_df = source_df.sort_values("timestamp_sort", na_position="last").reset_index(drop=True)
    agent_df["run_number"] = range(1, len(agent_df) + 1)
    agent_df["run_label"] = agent_df["run_number"].apply(lambda number: f"Run {number}")
    agent_df["timestamp_label"] = agent_df["timestamp"].apply(format_timestamp)
    agent_df["latency_label"] = agent_df["total_latency"].apply(lambda value: f"{value:.2f}s")

    run_count = len(agent_df)
    hover_fields = {
        "trace_id": True,
        "timestamp_label": True,
        "user_input": True,
        "status_label": True,
        "total_latency": ":.3f",
        "run_number": False,
        "run_label": True,
        "latency_label": False,
    }
    color_map = status_color_map(agent_df["status"].unique())
    axis_labels = {
        "run_number": "Execution",
        "total_latency": "Latency (seconds)",
        "status_label": "Run Status",
    }
    tick_step = max(1, (run_count + 9) // 10)
    tick_values = list(range(1, run_count + 1, tick_step))
    if tick_values[-1] != run_count:
        tick_values.append(run_count)

    if agent_name == "tool_agent":
        fig = px.scatter(
            agent_df, x="run_number", y="total_latency", color="status_label",
            color_discrete_map=color_map, text="latency_label",
            hover_data=hover_fields, labels=axis_labels,
        )
        fig.update_traces(
            mode="markers+text", textposition="top center", textfont={"size": 10},
            marker={"size": 10, "opacity": 0.9}, cliponaxis=False,
        )
        fig.update_xaxes(tickmode="array", tickvals=tick_values,
                         ticktext=[f"Run {n}" for n in tick_values])
        height = 420
    elif agent_name == "document_agent":
        fig = px.bar(
            agent_df, x="total_latency", y="run_label", orientation="h",
            color="status_label", color_discrete_map=color_map, text="latency_label",
            hover_data=hover_fields,
            labels={"run_label": "Document Query", "total_latency": "Latency (seconds)",
                    "status_label": "Run Status"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(categoryorder="array", categoryarray=agent_df["run_label"].tolist(),
                         title="Document Agent Runs")
        fig.update_xaxes(rangemode="tozero", title="Latency (seconds)")
        height = max(330, min(650, 42 * run_count + 120))
    else:
        fig = px.line(
            agent_df, x="run_number", y="total_latency", color="status_label",
            color_discrete_map=color_map, markers=True, text="latency_label",
            hover_data=hover_fields, labels=axis_labels,
        )
        fig.update_traces(textposition="top center", textfont={"size": 10}, cliponaxis=False)
        fig.update_xaxes(tickmode="array", tickvals=tick_values,
                         ticktext=[f"Run {n}" for n in tick_values])
        height = 400

    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    return fig, height, agent_df


# --------------------------------------------------
# WORKSPACE HELPERS
# --------------------------------------------------

TOOL_CATALOG = [
    {
        "name": "Calculator",
        "purpose": "Math, arithmetic and scientific calculations.",
        "capabilities": "Arithmetic, powers, roots, trigonometry, logarithms, factorial, combinations and rounding.",
        "example": "Calculate the compound interest on ₹50,000 at 8% for 3 years.",
        "kind": "Utility",
    },
    {
        "name": "System Info",
        "purpose": "Inspect the machine running the agent.",
        "capabilities": "CPU usage, RAM, disk, network and system uptime.",
        "example": "Show my current CPU usage, available RAM and disk space.",
        "kind": "System",
    },
    {
        "name": "Weather",
        "purpose": "Current and forecast weather for any city.",
        "capabilities": "Location lookup and weather conditions through Open-Meteo.",
        "example": "What's the weather in Bengaluru?",
        "kind": "Information",
    },
    {
        "name": "Wikipedia",
        "purpose": "Quick background on any topic.",
        "capabilities": "Searches Wikipedia and returns a summary with a source link.",
        "example": "Find information about LangGraph.",
        "kind": "Research",
    },
    {
        "name": "Flaky Tool",
        "purpose": "Simulates tool failures for observability testing.",
        "capabilities": "Success, controlled delay, forced error, timeout and intermittent failure.",
        "example": "Run flaky_tool in timeout mode with a 2-second timeout.",
        "kind": "Testing",
    },
]

TOOL_EXAMPLES = [
    "Calculate 125 × 48, then find its square root",
    "Show CPU usage, available RAM and disk space",
    "Compare the weather in Bengaluru and Mumbai",
    "Look up what LangGraph is and summarize it",
    "Test the flaky tool with intermittent failures",
]

DOC_SUGGESTIONS = [
    "Give me a short summary of this document",
    "What are the key points?",
    "List any important names, dates or numbers",
]

RESEARCH_STEPS = [
    ("Collect sources", "Searches the web or reads the URLs you provide."),
    ("Extract evidence", "Keeps only the passages that are relevant to your topic."),
    ("Write the report", "Builds a structured report with citations."),
    ("Critic review", "Scores relevance, evidence and citation quality, and flags unsupported claims."),
]


def workspace_banner(icon, title, text, chips=None):
    chip_html = "".join(f'<span class="chip cyan">{html.escape(c)}</span>' for c in (chips or []))
    st.markdown(
        f"""
        <div class="ws-banner">
            <div class="ws-icon">{svg_icon(icon)}</div>
            <div>
                <h3>{html.escape(title)}</h3>
                <p>{html.escape(text)}</p>
                <div style="margin-top:.55rem">{chip_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def steps_html(steps):
    return "".join(
        f'<div class="step-row"><div class="step-num">{index}</div>'
        f'<div class="step-text"><b>{html.escape(title)}</b><span>{html.escape(desc)}</span></div></div>'
        for index, (title, desc) in enumerate(steps, start=1)
    )


def empty_state(icon, title, text):
    st.markdown(
        f'<div class="empty-state"><div class="big">{svg_icon(icon)}</div>'
        f"<h4>{html.escape(title)}</h4><p>{html.escape(text)}</p></div>",
        unsafe_allow_html=True,
    )


def render_run_meta(run_status, tool_names, trace_id, executed_tools):
    chips = "".join(
        f'<span class="chip cyan">{html.escape(name)}</span>' for name in tool_names
    ) or '<span class="chip">No tools used</span>'
    st.markdown(
        f'<div style="margin-top:.5rem">{pill(run_status)}&nbsp; {chips}</div>',
        unsafe_allow_html=True,
    )

def render_run_meta(run_status, tool_names, trace_id, executed_tools):
    chips = "".join(
        f'<span class="chip cyan">{html.escape(name)}</span>' for name in tool_names
    ) or '<span class="chip">No tools used</span>'
    st.markdown(
        f'<div style="margin-top:.5rem">{pill(run_status)}&nbsp; {chips}</div>',
        unsafe_allow_html=True,
    )

    if run_status == "max_rounds_reached":
        st.warning(
            "The agent hit its tool-call limit, so this run is incomplete. "
            "Try splitting the request into smaller tasks."
        )
    elif status_kind(run_status) == "error":
        st.error("This run failed. Open its trace in Monitoring to see what went wrong.")

    if trace_id or executed_tools:
        with st.expander("Run details"):
            if trace_id:
                st.caption("Trace ID · find it in Monitoring → Trace Explorer")
                st.code(trace_id)
            for item in executed_tools or []:
                st.markdown(
                    f"**{pretty_name(item.get('tool_name', 'Tool'))}** "
                    f"{pill(item.get('status', 'unknown'))}",
                    unsafe_allow_html=True,
                )
                if item.get("error"):
                    st.error(str(item["error"]))


def render_tool_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("status"):
            render_run_meta(
                message["status"],
                message.get("tools", []),
                message.get("trace_id"),
                message.get("execution_details", []),
            )


def render_research_result(result):
    status = result.get("status", "unknown")
    trace_id = result.get("trace_id")
    metrics = result.get("metrics", {})
    critic = result.get("critic", {})

    if status == "success":
        st.success("Research completed successfully.")
    elif status == "needs_review":
        st.warning("The report was generated, but the critic found issues that need your review.")
    else:
        st.error(f"Research finished with status: {pretty_status(status)}")

    kpi_row([
        ("Status", pretty_status(status)),
        ("Pages Collected", metrics.get("pages_collected", 0)),
        ("Usable Sources", metrics.get("usable_sources", 0)),
        ("Evidence Items", metrics.get("evidence_items", 0)),
    ])

    report = result.get("report")
    if report:
        with st.container(border=True):
            head_col, btn_col = st.columns([4, 1])
            head_col.markdown("### :material/article: Research Report")
            btn_col.download_button(
                "Download",
                icon=":material/download:",
                data=report,
                file_name="research_report.md",
                mime="text/markdown",
                key="download_research_report",
                width="stretch",
            )
            st.markdown(report)

    if critic:
        with st.expander("Critic evaluation", expanded=True, icon=":material/science:"):
            kpi_row([
                ("Overall Score", critic.get("overall_score", "N/A")),
                ("Topic Relevance", critic.get("topic_relevance", "N/A")),
                ("Evidence Quality", critic.get("evidence_quality", "N/A")),
                ("Citation Quality", critic.get("citation_quality", "N/A")),
            ])

            if critic.get("citation_issues"):
                st.markdown("**Citation issues**")
                for issue in critic["citation_issues"]:
                    st.warning(issue)

            if critic.get("unsupported_claims"):
                st.markdown("**Unsupported claims**")
                st.json(critic["unsupported_claims"])

            if critic.get("verdict"):
                st.markdown("**Critic verdict**")
                st.write(critic["verdict"])

    if result.get("error"):
        st.error(result["error"])

    if trace_id:
        with st.expander("Run details"):
            st.caption("Trace ID · find it in Monitoring → Trace Explorer")
            st.code(trace_id)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "document_id": None,
    "document_name": None,
    "document_chat_history": [],
    "tool_chat_history": [],
    "research_result": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

NAV_LABELS = {
    "Monitoring": ":material/monitoring:  Monitoring",
    "Agent Lab": ":material/smart_toy:  Agent Lab",
}


def sidebar_label(text):
    st.markdown(f'<div class="sb-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def snapshot_html(snap):
    if not snap or not snap.get("total"):
        runs_v, failed_v, rate_v = "–", "–", "–"
        failed_cls, rate_cls = "", ""
    else:
        runs_v = f"{snap['total']:,}"
        failed_v = f"{snap['failed']:,}"
        rate_v = f"{snap['rate']:.0f}%"
        failed_cls = "bad" if snap["failed"] else "good"
        rate_cls = "good" if snap["rate"] >= 90 else ("bad" if snap["rate"] < 70 else "")
    return (
        '<div class="sb-snap">'
        f'<div class="sb-tile"><b>{runs_v}</b><span>Runs</span></div>'
        f'<div class="sb-tile {failed_cls}"><b>{failed_v}</b><span>Failed</span></div>'
        f'<div class="sb-tile {rate_cls}"><b>{rate_v}</b><span>Success</span></div>'
        "</div>"
    )


with st.sidebar:
    st.markdown(
        f"""
        <div class="sb-brand">
            <div class="sb-logo">{svg_icon("activity")}</div>
            <div>
                <div class="sb-name">Observality Room</div>
                <div class="sb-tag">AI agent monitoring</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar_label("Navigation")
    page = st.radio(
        "Navigate",
        list(NAV_LABELS.keys()),
        format_func=lambda key: NAV_LABELS[key],
        label_visibility="collapsed",
    )

    sidebar_label("System status")
    if api_online():
        st.markdown(
            f"""
            <div class="sb-status on">
                <div class="sb-dot"></div>
                <div>
                    <div class="sb-status-title">API connected</div>
                    <div class="sb-status-sub">{html.escape(API_URL)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sb-status off">
                <div class="sb-dot"></div>
                <div>
                    <div class="sb-status-title">API offline</div>
                    <div class="sb-status-sub">Backend is not reachable</div>
                </div>
            </div>
            <div class="sb-code">uvicorn agentops.api.main:app --reload</div>
            """,
            unsafe_allow_html=True,
        )

    sidebar_label("Live snapshot")
    snapshot_slot = st.empty()
    snapshot_slot.markdown(
        snapshot_html(st.session_state.get("snapshot")), unsafe_allow_html=True
    )

    sidebar_label("Actions")
    if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
        st.rerun()
    st.caption(
        "Last updated "
        + pd.Timestamp.now(tz="Asia/Kolkata").strftime("%d %b, %I:%M:%S %p")
    )

    st.markdown(
        '<div class="sb-tip"><b>Tip:</b> run an agent from the Agent Lab, '
        "then open Monitoring → Trace Explorer to see every step it took.</div>",
        unsafe_allow_html=True,
    )


# ==================================================
# MONITORING
# ==================================================

if page == "Monitoring":
    st.session_state.pop("_scrolled_for", None)

    page_hero(
        "layout-dashboard",
        "Mission control",
        "AI Agent Watch Tower",
        "One screen for every agent: runs, failures, silent wrong answers, tool calls and full traces.",
        badge="Live monitoring",
    )

    try:
        # Latest 200 runs PER agent, so a busy agent cannot hide the others.
        runs = []
        for agent_key in ("tool_agent", "research_agent", "document_agent"):
            agent_runs = api_get(
                "/agent-runs",
                params={"agent_name": agent_key, "limit": 200, "offset": 0},
            )
            runs.extend(agent_runs.get("runs", []))
        runs.sort(key=lambda item: item.get("timestamp") or "", reverse=True)

        tool_analytics = api_get("/tool-analytics")
        tool_events_data = api_get(
            "/tool-events",
            params={"limit": 200, "offset": 0}
        )
        tool_events = tool_events_data.get("events", [])

        # True totals of the whole database (older API versions: not available).
        try:
            agent_stats = api_get("/agent-stats")
        except requests.RequestException:
            agent_stats = None

        # Answer quality labels and failures (older API versions: not available).
        try:
            quality = api_get("/labels/summary")
            doc_outcomes = api_get("/labels/document-outcomes")
        except requests.RequestException:
            quality, doc_outcomes = None, None
        try:
            failures_data = api_get("/failures", params={"limit": 60})
        except requests.RequestException:
            failures_data = None
        agent_stats = apply_document_quality(agent_stats, doc_outcomes)

        # Legacy traces are retained as a separate historical source.
        try:
            legacy_traces = api_get("/traces")
        except requests.RequestException:
            legacy_traces = []

    except requests.RequestException as error:
        st.error(f"Could not connect to the Watch Tower API: {error}")
        st.stop()

    df_runs_all = apply_silent_failures(mark_interrupted(pd.DataFrame(runs)), failures_data)
    df_tool_events = pd.DataFrame(tool_events)
    df_legacy = pd.DataFrame(legacy_traces)

    # ----- derived run columns -----
    if not df_runs_all.empty:
        df_runs_all["kind"] = df_runs_all["status"].apply(status_kind)
        df_runs_all["status_label"] = df_runs_all["status"].apply(pretty_status)
        df_runs_all["agent_label"] = df_runs_all["agent_name"].apply(pretty_name)
        df_runs_all["latency_done"] = pd.to_numeric(
            df_runs_all["total_latency"], errors="coerce"
        ).where(~df_runs_all["status"].isin(INCOMPLETE_STATUSES))

    _groups, _from_database = stats_groups(agent_stats, df_runs_all)
    totals = run_totals(_groups, _from_database)

    if totals is None:
        _snap = {"total": 0, "failed": 0, "rate": 0.0}
    else:
        _snap = {
            "total": totals["total"],
            "failed": totals["bad"],
            "rate": totals["rate"],
        }
    st.session_state.snapshot = _snap
    snapshot_slot.markdown(snapshot_html(_snap), unsafe_allow_html=True)

    tab_overview, tab_agents, tab_tools, tab_history, tab_trace = st.tabs([
        ":material/dashboard: Overview",
        ":material/groups: Agents",
        ":material/build: Tools",
        ":material/history: Run History",
        ":material/manage_search: Trace Explorer",
    ])

    # ----------------------------------------------
    # OVERVIEW
    # ----------------------------------------------

    with tab_overview:
        render_overview(df_runs_all, df_tool_events, tool_analytics, totals, quality, failures_data)

    # ----------------------------------------------
    # AGENTS
    # ----------------------------------------------

    with tab_agents:
        heading("bot", "Agent health")

        known_agents = ["tool_agent", "research_agent", "document_agent"]
        present_agents = set(_groups["agent_name"].dropna().unique().tolist())
        if not df_runs_all.empty:
            present_agents |= set(df_runs_all["agent_name"].dropna().unique().tolist())
        agent_ids = known_agents + sorted(a for a in present_agents if a not in known_agents)

        cards = []
        for agent_id in agent_ids:
            grp = df_runs_all[df_runs_all["agent_name"] == agent_id] if not df_runs_all.empty else None
            group_df = _groups[_groups["agent_name"] == agent_id] if _from_database else None
            cards.append(agent_card_html(agent_id, grp, group_df))
        st.markdown('<div class="hc-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
        st.caption(
            "Cards count every run in the database; success rate uses finished runs only. "
            "Median, peak and the charts below use the latest 200 runs of each agent."
            if _from_database else
            "Based on the latest 200 runs of each agent."
        )

        if df_runs_all.empty:
            st.info("No agent runs yet. Run an agent from the Agent Lab and it will appear here automatically.")
        else:
            heading("bar-chart", "Run outcomes by agent")
            with st.container(border=True):
                _title(
                    "Every run, split by outcome",
                    "All statuses are shown, so success + failed + other always adds up to the total.",
                )
                outcomes = _groups.copy()
                outcomes["agent_label"] = outcomes["agent_name"].apply(pretty_name)
                outcomes["status_label"] = outcomes["status"].apply(pretty_status)
                outcomes = outcomes[["agent_label", "status_label", "runs"]]
                fig_outcomes = px.bar(
                    outcomes, y="agent_label", x="runs", color="status_label", orientation="h",
                    color_discrete_map=status_color_map(_groups["status"].unique()),
                    labels={"agent_label": "Agent", "runs": "Runs", "status_label": "Status"},
                )
                fig_outcomes.update_layout(barmode="stack", legend_title_text="")
                _dark(fig_outcomes, 290)
                fig_outcomes.update_layout(margin={"l": 10, "r": 10, "t": 52, "b": 10},
                                           legend={"y": 1.12})
                show_chart(fig_outcomes, key="agents_outcomes")

            heading("clock", "Latency")

            chart_source = df_runs_all.copy()
            chart_source["timestamp_sort"] = pd.to_datetime(
                chart_source["timestamp"], utc=True, errors="coerce"
            )
            running_count = int(chart_source["status"].isin(INCOMPLETE_STATUSES).sum())

            completed_source = chart_source[
                ~chart_source["status"].isin(INCOMPLETE_STATUSES)
            ].copy()
            completed_source["total_latency"] = pd.to_numeric(
                completed_source["total_latency"], errors="coerce"
            )
            completed_source = completed_source[
                completed_source["total_latency"].notna() & (completed_source["total_latency"] >= 0)
            ]

            if running_count:
                st.info(
                    f"{running_count} running or interrupted execution(s) are left out because "
                    "their final latency is not available."
                )

            if completed_source.empty:
                st.info("No completed executions with latency data are available.")
            else:
                latency_agents = sorted(completed_source["agent_name"].dropna().unique().tolist())
                view = st.segmented_control(
                    "Latency view",
                    ["all"] + latency_agents,
                    format_func=lambda x: "Compare all agents" if x == "all" else pretty_name(x),
                    default="all",
                    key="agents_latency_view",
                    label_visibility="collapsed",
                ) or "all"

                if view == "all":
                    with st.container(border=True):
                        _title(
                            "Latency distribution by agent (seconds)",
                            "Each box covers the middle 50% of runs; dots are outliers.",
                        )
                        log_cmp = st.toggle(
                            "Log scale", value=True, key="agents_cmp_log",
                            help="Keeps fast agents visible next to slow ones. Each step up the axis is 10x.",
                        )
                        fig_cmp = px.box(
                            completed_source, x="agent_label", y="total_latency", color="agent_label",
                            points="outliers",
                            color_discrete_map=agent_color_map(completed_source["agent_name"]),
                            labels={"agent_label": "Agent", "total_latency": "Latency (s)"},
                        )
                        fig_cmp.update_layout(showlegend=False)
                        if log_cmp:
                            fig_cmp.update_yaxes(type="log")
                        show_chart(_dark(fig_cmp, 360), key="agents_latency_box")
                else:
                    fig_agent, chart_height, agent_df = agent_latency_figure(
                        view, completed_source[completed_source["agent_name"] == view].copy()
                    )
                    kpi_grid([
                        _kpi("activity", "Completed Runs", len(agent_df)),
                        _kpi("timer", "Average Latency", f"{agent_df['total_latency'].mean():.2f}s"),
                        _kpi("gauge", "Median Latency", f"{agent_df['total_latency'].median():.2f}s",
                             accent="purple"),
                        _kpi("triangle-alert", "Peak Latency", f"{agent_df['total_latency'].max():.2f}s"),
                    ])
                    with st.container(border=True):
                        _title(f"{pretty_name(view)} · latency per execution")
                        show_chart(_dark(fig_agent, chart_height), key=f"latency_chart_{view}")

            heading("triangle-alert", "Failure breakdown")
            failed_runs_df = df_runs_all[df_runs_all["kind"] == "error"].copy()
            if failed_runs_df.empty:
                st.success("No failed runs in the records currently loaded.")
            else:
                if "error_type" not in failed_runs_df.columns:
                    failed_runs_df["error_type"] = "Unclassified"
                failed_runs_df["error_type"] = failed_runs_df["error_type"].fillna("Unclassified")
                by_error = (
                    failed_runs_df.groupby(["agent_label", "error_type"]).size().reset_index(name="failures")
                )
                with st.container(border=True):
                    _title("Failed runs by error type", "Which agent fails, and why.")
                    fig_fail = px.bar(
                        by_error, x="error_type", y="failures", color="agent_label",
                        color_discrete_map=agent_color_map(failed_runs_df["agent_name"]),
                        labels={"error_type": "Error Type", "failures": "Failed Runs", "agent_label": "Agent"},
                    )
                    show_chart(_dark(fig_fail, 300), key="agents_failures")

    # ----------------------------------------------
    # TOOLS
    # ----------------------------------------------

    with tab_tools:
        df_tools = pd.DataFrame(tool_analytics.get("tools", []))

        if df_tools.empty:
            st.info("No tool calls yet. Ask the Tool Agent to do something and its executions will show up here.")
        else:
            for col in ("total_calls", "successful_calls", "failed_calls",
                        "success_rate", "failure_rate", "average_latency"):
                if col not in df_tools.columns:
                    df_tools[col] = 0
                df_tools[col] = pd.to_numeric(df_tools[col], errors="coerce").fillna(0)
            df_tools["is_testing"] = df_tools["tool_name"].isin(TESTING_TOOLS)

            exclude_testing = st.toggle(
                "Exclude testing tools (Flaky Tool) from totals and charts",
                key="tools_exclude_testing",
            )
            view_tools = df_tools[~df_tools["is_testing"]] if exclude_testing else df_tools
            if view_tools.empty:
                view_tools = df_tools

            t_total = int(view_tools["total_calls"].sum())
            t_ok = int(view_tools["successful_calls"].sum())
            t_fail = int(view_tools["failed_calls"].sum())
            t_rate = t_fail / t_total * 100 if t_total else 0.0
            t_lat = (
                (view_tools["average_latency"] * view_tools["total_calls"]).sum() / t_total if t_total else 0.0
            )

            real_tools = df_tools[~df_tools["is_testing"]].sort_values("failed_calls", ascending=False)
            if not real_tools.empty and real_tools.iloc[0]["failed_calls"] > 0:
                worst_text = (
                    f"Most from {pretty_name(real_tools.iloc[0]['tool_name'])} "
                    f"({int(real_tools.iloc[0]['failed_calls'])})"
                )
            else:
                worst_text = "No failures recorded"

            kpi_grid([
                _kpi("wrench", "Total Tool Calls", f"{t_total:,}", "",
                     "Testing tools excluded" if exclude_testing else "Includes testing tools"),
                _kpi("check-circle", "Successful Calls", f"{t_ok:,}", "ok", f"{100 - t_rate:.1f}% succeeded"),
                _kpi("x-circle", "Failed Calls", f"{t_fail:,}", "bad" if t_fail else "ok", worst_text),
                _kpi("gauge", "Failure Rate", f"{t_rate:.1f}%",
                     "ok" if t_rate < 5 else ("warn" if t_rate < 15 else "bad"), "Healthy below 5%"),
                _kpi("timer", "Avg. Tool Latency", f"{t_lat:.2f}s", "", "Weighted by call count", "purple"),
            ])

            heading("wrench", "Tool health")
            ordered_tools = pd.concat([
                df_tools[~df_tools["is_testing"]].sort_values("success_rate"),
                df_tools[df_tools["is_testing"]],
            ])
            st.markdown(
                '<div class="hc-grid">'
                + "".join(tool_card_html(row) for row in ordered_tools.to_dict("records"))
                + "</div>",
                unsafe_allow_html=True,
            )

            heading("bar-chart", "Reliability and speed")
            chart_left, chart_right = st.columns(2, gap="medium")

            with chart_left, st.container(border=True):
                _title("Successful vs failed calls")
                tool_status = view_tools.melt(
                    id_vars="tool_name", value_vars=["successful_calls", "failed_calls"],
                    var_name="result", value_name="calls",
                )
                tool_status["result"] = tool_status["result"].map(
                    {"successful_calls": "Successful", "failed_calls": "Failed"}
                )
                tool_status["tool_label"] = tool_status["tool_name"].apply(pretty_name)
                fig_tool_status = px.bar(
                    tool_status, x="tool_label", y="calls", color="result", barmode="group",
                    color_discrete_map={"Successful": KIND_COLORS["success"], "Failed": KIND_COLORS["error"]},
                    labels={"tool_label": "Tool", "calls": "Calls", "result": "Result"},
                )
                show_chart(_dark(fig_tool_status, 320), key="tools_success_failed")

            with chart_right, st.container(border=True):
                _title("Average latency by tool (s)")
                latency_df = view_tools.copy()
                latency_df["tool_label"] = latency_df["tool_name"].apply(pretty_name)
                fig_tool_latency = px.bar(
                    latency_df.sort_values("average_latency"), x="average_latency", y="tool_label",
                    orientation="h", color="average_latency", color_continuous_scale=[CYAN, PURPLE],
                    labels={"tool_label": "Tool", "average_latency": "Seconds"},
                    text=latency_df.sort_values("average_latency")["average_latency"].round(2),
                )
                fig_tool_latency.update_coloraxes(showscale=False)
                fig_tool_latency.update_traces(textposition="outside", cliponaxis=False)
                show_chart(_dark(fig_tool_latency, 320), key="tools_latency")

            with st.expander("View raw breakdown table", icon=":material/table:"):
                raw_rows = [
                    [
                        html.escape(pretty_name(r["tool_name"])),
                        f'{int(r["total_calls"]):,}', f'{int(r["successful_calls"]):,}',
                        f'{int(r["failed_calls"]):,}', rate_cell(r["success_rate"]),
                        f'{r["failure_rate"]:.2f}%', f'{r["average_latency"]:.3f}s',
                    ]
                    for r in df_tools.to_dict("records")
                ]
                st.markdown(
                    _table(["Tool", "Total", "Successful", "Failed", "Success Rate", "Failure Rate", "Avg Latency"],
                           raw_rows, 320),
                    unsafe_allow_html=True,
                )

        heading("list", "Recent tool executions")

        if df_tool_events.empty:
            st.info("No tool calls yet. Ask the Tool Agent to do something and its executions will show up here.")
        else:
            f1, f2, f3 = st.columns(3)
            tool_names_present = sorted(df_tool_events["tool_name"].dropna().unique().tolist())
            status_present = sorted(df_tool_events["status"].dropna().unique().tolist())
            sel_tool = f1.selectbox("Tool", ["All"] + tool_names_present,
                                    format_func=lambda x: "All tools" if x == "All" else pretty_name(x),
                                    key="tool_evt_tool")
            sel_status = f2.selectbox("Status", ["All"] + status_present,
                                      format_func=lambda x: "All statuses" if x == "All" else pretty_status(x),
                                      key="tool_evt_status")
            row_limit = f3.selectbox("Rows to show", [25, 50, 100], key="tool_evt_limit")

            shown_events = [
                e for e in tool_events
                if (sel_tool == "All" or e.get("tool_name") == sel_tool)
                and (sel_status == "All" or e.get("status") == sel_status)
            ][:row_limit]

            if not shown_events:
                st.info("No executions match these filters.")
            else:
                event_rows = []
                for e in shown_events:
                    lat = e.get("latency")
                    event_rows.append([
                        html.escape(format_timestamp(e.get("timestamp"), compact=True)),
                        html.escape(pretty_name(e.get("tool_name"))),
                        pill(e.get("status")),
                        f"{float(lat):.2f}s" if lat not in (None, "") else "—",
                        html.escape(str(e.get("error_type"))) if e.get("error_type") else "—",
                        html.escape(short_id(e.get("trace_id"))),
                        _short(e.get("user_input"), 60),
                    ])
                st.markdown(
                    _table(["Time", "Tool", "Status", "Latency", "Error Type", "Trace", "User Input"],
                           event_rows, 340),
                    unsafe_allow_html=True,
                )

                event_options = {}
                for index, event in enumerate(shown_events, start=1):
                    label = (
                        f"{pretty_name(event.get('tool_name'))} · {pretty_status(event.get('status'))} · "
                        f"{format_timestamp(event.get('timestamp'))} · #{index}"
                    )
                    event_options[label] = event

                heading("search", "Inspect an execution")
                selected_event_label = st.selectbox(
                    "Execution", list(event_options.keys()), key="central_tool_event_select",
                    label_visibility="collapsed",
                )
                selected_event = event_options[selected_event_label]

                with st.container(border=True):
                    d1, d2 = st.columns([2, 3], gap="large")
                    with d1:
                        st.markdown(
                            f"**{pretty_name(selected_event.get('tool_name'))}** &nbsp; "
                            f"{pill(selected_event.get('status'))}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            mini_stats([
                                ("Latency", f"{float(selected_event.get('latency') or 0):.4f}s"),
                                ("When", format_timestamp(selected_event.get("timestamp"), False)),
                            ]),
                            unsafe_allow_html=True,
                        )
                        st.caption("Trace ID")
                        st.code(selected_event.get("trace_id", "N/A"))
                        st.caption("User input")
                        st.write(selected_event.get("user_input") or "N/A")
                        if selected_event.get("error_message"):
                            st.error(f"{selected_event.get('error_type')}: {selected_event.get('error_message')}")
                        if selected_event.get("injected_fault"):
                            st.warning(f"Injected fault: {selected_event['injected_fault']}")
                    with d2:
                        st.caption("Tool input")
                        st.json(selected_event.get("input_data") or {})
                        st.caption("Tool output")
                        st.json(selected_event.get("output_data") or {})

    # ----------------------------------------------
    # RUN HISTORY
    # ----------------------------------------------

    with tab_history:
        if df_runs_all.empty:
            st.info("No agent runs found yet. Run an agent from the Agent Lab first.")
        else:
            hf1, hf2, hf3, hf4 = st.columns([1.2, 1.2, 2, 1.2])
            agent_values = ["All"] + sorted(df_runs_all["agent_name"].dropna().unique().tolist())
            status_values = ["All"] + sorted(df_runs_all["status"].dropna().unique().tolist())
            window_hours = {"All time": None, "Last hour": 1, "Last 24 hours": 24,
                            "Last 7 days": 168, "Last 30 days": 720}

            chosen_agent = hf1.selectbox(
                "Agent", agent_values,
                format_func=lambda x: "All agents" if x == "All" else pretty_name(x),
                key="monitor_agent_filter",
            )
            chosen_status = hf2.selectbox(
                "Status", status_values,
                format_func=lambda x: "All statuses" if x == "All" else pretty_status(x),
                key="monitor_status_filter",
            )
            search_text = hf3.text_input("Search user input", placeholder="Type to search prompts...",
                                         key="history_search")
            chosen_window = hf4.selectbox("Time range", list(window_hours.keys()), key="history_window")

            hist = df_runs_all.copy()
            hist["ts"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
            if chosen_agent != "All":
                hist = hist[hist["agent_name"] == chosen_agent]
            if chosen_status != "All":
                hist = hist[hist["status"] == chosen_status]
            if search_text.strip():
                hist = hist[hist["user_input"].astype(str).str.contains(search_text.strip(), case=False,
                                                                         na=False, regex=False)]
            if window_hours[chosen_window]:
                hist = hist[hist["ts"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=window_hours[chosen_window])]
            hist = hist.sort_values("ts", ascending=False)

            h_failed = int((hist["kind"] == "error").sum())
            h_lat = hist["latency_done"].dropna()
            h_tokens = int(
                pd.to_numeric(hist["total_input_tokens"], errors="coerce").fillna(0).sum()
                + pd.to_numeric(hist["total_output_tokens"], errors="coerce").fillna(0).sum()
            )

            kpi_grid([
                _kpi("list", "Matching Runs", f"{len(hist):,}", "",
                     f"of {len(df_runs_all):,} loaded"
                     + (f" · {totals['total']:,} in database" if totals and totals["from_database"] else "")),
                _kpi("x-circle", "Failed", f"{h_failed:,}", "bad" if h_failed else "ok", "in this selection"),
                _kpi("timer", "Avg. Latency", f"{h_lat.mean():.2f}s" if not h_lat.empty else "N/A", "",
                     "completed runs only", "purple"),
                _kpi("coins", "Tokens", fmt_compact(h_tokens), "", "input + output"),
            ])

            if hist.empty:
                st.info("No agent runs match the selected filters.")
            else:
                history_rows = []
                for r in hist.head(150).to_dict("records"):
                    tokens_row = sum(
                        0 if pd.isna(pd.to_numeric(r.get(k), errors="coerce"))
                        else pd.to_numeric(r.get(k), errors="coerce")
                        for k in ("total_input_tokens", "total_output_tokens")
                    )
                    lat = r.get("latency_done")
                    history_rows.append([
                        html.escape(format_timestamp(r.get("timestamp"), compact=True)),
                        f'<span class="chip">{html.escape(pretty_name(r.get("agent_name")))}</span>',
                        _short(r.get("user_input"), 80),
                        pill(r.get("status")),
                        f"{lat:.2f}s" if pd.notna(lat) else "—",
                        f"{int(tokens_row):,}",
                        html.escape(short_id(r.get("trace_id"))),
                    ])
                st.markdown(
                    _table(["Time", "Agent", "User Input", "Status", "Latency", "Tokens", "Trace"],
                           history_rows, 480),
                    unsafe_allow_html=True,
                )

                cap_col, dl_col = st.columns([4, 1])
                cap_col.caption(
                    f"Showing {min(150, len(hist))} of {len(hist)} matching runs "
                    f"(latest 200 runs of each agent are loaded"
                    + (f"; {totals['total']:,} runs in the database)." if totals and totals["from_database"] else ").")
                )
                dl_col.download_button(
                    "Export CSV",
                    data=hist.drop(columns=["ts"]).to_csv(index=False).encode("utf-8"),
                    file_name="agent_runs.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    key="history_export",
                    width="stretch",
                )

        if not df_legacy.empty:
            with st.expander("Legacy execution history", icon=":material/inventory_2:"):
                legacy_rows = [
                    [
                        html.escape(format_timestamp(r.get("timestamp"), compact=True)),
                        html.escape(pretty_name(r.get("agent_name"))),
                        _short(r.get("input"), 70),
                        html.escape(str(r.get("model") or "—")),
                        html.escape(str(r.get("latency") if r.get("latency") is not None else "—")),
                        html.escape(str(r.get("tokens") if r.get("tokens") is not None else "—")),
                        pill(r.get("status")),
                    ]
                    for r in df_legacy.to_dict("records")[:100]
                ]
                st.markdown(
                    _table(["Time", "Agent", "Input", "Model", "Latency", "Tokens", "Status"], legacy_rows, 320),
                    unsafe_allow_html=True,
                )

    # ----------------------------------------------
    # TRACE EXPLORER
    # ----------------------------------------------

    with tab_trace:
        if df_runs_all.empty:
            st.info("No agent runs found yet. Run an agent from the Agent Lab first.")
        else:
            tf1, tf2, tf3, tf4 = st.columns([2, 2, 1, 1])
            trace_agent = tf1.selectbox(
                "Agent", ["All"] + sorted(df_runs_all["agent_name"].dropna().unique().tolist()),
                format_func=lambda x: "All agents" if x == "All" else pretty_name(x),
                key="trace_agent_filter",
            )
            trace_status = tf2.selectbox(
                "Status", ["All"] + sorted(df_runs_all["status"].dropna().unique().tolist()),
                format_func=lambda x: "All statuses" if x == "All" else pretty_status(x),
                key="trace_status_filter",
            )

            t_runs = df_runs_all
            if trace_agent != "All":
                t_runs = t_runs[t_runs["agent_name"] == trace_agent]
            if trace_status != "All":
                t_runs = t_runs[t_runs["status"] == trace_status]

            if t_runs.empty:
                st.info("No runs match these filters.")
            else:
                trace_options = {}
                for run_item in t_runs.to_dict("records"):
                    label = (
                        f"{pretty_name(run_item.get('agent_name'))} · "
                        f"{format_timestamp(run_item.get('timestamp'))} · "
                        f"{pretty_status(run_item.get('status'))} · "
                        f"{short_id(run_item.get('trace_id'))}"
                    )
                    trace_options[label] = run_item.get("trace_id")
                trace_labels = list(trace_options.keys())

                if st.session_state.get("monitor_trace_select") not in trace_labels:
                    st.session_state["monitor_trace_select"] = trace_labels[0]

                def _step_trace(delta, labels=trace_labels):
                    current = st.session_state.get("monitor_trace_select")
                    idx = labels.index(current) if current in labels else 0
                    st.session_state["monitor_trace_select"] = labels[(idx + delta) % len(labels)]

                tf3.markdown('<div style="height:1.7rem"></div>', unsafe_allow_html=True)
                tf3.button("Newer", icon=":material/chevron_left:", key="trace_newer",
                           on_click=_step_trace, args=(-1,), width="stretch")
                tf4.markdown('<div style="height:1.7rem"></div>', unsafe_allow_html=True)
                tf4.button("Older", icon=":material/chevron_right:", key="trace_older",
                           on_click=_step_trace, args=(1,), width="stretch")

                selected_label = st.selectbox("Execution", trace_labels, key="monitor_trace_select")
                st.caption(
                    f"Run {trace_labels.index(selected_label) + 1} of {len(trace_labels)} matching runs "
                    "· newest first. You can also type part of a trace ID to search."
                )
                trace_to_show = trace_options[selected_label]

                try:
                    detail = api_get(f"/agent-runs/{trace_to_show}")
                    run_detail = detail.get("run", {})
                    events = detail.get("events", [])

                    llm_calls = sum(1 for e in events if e.get("event_type") == "llm")
                    tool_calls = sum(1 for e in events if e.get("event_type") == "tool")
                    token_total = (
                        (run_detail.get("total_input_tokens") or 0)
                        + (run_detail.get("total_output_tokens") or 0)
                    )

                    st.markdown(
                        '<div class="tr-card"><div class="tr-grid">'
                        f'<div><div class="ao-label">Agent</div><div class="ao-value">'
                        f'{html.escape(pretty_name(run_detail.get("agent_name")))} '
                        f'{pill(run_detail.get("status", "unknown"))}</div></div>'
                        f'<div><div class="ao-label">Started</div><div class="ao-value">'
                        f'{html.escape(format_timestamp(run_detail.get("timestamp")))}</div></div>'
                        f'<div><div class="ao-label">Duration</div><div class="ao-value">'
                        f'{float(run_detail.get("total_latency") or 0):.3f}s</div></div>'
                        f'<div><div class="ao-label">Trace ID</div>'
                        f'<span class="trace-chip">{html.escape(short_id(run_detail.get("trace_id"), 13))}</span></div>'
                        '</div><div class="ao-label" style="margin-top:.9rem">User input</div>'
                        f'<div class="tr-quote">{html.escape(str(run_detail.get("user_input") or "N/A"))}</div></div>',
                        unsafe_allow_html=True,
                    )

                    kpi_grid([
                        _kpi("timer", "Total Latency", f"{float(run_detail.get('total_latency') or 0):.3f}s"),
                        _kpi("cpu", "LLM Calls", llm_calls, "", "", "purple"),
                        _kpi("wrench", "Tool Calls", tool_calls),
                        _kpi("coins", "Total Tokens", f"{token_total:,}", "", "input + output", "purple"),
                    ])

                    st.caption("Full trace ID (use the copy icon)")
                    st.code(run_detail.get("trace_id", "N/A"))

                    if run_detail.get("error_message"):
                        st.error(
                            f"{pretty_name(run_detail.get('error_type') or 'Error')}: "
                            f"{run_detail['error_message']}"
                        )

                    with st.expander("Final output", expanded=True, icon=":material/task_alt:"):
                        st.markdown(run_detail.get("final_output") or "No final output recorded.")

                    heading("activity", "Execution timeline")

                    if not events:
                        st.info("No execution events recorded for this run.")
                    else:
                        df_events = pd.DataFrame(events)
                        df_events["step"] = df_events.apply(
                            lambda row: f"{row['sequence_number']}. {pretty_name(row['event_name'])}", axis=1
                        )
                        df_events["type_label"] = df_events["event_type"].apply(pretty_name)
                        df_events["latency_value"] = pd.to_numeric(df_events["latency"], errors="coerce").fillna(0)

                        slowest_seq = None
                        if df_events["latency_value"].sum() > 0:
                            slow_row = df_events.loc[df_events["latency_value"].idxmax()]
                            slowest_seq = slow_row["sequence_number"]
                            st.markdown(
                                f'<div class="callout">Slowest step: <b>{html.escape(str(slow_row["step"]))}</b> '
                                f'took {slow_row["latency_value"]:.3f}s '
                                f'({slow_row["latency_value"] / df_events["latency_value"].sum() * 100:.0f}% '
                                "of measured time).</div>",
                                unsafe_allow_html=True,
                            )
                            with st.container(border=True):
                                fig_timeline = px.bar(
                                    df_events, x="latency_value", y="step", orientation="h",
                                    color="type_label",
                                    color_discrete_sequence=[PURPLE, CYAN, AMBER, "#22c55e"],
                                    hover_data={"latency_value": ":.4f", "status": True, "step": False},
                                    labels={"latency_value": "Latency (seconds)", "step": "Step", "type_label": "Type"},
                                )
                                fig_timeline.update_yaxes(autorange="reversed")
                                show_chart(
                                    _dark(fig_timeline, max(260, 34 * len(df_events) + 110)),
                                    key=f"timeline_{trace_to_show}",
                                )

                        for event in events:
                            icon = {
                                "success": ":material/check_circle:", "error": ":material/cancel:",
                                "running": ":material/hourglass_top:", "warn": ":material/warning:",
                            }.get(status_kind(event.get("status")), ":material/circle:")
                            is_slowest = slowest_seq is not None and event.get("sequence_number") == slowest_seq
                            event_title = (
                                f"{icon} Step {event.get('sequence_number', '?')} · "
                                f"{pretty_name(event.get('event_type', 'event'))} · "
                                f"{pretty_name(event.get('event_name', 'event'))} · "
                                f"{float(event.get('latency') or 0):.2f}s"
                                f"{' · slowest' if is_slowest else ''}"
                            )

                            with st.expander(event_title):
                                st.markdown(
                                    f"{pill(event.get('status', 'unknown'))} &nbsp; "
                                    f"<small>{html.escape(format_timestamp(event.get('timestamp')))}</small>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    mini_stats([
                                        ("Latency", f"{float(event.get('latency') or 0):.3f}s"),
                                        ("Input tokens", event.get("input_tokens") or 0),
                                        ("Output tokens", event.get("output_tokens") or 0),
                                        ("Max tokens", event.get("max_tokens") or "N/A"),
                                        ("Input chars", event.get("input_chars") or "N/A"),
                                    ]),
                                    unsafe_allow_html=True,
                                )
                                st.caption(
                                    f"Model: {event.get('model') or 'N/A'} · Temperature: "
                                    f"{event.get('temperature') if event.get('temperature') is not None else 'N/A'}"
                                    f" · Finish reason: {event.get('finish_reason') or 'N/A'}"
                                )

                                io_left, io_right = st.columns(2)
                                with io_left:
                                    st.write("**Input**")
                                    json_or_text(event.get("input_data"), "No input recorded.")
                                with io_right:
                                    st.write("**Output**")
                                    json_or_text(event.get("output_data"), "No output recorded.")

                                if event.get("status") == "error":
                                    st.error(
                                        f"{pretty_name(event.get('error_type', 'Error'))}: "
                                        f"{event.get('error_message') or 'No error message'}"
                                    )
                                if event.get("injected_fault"):
                                    st.warning(f"Injected fault: {event['injected_fault']}")

                except requests.RequestException as error:
                    st.error(f"Could not load trace details: {error}")


# ==================================================
# AGENT WORKSPACE
# ==================================================

elif page == "Agent Lab":

    page_hero(
        "layers",
        "Launch an agent",
        "Agent Lab",
        "Pick an agent, give it a task and watch it work. Every execution is traced and available in Monitoring.",
        badge="Every run is traced",
    )

    workspace_labels = {
        "Tool Agent": ":material/build: Tool Agent",
        "Research Agent": ":material/travel_explore: Research Agent",
        "Document Q&A": ":material/description: Document Q&A",
    }
    workspace = st.segmented_control(
        "Choose an agent",
        list(workspace_labels.keys()),
        format_func=lambda option: workspace_labels[option],
        default="Tool Agent",
        key="agent_workspace_choice",
        label_visibility="collapsed",
    ) or "Tool Agent"

    PANEL_HEIGHT = 640

    # Streamlit focuses the chat box when it appears and the page jumps to the bottom.
    # Once per agent tab: drop the focus and scroll back to the top.
    if st.session_state.get("_scrolled_for") != workspace:
        st.session_state["_scrolled_for"] = workspace
        # Hide the page for a moment while it is put back at the top, so the
        # input box stays pinned at the bottom but the page never visibly jumps.
        st.markdown(
            """<style>
            .stMainBlockContainer, [data-testid="stMainBlockContainer"] {
                opacity: 0; animation: ao-reveal .25s ease-out .65s forwards;
            }
            @keyframes ao-reveal {from {opacity: 0;} to {opacity: 1;}}
            </style>""",
            unsafe_allow_html=True,
        )
        script = """<script>
                const d = window.parent.document;
                const go = () => {
                  const a = d.activeElement;
                  if (a && a.tagName === 'TEXTAREA') a.blur();
                  const m = d.querySelector('[data-testid="stAppScrollToBottomContainer"]') || d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main');
                  if (m) m.scrollTo({top: 0});
                };
                [0, 50, 120, 220, 340, 460, 580, 700, 900, 1300].forEach(t => setTimeout(go, t));
                </script>"""
        try:
            if hasattr(st, "iframe"):          # newer Streamlit
                st.iframe(script, height=1)
            else:                              # older Streamlit
                components.html(script, height=0)
        except Exception:   # never break the page over a scroll helper
            pass

    # --------------------------------------------------
    # TOOL AGENT WORKSPACE
    # --------------------------------------------------

    if workspace == "Tool Agent":
        workspace_banner(
            "wrench",
            "Tool Agent",
            "Describe a task in plain language. The agent picks the right tools, "
            "runs them and combines the results into one answer.",
            [tool["name"] for tool in TOOL_CATALOG],
        )

        chat_prompt = st.chat_input(
            "Describe a task for Tool Agent...",
            key="tool_agent_chat_input",
        )
        queued_prompt = st.session_state.pop("tool_queued_prompt", None)
        prompt = chat_prompt or queued_prompt

        chat_col, side_col = st.columns([3, 2], gap="medium")

        with side_col:
            side_box = st.container(height=PANEL_HEIGHT, border=True)
            with side_box:
                tab_quick, tab_catalog = st.tabs([
                    ":material/bolt: Quick prompts",
                    ":material/construction: Tools",
                ])

                with tab_quick:
                    st.caption("Click a prompt to run it right away.")
                    for index, example in enumerate(TOOL_EXAMPLES):
                        if st.button(example, key=f"tool_example_{index}", width="stretch"):
                            st.session_state.tool_queued_prompt = example
                            st.rerun()
                    st.divider()
                    if st.button("Clear conversation", icon=":material/delete:",
                                 key="clear_tool_chat", width="stretch"):
                        st.session_state.tool_chat_history = []
                        st.rerun()
                    st.caption(
                        "Every request is recorded as an agent run. Use Monitoring to inspect "
                        "history, tool calls, errors and traces."
                    )

                with tab_catalog:
                    st.caption("The agent chooses tools by itself. No manual selection needed.")
                    for tool in TOOL_CATALOG:
                        with st.expander(f"{tool['name']} · {tool['kind']}"):
                            st.write(tool["purpose"])
                            st.caption(tool["capabilities"])
                            st.markdown(
                                f"<small><b>Try:</b> {html.escape(tool['example'])}</small>",
                                unsafe_allow_html=True,
                            )

        with chat_col:
            chat_box = st.container(height=PANEL_HEIGHT, border=True)
            with chat_box:
                if not st.session_state.tool_chat_history and not prompt:
                    empty_state(
                        "wrench",
                        "What should the agent do?",
                        "Type a task below or pick a quick prompt on the right.",
                    )

                for message in st.session_state.tool_chat_history:
                    render_tool_message(message)

                if prompt:
                    st.session_state.tool_chat_history.append({
                        "role": "user",
                        "content": prompt,
                    })

                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        try:
                            from agentops.agents.tool.agent import run_tool_agent

                            with st.spinner("Understanding your request and running tools..."):
                                result = run_tool_agent(prompt)

                            answer = (
                                result.get("answer")
                                or result.get("error")
                                or "The agent did not return a response."
                            )
                            executed_tools = result.get("executed_tools", [])
                            tool_names = [
                                pretty_name(item.get("tool_name", "Tool"))
                                for item in executed_tools
                            ]
                            trace_id = result.get("trace_id")
                            run_status = result.get("status", "unknown")

                            st.markdown(answer)
                            render_run_meta(run_status, tool_names, trace_id, executed_tools)

                            st.session_state.tool_chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "status": run_status,
                                "tools": tool_names,
                                "trace_id": trace_id,
                                "execution_details": executed_tools,
                            })

                        except Exception as error:
                            error_message = str(error)
                            st.error(f"Tool Agent could not complete the task: {error_message}")
                            st.caption(
                                "Make sure Streamlit is launched from the project root "
                                "with the project's virtual environment activated."
                            )
                            st.session_state.tool_chat_history.append({
                                "role": "assistant",
                                "content": f"Tool Agent error: {error_message}",
                            })

    # --------------------------------------------------
    # RESEARCH AGENT WORKSPACE
    # --------------------------------------------------

    elif workspace == "Research Agent":
        workspace_banner(
            "search",
            "Research Agent",
            "Investigate any topic from the web or from your own links. "
            "The agent gathers sources, extracts evidence, writes a cited report "
            "and has a critic check every claim.",
            ["Web search", "Direct URLs", "Cited report", "Critic review"],
        )

        with st.container(border=True):
            form_col, info_col = st.columns([3, 2], gap="large")

            with form_col:
                st.markdown("##### :material/edit_note: Research brief")
                research_topic = st.text_area(
                    "What do you want to research?",
                    placeholder="e.g. How does RAG work? Explain architecture, benefits and limitations.",
                    height=120,
                    key="research_agent_topic",
                )

                research_mode = st.radio(
                    "Where should the agent look?",
                    ["Web Search", "Direct URLs"],
                    horizontal=True,
                    key="research_agent_mode",
                )

                research_urls = ""
                if research_mode == "Direct URLs":
                    research_urls = st.text_area(
                        "URLs (one per line)",
                        placeholder="https://example.com/article",
                        height=120,
                        key="research_agent_urls",
                    )

                run_clicked = st.button(
                    "Run Research",
                    icon=":material/rocket_launch:",
                    type="primary",
                    key="run_research_agent",
                    width="stretch",
                )

            with info_col:
                st.markdown(
                    '<div class="steps-panel">'
                    '<div class="ag-title" style="margin-bottom:.4rem">How it works</div>'
                    + steps_html(RESEARCH_STEPS)
                    + "</div>",
                    unsafe_allow_html=True,
                )

        if run_clicked:
            if not research_topic.strip():
                st.warning("Enter a research topic first.")
            elif research_mode == "Direct URLs" and not research_urls.strip():
                st.warning("Add at least one URL to use Direct URLs mode.")
            else:
                try:
                    from agentops.agents.research.pipeline import run_research_pipeline

                    url_list = [
                        item.strip()
                        for item in research_urls.splitlines()
                        if item.strip()
                    ]

                    with st.spinner(
                        "Researching: collecting sources, analyzing evidence, "
                        "writing the report and checking citations..."
                    ):
                        st.session_state.research_result = run_research_pipeline(
                            topic=research_topic.strip(),
                            mode="url" if research_mode == "Direct URLs" else "search",
                            urls=url_list if research_mode == "Direct URLs" else None,
                        )

                except Exception as error:
                    st.session_state.research_result = None
                    st.error(f"Research Agent failed: {error}")
                    st.caption(
                        "Check the backend terminal for the full traceback. "
                        "The research pipeline must be importable from the project root."
                    )

        if st.session_state.get("research_result"):
            st.divider()
            render_research_result(st.session_state.research_result)
        else:
            st.caption(
                "Every research run is recorded centrally. Use Monitoring → Run History "
                "and Trace Explorer to inspect each stage."
            )

    # --------------------------------------------------
    # DOCUMENT INTELLIGENCE WORKSPACE
    # --------------------------------------------------

    elif workspace == "Document Q&A":
        workspace_banner(
            "file-text",
            "Document Q&A",
            "Upload a file and ask questions in plain language. Answers are grounded "
            "in your document using retrieval-augmented generation, with sources.",
            ["PDF", "DOCX", "TXT", "CSV", "XLSX", "PPTX"],
        )

        question = None
        if st.session_state.document_id:
            question = st.chat_input(
                "Ask something about your document...",
                key="document_chat_input",
            ) or st.session_state.pop("doc_queued_question", None)
        else:
            st.session_state.pop("doc_queued_question", None)

        left_col, right_col = st.columns([2, 3], gap="medium")

        with left_col:
            left_box = st.container(height=PANEL_HEIGHT, border=True)
            with left_box:
                st.markdown("##### :material/upload_file: Upload a document")

                flash = st.session_state.pop("doc_flash", None)
                if flash:
                    st.success(flash)

                uploaded_file = st.file_uploader(
                    "Upload your document",
                    type=["pdf", "docx", "txt", "csv", "xlsx", "pptx"],
                    key="document_uploader",
                    label_visibility="collapsed",
                )

                if uploaded_file is not None:
                    if st.button(
                        "Upload & Process Document",
                        icon=":material/bolt:",
                        type="primary",
                        key="upload_document_button",
                        width="stretch",
                    ):
                        try:
                            with st.spinner("Reading and indexing your document..."):
                                response = requests.post(
                                    f"{API_URL}/documents/upload",
                                    files={
                                        "file": (
                                            uploaded_file.name,
                                            uploaded_file.getvalue(),
                                            uploaded_file.type,
                                        )
                                    },
                                    timeout=180,
                                )
                                response.raise_for_status()
                                result = response.json()

                                st.session_state.document_id = result["document_id"]
                                st.session_state.document_name = result["filename"]
                                st.session_state.document_chat_history = []

                            st.session_state.doc_flash = (
                                f"Document ready. {result.get('chunks_stored', 0)} chunks indexed."
                            )
                            st.rerun()

                        except requests.RequestException as error:
                            st.error(f"Upload failed: {error}")

                st.divider()

                if st.session_state.document_id:
                    st.markdown("##### :material/attach_file: Active document")
                    st.markdown(
                        f'<span class="chip cyan">{html.escape(str(st.session_state.document_name))}</span>',
                        unsafe_allow_html=True,
                    )
                    asked = sum(
                        1 for m in st.session_state.document_chat_history if m["role"] == "user"
                    )
                    st.caption(f"{asked} question(s) asked in this session")

                    st.markdown("**Suggested questions**")
                    for index, suggestion in enumerate(DOC_SUGGESTIONS):
                        if st.button(suggestion, key=f"doc_suggest_{index}", width="stretch"):
                            st.session_state.doc_queued_question = suggestion
                            st.rerun()

                    if st.button("Remove document", icon=":material/close:",
                                 key="remove_document", width="stretch"):
                        st.session_state.document_id = None
                        st.session_state.document_name = None
                        st.session_state.document_chat_history = []
                        st.rerun()
                else:
                    st.markdown("##### :material/tune: How it works")
                    st.markdown(steps_html(DOC_STEPS), unsafe_allow_html=True)

        with right_col:
            chat_box = st.container(height=PANEL_HEIGHT, border=True)
            with chat_box:
                if not st.session_state.document_id:
                    empty_state(
                        "file-text",
                        "No document yet",
                        "Upload and process a file on the left, then ask questions about it here.",
                    )
                else:
                    if not st.session_state.document_chat_history and not question:
                        empty_state(
                            "message-square",
                            "Ask your first question",
                            f"Ask anything about {st.session_state.document_name}.",
                        )

                    for message in st.session_state.document_chat_history:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                            if message.get("sources"):
                                with st.expander("View sources"):
                                    for source in message["sources"]:
                                        st.write(f"- {source}")

                    if question:
                        st.session_state.document_chat_history.append({
                            "role": "user",
                            "content": question,
                        })

                        with st.chat_message("user"):
                            st.markdown(question)

                        with st.chat_message("assistant"):
                            try:
                                with st.spinner("Searching the document and writing an answer..."):
                                    response = requests.post(
                                        f"{API_URL}/documents/ask",
                                        json={
                                            "question": question,
                                            "document_id": st.session_state.document_id,
                                        },
                                        timeout=180,
                                    )
                                    response.raise_for_status()
                                    result = response.json()

                                answer = result.get("answer", "No answer returned.")
                                sources = list(dict.fromkeys(result.get("sources", [])))

                                st.markdown(answer)

                                if sources:
                                    with st.expander("View sources"):
                                        for source in sources:
                                            st.write(f"- {source}")

                                st.caption(
                                    f"Retrieved {result.get('retrieved_chunks', 0)} chunks · "
                                    f"Retrieval {result.get('retrieval_latency', 0)}s · "
                                    f"Generation {result.get('generation_latency', 0)}s"
                                )

                                st.session_state.document_chat_history.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "sources": sources,
                                })

                            except requests.RequestException as error:
                                st.error(f"Question failed: {error}")