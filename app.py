"""
╔══════════════════════════════════════════════════════════════════╗
║         LAKTATTEST-KALKYLATOR  –  Endurance Lab                  ║
║         Stateless (ingen data lagras) · Privacy by Design        ║
╚══════════════════════════════════════════════════════════════════╝

Starta appen:  streamlit run app.py
"""

import io
import json
import math
import base64
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import google.generativeai as genai
from scipy.interpolate import CubicSpline
from fpdf import FPDF

# ──────────────────────────────────────────────
# LÖSENORD FÖR ÅTKOMST
# ──────────────────────────────────────────────
CLINIC_PASSWORD = "oktisarenärbättreänhagaby"

# ──────────────────────────────────────────────
# SIDKONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Laktattest-kalkylator | Endurance Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS  –  ljust, professionellt tema
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Bakgrund ── */
.stApp {
    background: #f4f6fa;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e6ef;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #374151 !important;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea {
    background: #f9fafb !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* ── Huvudyta – text ── */
.stApp h1 {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.2rem;
}
.stApp h2 {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    font-weight: 600;
    font-size: 1.35rem;
    margin-top: 8px;
}
.stApp h3 {
    color: #374151 !important;
    -webkit-text-fill-color: #374151 !important;
    font-weight: 500;
}
p, label, span, div {
    color: #374151;
}

/* ── Divider ── */
hr { border-color: #e2e6ef; margin: 16px 0; }

/* ── Metrikkort ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e6ef;
    border-radius: 14px;
    padding: 20px 22px;
    margin: 6px 0;
    box-shadow: 0 2px 12px rgba(79, 70, 229, 0.07);
    transition: box-shadow 0.2s ease;
}
.metric-card:hover {
    box-shadow: 0 6px 24px rgba(79, 70, 229, 0.13);
}
.metric-card h3 {
    color: #6b7280 !important;
    -webkit-text-fill-color: #6b7280 !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 0 0 6px 0;
    font-weight: 600;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    margin: 2px 0 4px 0;
    line-height: 1.1;
}
.metric-card .sub {
    color: #9ca3af;
    font-size: 0.8rem;
    font-weight: 500;
}

/* ── LT-färger ── */
.lt1-color { color: #059669; }   /* djup grön */
.lt2-color { color: #ea580c; }   /* djup orange */

/* ── Knappar ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #6d28d9 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 1rem;
    font-weight: 600;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    width: 100%;
    letter-spacing: 0.2px;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.45);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── PDF-ladda-ner-knapp ── */
.pdf-btn > button {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 4px 14px rgba(5, 150, 105, 0.35) !important;
}
.pdf-btn > button:hover {
    box-shadow: 0 8px 22px rgba(5, 150, 105, 0.45) !important;
}

/* ── Input-fält ── */
.stNumberInput input,
.stTextInput input {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
}
.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}

/* ── Dataframe / tabell ── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e6ef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* ── Info/varning/success-boxar ── */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ── Välkoms-ikon-kort ── */
.welcome-card {
    background: #ffffff;
    border: 1px solid #e2e6ef;
    border-radius: 14px;
    padding: 22px 18px;
    width: 170px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(79,70,229,0.07);
    transition: transform 0.2s, box-shadow 0.2s;
    display: inline-block;
    margin: 8px;
}
.welcome-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(79,70,229,0.13);
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #ffffff;
    border: 1px solid #e2e6ef;
    border-radius: 10px;
    color: #374151 !important;
    font-weight: 500;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    color: #111827 !important;
    border-radius: 8px !important;
}

/* ── Section-block bakgrunder ── */
.section-box {
    background: #ffffff;
    border: 1px solid #e2e6ef;
    border-radius: 14px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ALGORITMER
# ══════════════════════════════════════════════════════════════════

def berakna_lt1(x_interp: np.ndarray, y_interp: np.ndarray, baslinje: float, delta: float = 0.5):
    """
    LT1 – Aerob tröskel (Baslinje + Fast Stegring)
    ─────────────────────────────────────────────────
    Identifiera baslinjen (lägsta laktatet i testets inledning).
    LT1 = den punkt där laktatet har stigit med exakt 'delta' (standard 0,5 mmol/L)
    över baslinjen.

    Returnerar: (x_lt1, y_lt1) eller None om tröskeln ej nås.
    """
    target = baslinje + delta
    for i in range(len(y_interp) - 1):
        if y_interp[i] <= target <= y_interp[i + 1]:
            # Linjär interpolation för exakt korsning
            frac = (target - y_interp[i]) / (y_interp[i + 1] - y_interp[i])
            x_lt1 = x_interp[i] + frac * (x_interp[i + 1] - x_interp[i])
            return float(x_lt1), float(target)
    return None


def berakna_lt1_loglog(x_raw: np.ndarray, lac_raw: np.ndarray,
                       cs_lac: 'CubicSpline', cs_hr: 'CubicSpline') -> tuple | None:
    """
    LT1 – Aerob tröskel via Log-Log-metoden (Beaver et al., 1985)
    ──────────────────────────────────────────────────────────────
    Princip:
      Laktatkurvan uppvisar ett tydligt beteende i log-log-rymden:
      en nästan horisontell fas (baslinjefas) följt av en brant uppgång
      (stegringsfas). LT1 definieras som skärningspunkten mellan de två
      bäst anpassade räta linjerna i detta logaritmiska koordinatsystem.

    Algoritm:
      1. Konvertera rådata till bas-10 logaritmer:
             log_X = log10(effekt/fart)   log_Y = log10(laktat)
      2. Utför 'Piecewise Linear Regression' (PLR):
         Iterera igenom alla möjliga brott-index k (minst 2 punkter per segment).
         För varje k – anpassa en linjär regression på [0..k] och [k..end].
         Beräkna totalt RSS (Residual Sum of Squares) för de två linjerna.
         Välj det k som minimerar det totala RSS → optimalt brott-index.
      3. Beräkna skärningspunkten (X_int, Y_int) för de två räta linjerna
         i det logaritmiska planet:
             Linje i: Y = a1*X + b1   (baslinjefas)
             Linje ii: Y = a2*X + b2  (stegringsfas)
             X_int = (b2 - b1) / (a1 - a2)
      4. Konvertera tillbaka till vanliga enheter:
             LT1_x = 10**X_int
      5. Hämta laktat och puls vid LT1_x via Cubic Spline-interpolation.

    Returnerar: (lt1_x, lt1_y, lt1_hr) eller None om beräkning misslyckas.
    """
    try:
        # ── 1. Logaritmisk transformation ────────────────────────────────────
        # Filtrera bort nollor/negativa värden för att undvika log10(0) = -inf
        valid = (x_raw > 0) & (lac_raw > 0)
        if np.sum(valid) < 4:
            return None  # Behöver minst 4 giltiga punkter för PLR

        x_v   = x_raw[valid]
        lac_v = lac_raw[valid]
        n     = len(x_v)

        log_X = np.log10(x_v)    # log10(Effekt eller Fart)
        log_Y = np.log10(lac_v)  # log10(Laktat)

        # ── 2. Piecewise Linear Regression (PLR) ─────────────────────────────
        # Vi testar varje möjligt brott-index k (2 ≤ k ≤ n-2)
        # för att garantera minst 2 punkter i varje segment.
        best_rss  = np.inf
        best_k    = 2
        best_fit  = None  # (a1, b1, a2, b2)

        for k in range(2, n - 1):
            # Segment 1: baslinjefas [0 .. k] (k+1 punkter)
            X1, Y1 = log_X[:k+1], log_Y[:k+1]
            # Minsta-kvadrat för segment 1
            coeffs1 = np.polyfit(X1, Y1, 1)   # [lutning, intercept]
            a1, b1  = coeffs1[0], coeffs1[1]
            resid1  = Y1 - (a1 * X1 + b1)
            rss1    = float(np.dot(resid1, resid1))

            # Segment 2: stegringsfas [k .. n-1] (n-k punkter)
            X2, Y2 = log_X[k:], log_Y[k:]
            coeffs2 = np.polyfit(X2, Y2, 1)
            a2, b2  = coeffs2[0], coeffs2[1]
            resid2  = Y2 - (a2 * X2 + b2)
            rss2    = float(np.dot(resid2, resid2))

            total_rss = rss1 + rss2
            if total_rss < best_rss:
                best_rss = total_rss
                best_k   = k
                best_fit = (a1, b1, a2, b2)

        if best_fit is None:
            return None

        a1, b1, a2, b2 = best_fit

        # ── 3. Skärningspunkt i log-log-rymden ───────────────────────────────
        # Linje 1: log_Y = a1 * log_X + b1
        # Linje 2: log_Y = a2 * log_X + b2
        # Vid skärning: a1*X + b1 = a2*X + b2  =>  X = (b2 - b1) / (a1 - a2)
        denom_ll = a1 - a2
        if abs(denom_ll) < 1e-10:
            return None  # Parallella linjer – ingen skärningspunkt

        log_X_int = (b2 - b1) / denom_ll
        # log_Y_int = a1 * log_X_int + b1  (kontroll)

        # ── 4. Konvertera tillbaka till vanliga enheter ───────────────────────
        # LT1_x är skärningspunktens X-värde i den ursprungliga skalan
        lt1_x = 10 ** log_X_int

        # Sanitetskontroll: LT1 bör ligga inom testintervallet
        if lt1_x < x_v[0] or lt1_x > x_v[-1]:
            return None

        # ── 5. Hämta laktat och puls via Cubic Spline ─────────────────────────
        lt1_y  = float(cs_lac(lt1_x))
        lt1_hr = float(cs_hr(lt1_x))

        return float(lt1_x), float(lt1_y), float(lt1_hr)

    except Exception:
        return None


def berakna_lt2_dmax(x_raw: np.ndarray, y_raw: np.ndarray,
                     x_interp: np.ndarray, y_interp: np.ndarray):
    """
    LT2 – Anaerob tröskel via D-max-metoden
    ────────────────────────────────────────
    Steg 1: Dra en rät linje från (x_raw[0], y_raw[0]) till (x_raw[-1], y_raw[-1]).
    Steg 2: Beräkna det vinkelräta (Euklidiska) avståndet från varje punkt på
            den interpolerade kurvan till den räta linjen.
    Steg 3: D-max-punkten är den interpolerade punkt med störst avstånd.
            Det är LT2.

    Formel för punktavstånd till linje Ax + By + C = 0:
        d = |Ax₀ + By₀ + C| / sqrt(A² + B²)

    Returnerar: (x_lt2, y_lt2) eller None.
    """
    x1, y1 = x_raw[0], y_raw[0]
    x2, y2 = x_raw[-1], y_raw[-1]

    # Linjens ekvation på formen Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    denom = math.sqrt(A**2 + B**2)
    if denom == 0:
        return None  # Alla punkter på samma ställe – degenerat test

    # Beräkna avstånd för varje interpolerad punkt
    distances = np.abs(A * x_interp + B * y_interp + C) / denom

    idx_max = int(np.argmax(distances))
    return float(x_interp[idx_max]), float(y_interp[idx_max])


def berakna_fbla(x_fine: np.ndarray, y_lac_fine: np.ndarray,
                 cs_hr, target_lac: float) -> tuple | None:
    """
    FBLA – Fixed Blood Lactate Accumulation
    ─────────────────────────────────────────
    Hittar X (Effekt/Fart) där den interpolerade laktatkurvan passerar ett givet
    fast laktatvärde (t.ex. 2.0, 2.5, 3.0, 3.5 eller 4.0 mmol/L).
    Används även för Initial Rise (baslinje + 1.0 mmol/L).

    Returnerar: (x_val, target_lac, hr) eller None om värdet ej nås.
    """
    for i in range(len(y_lac_fine) - 1):
        if y_lac_fine[i] <= target_lac <= y_lac_fine[i + 1]:
            frac  = (target_lac - y_lac_fine[i]) / (y_lac_fine[i + 1] - y_lac_fine[i])
            x_val = float(x_fine[i] + frac * (x_fine[i + 1] - x_fine[i]))
            try:
                hr = float(cs_hr(x_val))
            except Exception:
                hr = None
            return x_val, float(target_lac), hr
    return None


def berakna_lt2_modified_dmax(x_raw: np.ndarray, lac_raw: np.ndarray,
                               x_fine: np.ndarray, y_fine: np.ndarray,
                               lt1_x: float, lt1_y: float) -> tuple | None:
    """
    LT2 – Modified D-max
    ─────────────────────
    Identisk med klassisk D-max men konstruktionslinjen dras från den beräknade
    LT1-punkten (Baslinje+0.5) till sista mätpunkten, istället från allra ersten.
    Ger en mer fysiologiskt träffsäker LT2 (Bishop et al., 1998).

    Steg 1: Konstruktionslinje: LT1-punkt → sista uppmätta punkt.
    Steg 2: Beräkna vinkelrätt avstånd för varje interpolerad punkt EFTER lt1_x.
    Steg 3: Modified D-max = punkten med störst avstånd.

    Returnerar: (x_lt2, y_lt2) eller None.
    """
    x1, y1 = lt1_x, lt1_y
    x2, y2 = float(x_raw[-1]), float(lac_raw[-1])

    # Begränsa till interpolerade punkter efter lt1_x
    mask  = x_fine >= lt1_x
    x_seg = x_fine[mask]
    y_seg = y_fine[mask]

    if len(x_seg) < 2:
        return None

    A     = y2 - y1
    B     = x1 - x2
    C     = x2 * y1 - x1 * y2
    denom = math.sqrt(A**2 + B**2)
    if denom == 0:
        return None

    distances = np.abs(A * x_seg + B * y_seg + C) / denom
    idx_max   = int(np.argmax(distances))
    return float(x_seg[idx_max]), float(y_seg[idx_max])


def interpolera_puls(x_interp_fine: np.ndarray, x_raw: np.ndarray,
                     hr_raw: np.ndarray, x_target: float) -> float | None:
    """
    Hämta HR vid ett givet x-värde via kubisk spline på pulsdata.
    Returnerar float eller None om data saknas / otillräckliga punkter.
    """
    try:
        if len(x_raw) < 2:
            return None
        mask = ~np.isnan(hr_raw.astype(float))
        xv, yv = x_raw[mask], hr_raw.astype(float)[mask]
        if len(xv) < 2:
            return None
        cs_hr = CubicSpline(xv, yv)
        val = float(cs_hr(x_target))
        return val
    except Exception:
        return None

def watt_to_500m_pace(watt: float) -> str:
    """Konverterar Watt till /500m tempo för Concept2."""
    if watt is None or watt <= 0 or np.isnan(watt):
        return ""
    sekunder = 500 * (2.8 / float(watt))**(1/3)
    mm = int(sekunder // 60)
    ss = int(sekunder % 60)
    return f"{mm:02d}:{ss:02d} /500m"



def berakna_zoner(x_lt1: float, x_lt2: float,
                  hr_lt1: float | None, hr_lt2: float | None,
                  x_unit: str,
                  zon_system: str,
                  is_concept2: bool,
                  cs_lac, cs_hr, x_raw) -> pd.DataFrame:
    def fmt_x(lo, hi=None):
        if x_unit == "min/km" or x_unit == "m/s":
            # Lägre värde = snabbare för min/km (inverterade gränser)
            lo_s = f"{lo:.2f}" if lo is not None else "–"
            hi_s = f"{hi:.2f}" if hi is not None else "–"
            if hi is None: return f"< {lo_s}" if x_unit != "min/km" else f"> {lo_s}"
            if lo is None: return f"> {hi_s}" if x_unit != "min/km" else f"< {hi_s}"
            return f"{lo_s} – {hi_s}" if x_unit != "min/km" else f"{hi_s} – {lo_s}"
        else:
            lo_s = f"{lo:.0f}" if lo is not None else "–"
            hi_s = f"{hi:.0f}" if hi is not None else "–"

            c2_lo = f" ({watt_to_500m_pace(lo)})" if is_concept2 and lo is not None else ""
            c2_hi = f" ({watt_to_500m_pace(hi)})" if is_concept2 and hi is not None else ""

            if hi is None: return f"> {lo_s}{c2_lo}"
            if lo is None: return f"< {hi_s}{c2_hi}"
            return f"{lo_s}{c2_lo} – {hi_s}{c2_hi}"

    def fmt_hr(lo, hi=None):
        if lo is None and hi is None: return "–"
        lo_s = f"{lo:.0f}" if lo is not None else "–"
        hi_s = f"{hi:.0f}" if hi is not None else "–"
        if hi is None: return f"> {lo_s} bpm"
        if lo is None: return f"< {hi_s} bpm"
        return f"{lo_s} – {hi_s} bpm"

    hr1 = hr_lt1 if hr_lt1 is not None else 0
    hr2 = hr_lt2 if hr_lt2 is not None else 0

    if "Olympiatoppen" in zon_system or "Klassisk" in zon_system:
        zoner = [
            {"Zon": "🔵 Zon 1 – Återhämtning", x_unit: fmt_x(None, x_lt1 * 0.85), "Puls": fmt_hr(None, hr1 * 0.85), "Intensitet": "< 85 % av LT1"},
            {"Zon": "🟢 Zon 2 – Aerob bas", x_unit: fmt_x(x_lt1 * 0.85, x_lt1), "Puls": fmt_hr(hr1 * 0.85, hr1), "Intensitet": "85–100 % av LT1"},
            {"Zon": "🟡 Zon 3 – Tempo", x_unit: fmt_x(x_lt1, x_lt2 * 0.95), "Puls": fmt_hr(hr1, hr2 * 0.95), "Intensitet": "LT1 – 95 % av LT2"},
            {"Zon": "🟠 Zon 4 – Tröskel", x_unit: fmt_x(x_lt2 * 0.95, x_lt2 * 1.05), "Puls": fmt_hr(hr2 * 0.95, hr2 * 1.05), "Intensitet": "95–105 % av LT2"},
            {"Zon": "🔴 Zon 5 – VO₂max", x_unit: fmt_x(x_lt2 * 1.05, None), "Puls": fmt_hr(hr2 * 1.05, None), "Intensitet": "> 105 % av LT2"},
        ]
    elif "Polariserad" in zon_system:
        zoner = [
            {"Zon": "🔵 Zon 1 (Låg) – Aerob", x_unit: fmt_x(None, x_lt1), "Puls": fmt_hr(None, hr1), "Intensitet": "< LT1"},
            {"Zon": "🟡 Zon 2 (Mellan) – Tröskel", x_unit: fmt_x(x_lt1, x_lt2), "Puls": fmt_hr(hr1, hr2), "Intensitet": "LT1 – LT2"},
            {"Zon": "🔴 Zon 3 (Hög) – Anaerob", x_unit: fmt_x(x_lt2, None), "Puls": fmt_hr(hr2, None), "Intensitet": "> LT2"},
        ]
    elif "Effektbaserad" in zon_system:
        ftp = x_lt2
        zoner = [
            {"Zon": "Zon 1 - Recovery", x_unit: fmt_x(None, ftp * 0.55), "Puls": "–", "Intensitet": "< 55% FTP"},
            {"Zon": "Zon 2 - Endurance", x_unit: fmt_x(ftp * 0.55, ftp * 0.75), "Puls": "–", "Intensitet": "55-75% FTP"},
            {"Zon": "Zon 3 - Tempo", x_unit: fmt_x(ftp * 0.76, ftp * 0.87), "Puls": "–", "Intensitet": "76-87% FTP"},
            {"Zon": "Sweetspot", x_unit: fmt_x(ftp * 0.88, ftp * 0.94), "Puls": "–", "Intensitet": "88-94% FTP"},
            {"Zon": "Zon 4 - Threshold", x_unit: fmt_x(ftp * 0.95, ftp * 1.05), "Puls": "–", "Intensitet": "95-105% FTP"},
            {"Zon": "Zon 5 - VO2max", x_unit: fmt_x(ftp * 1.06, None), "Puls": "–", "Intensitet": "> 105% FTP"},
        ]
    elif "Norsk" in zon_system:
        def find_x(target_lac):
            x_test = np.linspace(x_raw[0], x_raw[-1], 2000)
            y_test = cs_lac(x_test)
            diffs = np.abs(y_test - target_lac)
            idx = np.argmin(diffs)
            if diffs[idx] > 0.4: return None
            return float(x_test[idx])
            
        x_23 = find_x(2.3)
        hr_23 = float(cs_hr(x_23)) if x_23 else None
        x_30 = find_x(3.0)
        hr_30 = float(cs_hr(x_30)) if x_30 else None
        
        def s_x(v): return f"{v:.1f}" + (f" ({watt_to_500m_pace(v)})" if is_concept2 else "") if v else "Saknas"
        def s_hr(v): return f"{v:.0f} bpm" if v else "Saknas"
        
        zoner = [
            {"Zon": "Lugn Distans", x_unit: f"< {s_x(x_lt1)}", "Puls": f"< {s_hr(hr1)}", "Intensitet": "Under LT1"},
            {"Zon": "Låg Tröskel", x_unit: f"~ {s_x(x_23)}", "Puls": f"~ {s_hr(hr_23)}", "Intensitet": "2.3 mmol/L"},
            {"Zon": "Hög Tröskel", x_unit: f"~ {s_x(x_30)}", "Puls": f"~ {s_hr(hr_30)}", "Intensitet": "3.0 mmol/L"},
            {"Zon": "LT2 / MLSS Tak", x_unit: f"~ {s_x(x_lt2)}", "Puls": f"~ {s_hr(hr2)}", "Intensitet": "D-max punkt"},
        ]
        
    return pd.DataFrame(zoner)


# ══════════════════════════════════════════════════════════════════
# PDF-GENERERING  (FPDF2)
# ══════════════════════════════════════════════════════════════════

# Tecken-mappning – Helvetica latin-1 stöder ej Unicode-specialtecken
_UNICODE_MAP = str.maketrans({
    "\u2013": "-",    # en-dash
    "\u2014": "--",   # em-dash
    "\u2019": "'",    # höger apostrof
    "\u2018": "'",    # vänster apostrof
    "\u201c": '"',    # vänster citattecken
    "\u201d": '"',    # höger citattecken
    "\u2026": "...",  # ellipsis
    "\u2082": "2",    # subscript 2
    "\u00b2": "2",    # superscript 2
    "\u00b0": " gr",  # gradtecken
    "\u2192": "->",   # pil höger
    "\u00d7": "x",    # multiplikationstecken
})

def sanitize(text: str) -> str:
    """Gör texten Latin-1-säker för FPDF2/Helvetica."""
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(_UNICODE_MAP)
    # Ta bort kvarvarande icke-Latin-1-tecken (t.ex. emoji)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


class LaktatRapport(FPDF):
    """Premium PDF-layout: vit bakgrund, tunna linjer, minimal bläckanvändning."""

    def __init__(self, klient: str, testdatum: str, protokoll: str):
        super().__init__()
        self.klient    = klient
        self.testdatum = testdatum
        self.protokoll = protokoll
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(14, 14, 14)

    # ── Hjälpare ──────────────────────────────────────────────────────────
    def hline(self, y=None, r=180, g=180, b=180, lw=0.3):
        """Rita en tunn horisontell linje tvärs över textytan."""
        if y is None:
            y = self.get_y()
        self.set_draw_color(r, g, b)
        self.set_line_width(lw)
        self.line(14, y, 196, y)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.2)

    def accent_bar(self, h=0.8, r=79, g=70, b=229):
        """Tunn lila accentlinje ovanför en sektion – ingen stor fyllning."""
        y = self.get_y()
        self.set_fill_color(r, g, b)
        self.rect(14, y, 182, h, "F")
        self.ln(h + 2)

    def sektionsrubrik(self, text: str):
        """Sektion: tunn accentlinje + fetstilt rubrik, ingen bakgrundsfyllning."""
        self.ln(4)
        self.accent_bar()
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 60)
        self.cell(0, 7, sanitize(text.upper()), ln=True)
        self.set_text_color(30, 30, 30)
        self.ln(1)

    def kv_rad(self, etikett: str, varde: str, col_w=48):
        """En rad med etikett (grå) + värde (svart)."""
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 130)
        self.cell(col_w, 7, sanitize(etikett), border=0)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(20, 20, 30)
        self.cell(0, 7, sanitize(varde), border=0, ln=True)
        self.set_text_color(30, 30, 30)

    # ── Sidhuvud ──────────────────────────────────────────────────────────
    def header(self):
        # Övre vit yta – bara text och en tunn linje längst ner
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 40)
        self.set_xy(14, 10)
        self.cell(120, 8, "ENDURANCE LAB", border=0)

        # Höger: "Laktattest-rapport" i grått
        self.set_font("Helvetica", "", 9)
        self.set_text_color(130, 130, 140)
        self.set_xy(134, 12)
        self.cell(62, 6, "Laktattest-rapport", border=0, align="R", ln=True)

        # Tunn skiljelinje
        self.hline(y=20, r=200, g=200, b=210, lw=0.4)
        self.ln(6)
        self.set_text_color(30, 30, 30)

    # ── Sidfot ────────────────────────────────────────────────────────────
    def footer(self):
        self.set_y(-14)
        self.hline(r=210, g=210, b=215, lw=0.3)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(160, 160, 170)
        txt = sanitize(
            f"Sida {self.page_no()}/{{nb}}  |  "
            f"Genererad {date.today().strftime('%Y-%m-%d')}  |  "
            f"Konfidentiellt - ej for spridning"
        )
        self.cell(0, 6, txt, align="C")


def generera_pdf(
    klient: str,
    testdatum: str,
    protokoll: str,
    coach_kommentar: str,
    ai_analys: str,
    lt1_x: float,
    lt1_y: float,
    lt1_hr: float | None,
    lt2_x: float,
    lt2_y: float,
    lt2_hr: float | None,
    fatmax_x: float | None,
    fatmax_hr: float | None,
    x_unit: str,
    is_concept2: bool,
    df_zoner: pd.DataFrame,
    df_profil: pd.DataFrame,
    raw_df: pd.DataFrame,
    graf_bytes: bytes,
    ai_zon_reasoning: str = "",
) -> bytes:
    """Bygger den premiumformade PDF-rapporten och returnerar den som bytes."""
    import tempfile, os

    pdf = LaktatRapport(klient, testdatum, protokoll)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── 1. Metadata-block ─────────────────────────────────────────────────
    pdf.sektionsrubrik("Testinformation")
    col_w = 38
    pdf.set_font("Helvetica", "", 9)

    # Rad 1: Klient + Datum
    pdf.set_text_color(120, 120, 130)
    pdf.cell(col_w, 7, "Klient", border=0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(20, 20, 30)
    pdf.cell(72, 7, sanitize(klient), border=0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 130)
    pdf.cell(30, 7, "Datum", border=0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(20, 20, 30)
    pdf.cell(0, 7, sanitize(testdatum), border=0, ln=True)

    # Rad 2: Protokoll
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 130)
    pdf.cell(col_w, 7, "Protokoll", border=0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(20, 20, 30)
    pdf.cell(0, 7, sanitize(protokoll), border=0, ln=True)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(3)

    # ── 2. Trösklar – premium 2-kolumn layout ────────────────────────────
    pdf.sektionsrubrik("Tröskelresultat")

    def troskel_box(x_pos, y_pos, rubrik, x_val, lac_val, hr_val, accent_rgb):
        """Ritar en vit box med tunn accentkant för en tröskel."""
        w, h = 60, 30
        # Vit bakgrund + tunn ram
        pdf.set_fill_color(255, 255, 255)
        pdf.set_draw_color(*accent_rgb)
        pdf.set_line_width(0.4)
        pdf.rect(x_pos, y_pos, w, h, "D")
        # Accentlinje vänsterkant
        pdf.set_fill_color(*accent_rgb)
        pdf.rect(x_pos, y_pos, 1.5, h, "F")
        pdf.set_line_width(0.2)
        # Rubrik
        pdf.set_xy(x_pos + 4, y_pos + 3)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*accent_rgb)
        pdf.cell(w - 5, 5, sanitize(rubrik), border=0, ln=True)
        # Värden
        pdf.set_xy(x_pos + 4, y_pos + 9)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 20, 30)
        # concept2 pace tillägg
        pace_str = f" ({watt_to_500m_pace(x_val)})" if is_concept2 and x_val else ""
        pdf.cell(w - 5, 8, sanitize(f"{x_val:.1f} {x_unit}{pace_str}"), border=0, ln=True)
        pdf.set_xy(x_pos + 4, y_pos + 18)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 100, 110)
        hr_s = f"{hr_val:.0f} bpm" if hr_val else "-"
        lac_s = f"Laktat: {lac_val:.2f} mmol/L"
        pdf.cell(w - 5, 5, sanitize(f"{lac_s}   {hr_s}"), border=0, ln=True)
        pdf.set_text_color(30, 30, 30)

    y_start = pdf.get_y()
    troskel_box(14,  y_start, "LT1 - Aerob troskel",    lt1_x, lt1_y, lt1_hr, (5, 150, 105))
    if fatmax_x is not None:
        troskel_box(76, y_start, "FatMax", fatmax_x, lt1_y, fatmax_hr, (16, 185, 129))
    troskel_box(138, y_start, "LT2 - Anaerob", lt2_x, lt2_y, lt2_hr, (234, 88, 12))
    pdf.set_xy(14, y_start + 34)

    # ── 3. Graf ───────────────────────────────────────────────────────────
    pdf.sektionsrubrik("Laktat- och pulsprofil")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(graf_bytes)
    tmp.close()
    try:
        pdf.image(tmp.name, x=14, w=182)
    finally:
        os.unlink(tmp.name)
    pdf.ln(4)

    # ── 4. Fysiologisk Profil (sammanställning av alla modeller) ────────────
    if not df_profil.empty:
        pdf.sektionsrubrik("Fysiologisk profil (sammanstallning)")
        prof_cols   = list(df_profil.columns)
        prof_widths = [64, 60, 40]
        # Rubrikrad
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(80, 80, 90)
        pdf.set_draw_color(200, 200, 205)
        pdf.set_line_width(0.25)
        for h, w in zip(prof_cols, prof_widths):
            pdf.cell(w, 7, sanitize(h), border="B", align="L")
        pdf.ln()
        pdf.set_text_color(30, 30, 30)
        for _, row in df_profil.iterrows():
            pdf.set_font("Helvetica", "", 8)
            for col, w in zip(prof_cols, prof_widths):
                pdf.cell(w, 6.5, sanitize(str(row[col])), border="B", align="L")
            pdf.ln()
        pdf.ln(5)

    # ── 5. AI-optimerade trösklar (om tillgängligt) ───────────────────────
    if ai_zon_reasoning and ai_zon_reasoning.strip():
        pdf.sektionsrubrik("AI-optimerade trosklar")
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(60, 60, 80)
        pdf.multi_cell(0, 5.5, sanitize(ai_zon_reasoning), border=0)
        pdf.ln(3)

    # ── 6. Träningszoner – ren tabell med tunna linjer ───────────────────
    if not df_zoner.empty:
        pdf.sektionsrubrik("Traningszoner")
        zon_cols   = list(df_zoner.columns)
        zon_widths = [58, 40, 48, 36]
        # Adjust widths to match number of columns
        while len(zon_widths) < len(zon_cols):
            zon_widths.append(36)
        zon_widths = zon_widths[:len(zon_cols)]

        # Rubrikrad
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(80, 80, 90)
        pdf.set_draw_color(200, 200, 205)
        pdf.set_line_width(0.25)
        for h, w in zip(zon_cols, zon_widths):
            pdf.cell(w, 7, sanitize(h), border="B", align="L")
        pdf.ln()
        pdf.set_text_color(30, 30, 30)
        for _, row in df_zoner.iterrows():
            pdf.set_font("Helvetica", "", 8)
            for col, w in zip(zon_cols, zon_widths):
                pdf.cell(w, 6.5, sanitize(str(row[col])), border="B", align="L")
            pdf.ln()
        pdf.ln(5)

    # ── 7. Rådata ─────────────────────────────────────────────────────────
    pdf.sektionsrubrik("Testdata (rådata)")
    raw_cols   = list(raw_df.columns)
    raw_widths = [45] * len(raw_cols)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(80, 80, 90)
    for col, w in zip(raw_cols, raw_widths):
        pdf.cell(w, 7, sanitize(col), border="B", align="C")
    pdf.ln()
    pdf.set_text_color(30, 30, 30)
    for idx, row in raw_df.iterrows():
        pdf.set_font("Helvetica", "", 8)
        for col, w in zip(raw_cols, raw_widths):
            pdf.cell(w, 6.5, sanitize(str(row[col])), border="B", align="C")
        pdf.ln()
    pdf.ln(5)

    # ── 8. Coach-kommentar & AI Analys ────────────────────────────────────
    if coach_kommentar.strip():
        pdf.sektionsrubrik("Coach-kommentar")
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(30, 30, 40)
        pdf.multi_cell(0, 6, sanitize(coach_kommentar), border=0)
        
    if ai_analys.strip():
        pdf.sektionsrubrik("AI-Coach Analys")
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(30, 30, 40)
        pdf.multi_cell(0, 6, sanitize(ai_analys), border=0)

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════
# INLOGGNINGSSKÄRM  –  Lösenordsskydd
# ══════════════════════════════════════════════════════════════════

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    # Centrerad inloggningsruta
    _login_spacer_l, _login_col, _login_spacer_r = st.columns([1, 2, 1])
    with _login_col:
        st.markdown("""
        <div style='text-align:center; padding: 60px 0 20px;'>
            <div style='font-size:4rem; margin-bottom:12px;'>🔬</div>
            <h2 style='color:#4f46e5; font-weight:700; font-size:1.8rem; margin-bottom:4px;'>Endurance Lab</h2>
            <p style='color:#6b7280; font-size:0.95rem; margin-bottom:30px;'>Laktattest-kalkylator · Professionell labbanalys</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("<p style='color:#374151; font-weight:500; margin-bottom:4px;'>Ange kliniklösenord för att fortsätta</p>", unsafe_allow_html=True)
            password_input = st.text_input("Lösenord", type="password", placeholder="Ange lösenord…", label_visibility="collapsed")
            login_btn = st.form_submit_button("🔓 Logga in", use_container_width=True)

            if login_btn:
                if password_input == CLINIC_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ Fel lösenord. Försök igen.")

        st.markdown("""
        <div style='text-align:center; margin-top:40px;'>
            <p style='color:#9ca3af; font-size:0.75rem;'>🔒 Privacy by Design · Ingen data lagras</p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════
# STREAMLIT UI  –  SIDOFÄLT (METADATA)
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔬 Endurance Lab")
    st.markdown("**Laktattest-kalkylator**")
    st.markdown("---")

    st.markdown("### 📋 Testmetadata")
    klient_namn = st.text_input("Klientens namn", placeholder="Anna Eriksson", key="klient")
    testdatum   = st.date_input("Testdatum", value=date.today(), key="datum")
    testprotokoll = st.text_input(
        "Testprotokoll",
        placeholder="t.ex. Löpband 4 min/steg",
        key="protokoll"
    )
    sport_typ = st.selectbox(
        "Typ av Ergometer/Sport",
        options=["Löpning (km/h)", "Cykling (Watt)", "Concept2 SkiErg/Rodd (Watt)"],
        key="sport_typ"
    )
    if "km/h" in sport_typ:
        x_enhet = "km/h"
        is_concept2 = False
    elif "Concept2" in sport_typ:
        x_enhet = "Watt"
        is_concept2 = True
    else:
        x_enhet = "Watt"
        is_concept2 = False

    zon_system_val = st.selectbox(
        "Välj Zonsystem för export",
        options=[
            "Klassisk 5-Zon (Olympiatoppen)",
            "Polariserad 3-Zon (Seiler)",
            "Effektbaserad 6-Zon (Coggan/Cykel)",
            "Norsk Modell (Dubbeltröskel)"
        ],
        key="zon_system"
    )


    st.markdown("---")
    st.markdown("### 🤖 AI-Coach")
    gemini_api_key = st.text_input("Gemini API-nyckel", type="password", help="Används för AI-analys av tröskelvärden via Google Gemini")
    
    st.markdown("---")
    st.markdown("### 💬 Coach-kommentar")
    coach_kommentar = st.text_area(
        "Analys och rekommendationer",
        placeholder="Skriv din analys, rekommendationer och observationer här...",
        height=200,
        key="coach_kommentar"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; text-align:center;'>
    🔒 <b>Privacy by Design</b><br>
    Ingen data lagras.<br>
    All data försvinner vid siduppdatering.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STREAMLIT UI  –  HUVUDYTA
# ══════════════════════════════════════════════════════════════════

st.markdown("# 🔬 Laktattest-kalkylator")
st.markdown("Mata in stegdata nedan, tryck på **Beräkna** och exportera din rapport.")
st.markdown("---")

# ── DATAINMATNING ──────────────────────────────────────────────────
st.markdown("## 📊 Stegdata")

info_col, _ = st.columns([3, 1])
with info_col:
    st.info("💡 Lägg till minst **4 mätsteg** för tillförlitlig kurvanpassning. Fyll i alla kolumner för bästa resultat.")

# Antal steg att visa
n_steps = st.number_input(
    "Antal steg i testet",
    min_value=3,
    max_value=20,
    value=9,
    step=1,
    key="n_steps",
    help="Välj antal steg/belastningsnivåer i ditt laktattest"
)

# ── Dynamisk inmatningstabell ──────────────────────────────────────────────
st.markdown(f"### Fyll i värden för varje steg ({x_enhet} / Puls / Laktat / Borg)")

header_cols = st.columns([2, 2, 2, 2])
header_cols[0].markdown(f"**{x_enhet}**")
header_cols[1].markdown("**Puls (bpm)**")
header_cols[2].markdown("**Laktat (mmol/L)**")
header_cols[3].markdown("**Borg (6–20)**")

# Standardvärden (Watt/cykel-exempel)
default_data = [
    [100, 115, 1.2, 10],
    [120, 123, 1.0, 11],
    [140, 131, 1.1, 12],
    [160, 139, 1.3, 13],
    [180, 148, 1.7, 14],
    [200, 157, 2.4, 15],
    [220, 165, 3.5, 16],
    [240, 174, 5.2, 18],
    [260, 182, 8.1, 19]
]

steg_data = []
for i in range(int(n_steps)):
    cols = st.columns([2, 2, 2, 2])
    d = default_data[i] if i < len(default_data) else [0, 0, 0, 0]
    with cols[0]:
        x_val = st.number_input(
            f"{x_enhet} steg {i+1}", value=float(d[0]),
            min_value=0.0, step=0.5, label_visibility="collapsed", key=f"x_{i}"
        )
    with cols[1]:
        hr_val = st.number_input(
            f"HR steg {i+1}", value=float(d[1]),
            min_value=0.0, max_value=250.0, step=1.0, label_visibility="collapsed", key=f"hr_{i}"
        )
    with cols[2]:
        lac_val = st.number_input(
            f"Laktat steg {i+1}", value=float(d[2]),
            min_value=0.0, max_value=30.0, step=0.1, label_visibility="collapsed", key=f"lac_{i}"
        )
    with cols[3]:
        borg_val = st.number_input(
            f"Borg steg {i+1}", value=float(d[3]),
            min_value=6.0, max_value=30.0, step=1.0, label_visibility="collapsed", key=f"borg_{i}"
        )
    steg_data.append([x_val, hr_val, lac_val, borg_val])

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# BERÄKNA-KNAPP
# ══════════════════════════════════════════════════════════════════

calc_col, _ = st.columns([2, 3])
with calc_col:
    berakna_tryckt = st.button("⚡ Beräkna Trösklar & Skapa Grafer", key="berakna_btn", use_container_width=True)

if berakna_tryckt or st.session_state.get("beraknat", False):
    if berakna_tryckt:
        st.session_state["beraknat"] = True
        st.session_state["steg_data_cache"] = steg_data
        st.session_state["x_enhet_cache"] = x_enhet
        st.session_state["sport_typ_cache"] = sport_typ
        st.session_state["zon_system_cache"] = zon_system_val
        st.session_state["is_concept2_cache"] = is_concept2

    steg_data_use   = st.session_state.get("steg_data_cache", steg_data)
    x_enhet_use     = st.session_state.get("x_enhet_cache", x_enhet)
    is_concept2_use = st.session_state.get("is_concept2_cache", is_concept2)
    zon_system_use  = st.session_state.get("zon_system_cache", zon_system_val)

    # ── Bygg DataFrames ────────────────────────────────────────────
    cols_names = [x_enhet_use, "Puls (bpm)", "Laktat (mmol/L)", "Borg (6-20)"]
    df = pd.DataFrame(steg_data_use, columns=cols_names)

    x_raw   = df[x_enhet_use].values.astype(float)
    hr_raw  = df["Puls (bpm)"].values.astype(float)
    lac_raw = df["Laktat (mmol/L)"].values.astype(float)

    # Validering
    if len(x_raw) < 3:
        st.error("❌ Minst 3 steg krävs för beräkning.")
        st.stop()

    if np.any(np.diff(x_raw) <= 0):
        st.error("❌ Intensitetsvärden (X-axeln) måste vara strikt stigande. Kontrollera din data.")
        st.stop()

    # ── Kubisk Spline Interpolation ───────────────────────────────
    cs_lac = CubicSpline(x_raw, lac_raw)
    cs_hr  = CubicSpline(x_raw, hr_raw)

    # 1000 punkter för mjuk kurva
    x_fine = np.linspace(x_raw[0], x_raw[-1], 1000)
    y_lac_fine = cs_lac(x_fine)
    y_hr_fine  = cs_hr(x_fine)

    # ══════════════════════════════════════════════════════════════
    # BERÄKNA ALLA FYSIOLOGISKA MODELLER
    # ══════════════════════════════════════════════════════════════
    baslinje = float(np.min(lac_raw[:2]))

    # Hjälpfunktion för att formatera X-värde (inkl. Concept2 pace)
    def _fmt_x(val):
        if val is None:
            return "–"
        s = f"{val:.1f}"
        if is_concept2_use:
            s += f" ({watt_to_500m_pace(val)})"
        return s

    def _fmt_hr(val):
        return f"{val:.0f}" if val is not None else "–"

    # ── LT1: Baslinje + 0.5 mmol/L (Standard) ────────────────────
    _lt1_std_res = berakna_lt1(x_fine, y_lac_fine, baslinje, delta=0.5)
    lt1_std_x  = _lt1_std_res[0] if _lt1_std_res else None
    lt1_std_y  = _lt1_std_res[1] if _lt1_std_res else None
    lt1_std_hr = interpolera_puls(x_fine, x_raw, hr_raw, lt1_std_x) if lt1_std_x else None

    # ── LT1: Log-Log (Beaver et al., 1985) ───────────────────────
    _lt1_ll_res = berakna_lt1_loglog(x_raw, lac_raw, cs_lac, cs_hr)
    lt1_ll_x  = _lt1_ll_res[0] if _lt1_ll_res else None
    lt1_ll_y  = _lt1_ll_res[1] if _lt1_ll_res else None
    lt1_ll_hr = _lt1_ll_res[2] if _lt1_ll_res else None

    # ── LT2: D-max klassisk (första → sista mätpunkt) ────────────
    _lt2_dmax_res = berakna_lt2_dmax(x_raw, lac_raw, x_fine, y_lac_fine)
    lt2_dmax_x  = _lt2_dmax_res[0] if _lt2_dmax_res else None
    lt2_dmax_y  = _lt2_dmax_res[1] if _lt2_dmax_res else None
    lt2_dmax_hr = interpolera_puls(x_fine, x_raw, hr_raw, lt2_dmax_x) if lt2_dmax_x else None

    # ── LT2: Modified D-max (LT1 +0.5 → sista mätpunkt) ─────────
    if lt1_std_x is not None and lt1_std_y is not None:
        _lt2_mdmax_res = berakna_lt2_modified_dmax(
            x_raw, lac_raw, x_fine, y_lac_fine, lt1_std_x, lt1_std_y
        )
    else:
        _lt2_mdmax_res = None
    lt2_mdmax_x  = _lt2_mdmax_res[0] if _lt2_mdmax_res else None
    lt2_mdmax_y  = _lt2_mdmax_res[1] if _lt2_mdmax_res else None
    lt2_mdmax_hr = interpolera_puls(x_fine, x_raw, hr_raw, lt2_mdmax_x) if lt2_mdmax_x else None

    # ── LT2: Initial Rise – Balke (Baslinje + 1.0 mmol/L) ────────
    _ir_10_res  = berakna_fbla(x_fine, y_lac_fine, cs_hr, baslinje + 1.0)
    lt2_ir10_x  = _ir_10_res[0] if _ir_10_res else None
    lt2_ir10_hr = _ir_10_res[2] if _ir_10_res else None

    # ── LT2: Initial Rise – Dickhuth (Baslinje + 1.5 mmol/L) ─────
    _ir_15_res  = berakna_fbla(x_fine, y_lac_fine, cs_hr, baslinje + 1.5)
    lt2_ir15_x  = _ir_15_res[0] if _ir_15_res else None
    lt2_ir15_hr = _ir_15_res[2] if _ir_15_res else None

    # ── FBLA (Fixed Blood Lactate Accumulation) ───────────────────
    _fbla_2_0 = berakna_fbla(x_fine, y_lac_fine, cs_hr, 2.0)
    _fbla_2_5 = berakna_fbla(x_fine, y_lac_fine, cs_hr, 2.5)
    _fbla_3_0 = berakna_fbla(x_fine, y_lac_fine, cs_hr, 3.0)
    _fbla_3_5 = berakna_fbla(x_fine, y_lac_fine, cs_hr, 3.5)
    _fbla_4_0 = berakna_fbla(x_fine, y_lac_fine, cs_hr, 4.0)

    # ── Primära variabler (för graf, FatMax, AI, PDF-boxar) ───────
    # LT1 primär = Baslinje + 0.5, LT2 primär = Modified D-max
    lt1_x, lt1_y, lt1_hr = lt1_std_x, lt1_std_y, lt1_std_hr
    lt2_x = lt2_mdmax_x if lt2_mdmax_x is not None else lt2_dmax_x
    lt2_y = lt2_mdmax_y if lt2_mdmax_x is not None else lt2_dmax_y
    lt2_hr = lt2_mdmax_hr if lt2_mdmax_x is not None else lt2_dmax_hr

    fatmax_x  = lt1_x
    fatmax_hr = lt1_hr

    # ── SAMMANSTÄLLNING: alla modeller i en dict → DataFrame ─────
    # Nyckel = metodnamn, värde = (x, hr)
    profil_rader = [
        ("LT1: Log-Log (Beaver)",              lt1_ll_x,             lt1_ll_hr),
        ("LT1: Baslinje +0.5 mmol/L",          lt1_std_x,            lt1_std_hr),
        ("LT2: D-max (klassisk)",              lt2_dmax_x,           lt2_dmax_hr),
        ("LT2: Modified D-max",                lt2_mdmax_x,          lt2_mdmax_hr),
        ("LT2: Initial Rise +1.0 (Balke)",     lt2_ir10_x,           lt2_ir10_hr),
        ("LT2: Initial Rise +1.5 (Dickhuth)",  lt2_ir15_x,           lt2_ir15_hr),
        ("FBLA 2.0 mmol/L",                    _fbla_2_0[0] if _fbla_2_0 else None, _fbla_2_0[2] if _fbla_2_0 else None),
        ("FBLA 2.5 mmol/L",                    _fbla_2_5[0] if _fbla_2_5 else None, _fbla_2_5[2] if _fbla_2_5 else None),
        ("FBLA 3.0 mmol/L",                    _fbla_3_0[0] if _fbla_3_0 else None, _fbla_3_0[2] if _fbla_3_0 else None),
        ("FBLA 3.5 mmol/L",                    _fbla_3_5[0] if _fbla_3_5 else None, _fbla_3_5[2] if _fbla_3_5 else None),
        ("FBLA 4.0 mmol/L (OBLA)",             _fbla_4_0[0] if _fbla_4_0 else None, _fbla_4_0[2] if _fbla_4_0 else None),
    ]

    x_col_name = f"Effekt / Fart ({x_enhet_use})"
    df_profil = pd.DataFrame(
        [{"Metod": m, x_col_name: _fmt_x(x), "Puls (bpm)": _fmt_hr(hr)} for m, x, hr in profil_rader]
    )

    # ── PLOTLY GRAF ──────────────────────────────────────────────
    st.markdown("## 📈 Laktat- och pulsprofil")

    fig = go.Figure()

    # Laktat kurva – djupröd, tydlig mot vit bakgrund
    fig.add_trace(go.Scatter(
        x=x_fine, y=y_lac_fine,
        name="Laktat (interpolerat)",
        line=dict(color="#dc2626", width=3),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=x_raw, y=lac_raw,
        name="Laktat (mätvärden)",
        mode="markers",
        marker=dict(color="#dc2626", size=10, symbol="circle",
                    line=dict(color="white", width=2)),
    ))

    # Puls kurva – djupblå, sekundär y-axel
    fig.add_trace(go.Scatter(
        x=x_fine, y=y_hr_fine,
        name="Puls (interpolerat)",
        line=dict(color="#2563eb", width=2.5, dash="dot"),
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=x_raw, y=hr_raw,
        name="Puls (mätvärden)",
        mode="markers",
        marker=dict(color="#2563eb", size=9, symbol="diamond",
                    line=dict(color="white", width=2)),
        yaxis="y2",
    ))

    # LT1 (+0.5) – primär grön streckad linje
    if lt1_x is not None:
        _lt1_ann = (
            f"<b>LT1 (+0.5)</b><br>{lt1_x:.1f} {x_enhet_use}<br>{lt1_hr:.0f} bpm"
            if lt1_hr else
            f"<b>LT1 (+0.5)</b><br>{lt1_x:.1f} {x_enhet_use}"
        )
        fig.add_vline(x=lt1_x, line_dash="dash", line_color="#059669", line_width=2.5,
                      annotation_text=_lt1_ann,
                      annotation_position="top right",
                      annotation_font_color="#065f46",
                      annotation_bgcolor="rgba(209,250,229,0.85)",
                      annotation_bordercolor="#059669",
                      annotation_font_size=12)

    # LT1 Log-Log (Beaver) – sekundär lila linje (visas bara om den skiljer sig)
    if lt1_ll_x is not None and (lt1_x is None or abs(lt1_ll_x - lt1_x) > 0.5):
        fig.add_vline(x=lt1_ll_x, line_dash="dot", line_color="#7c3aed", line_width=1.5,
                      annotation_text=f"<b>LT1 Log-Log</b><br>{lt1_ll_x:.1f} {x_enhet_use}",
                      annotation_position="top left",
                      annotation_font_color="#5b21b6",
                      annotation_bgcolor="rgba(237,233,254,0.85)",
                      annotation_bordercolor="#7c3aed",
                      annotation_font_size=10)

    # LT2 (primär) – orange streckad linje
    if lt2_x is not None:
        _lt2_label = "Mod. D-max" if lt2_mdmax_x is not None else "D-max"
        _lt2_ann = (
            f"<b>LT2 ({_lt2_label})</b><br>{lt2_x:.1f} {x_enhet_use}<br>{lt2_hr:.0f} bpm"
            if lt2_hr else
            f"<b>LT2 ({_lt2_label})</b><br>{lt2_x:.1f} {x_enhet_use}"
        )
        fig.add_vline(x=lt2_x, line_dash="dash", line_color="#ea580c", line_width=2.5,
                      annotation_text=_lt2_ann,
                      annotation_position="top left",
                      annotation_font_color="#7c2d12",
                      annotation_bgcolor="rgba(255,237,213,0.85)",
                      annotation_bordercolor="#ea580c",
                      annotation_font_size=12)

    # Modified D-max konstruktionslinje (LT1 → sista punkt)
    if lt1_x is not None and lt1_y is not None:
        fig.add_trace(go.Scatter(
            x=[lt1_x, x_raw[-1]],
            y=[lt1_y, lac_raw[-1]],
            name="Mod. D-max konstruktionslinje",
            line=dict(color="#9ca3af", width=1.5, dash="dot"),
            mode="lines",
            opacity=0.7,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[x_raw[0], x_raw[-1]],
            y=[lac_raw[0], lac_raw[-1]],
            name="D-max konstruktionslinje",
            line=dict(color="#9ca3af", width=1.5, dash="dot"),
            mode="lines",
            opacity=0.7,
        ))

    # LT2 punkt-markör
    if lt2_x is not None:
        fig.add_trace(go.Scatter(
            x=[lt2_x], y=[lt2_y],
            name=f"LT2 ({_lt2_label} punkt)",
            mode="markers",
            marker=dict(color="#ea580c", size=14, symbol="star",
                        line=dict(color="white", width=2)),
        ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafbff",
        font=dict(family="Inter", color="#1e293b", size=13),
        title=dict(
            text=f"<b>Laktatprofil</b>  –  {klient_namn or 'Klient'}  |  {testdatum}",
            font=dict(size=17, color="#4f46e5"),
            x=0.02,
        ),
        xaxis=dict(
            title=dict(text=x_enhet_use, font=dict(color="#374151", size=13)),
            gridcolor="#e5e7eb",
            linecolor="#d1d5db",
            showgrid=True,
            tickfont=dict(color="#374151"),
        ),
        yaxis=dict(
            title=dict(text="Laktat (mmol/L)", font=dict(color="#b91c1c", size=13)),
            gridcolor="#e5e7eb",
            linecolor="#d1d5db",
            tickfont=dict(color="#b91c1c"),
        ),
        yaxis2=dict(
            title=dict(text="Puls (bpm)", font=dict(color="#1d4ed8", size=13)),
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#1d4ed8"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#e2e6ef",
            borderwidth=1,
            font=dict(color="#374151"),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_color="#1e293b", bordercolor="#e2e6ef"),
        height=520,
        margin=dict(l=60, r=80, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Exportera graf som PNG (för PDF) ──────────────────────────
    try:
        graf_png = fig.to_image(format="png", width=1200, height=520, scale=2)
    except Exception:
        # Fallback om kaleido saknas
        graf_png = None
        st.warning("⚠️ Kaleido ej installerat – grafbilden kan inte inbäddas i PDF. Kör: pip install kaleido")

    # ══════════════════════════════════════════════════════════════
    # FYSIOLOGISK PROFIL – SAMMANSTÄLLNINGSTABELL
    # ══════════════════════════════════════════════════════════════
    st.markdown("## 🧬 Fysiologisk Profil (Sammanställning)")
    st.dataframe(df_profil, use_container_width=True, hide_index=True)

    # ── TRÖSKELRESULTAT – METRICS ─────────────────────────────────
    st.markdown("## 🎯 Tröskelresultat (Primära trösklar)")
    m1, m2, m3, m4 = st.columns(4)

    lt1_hr_str = f"{lt1_hr:.0f}" if lt1_hr is not None else "–"
    lt2_hr_str = f"{lt2_hr:.0f}" if lt2_hr is not None else "–"

    if lt1_x is not None:
        with m1:
            st.markdown(f"""
            <div class="metric-card">
              <h3>LT1 – Aerob tröskel (+0.5)</h3>
              <div class="value lt1-color">{lt1_x:.1f}</div>
              <div class="sub">{x_enhet_use}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
              <h3>LT1 – Puls</h3>
              <div class="value lt1-color">{lt1_hr_str}</div>
              <div class="sub">bpm</div>
            </div>""", unsafe_allow_html=True)

    if lt2_x is not None:
        with m3:
            st.markdown(f"""
            <div class="metric-card">
              <h3>LT2 – Anaerob tröskel (Mod. D-max)</h3>
              <div class="value lt2-color">{lt2_x:.1f}</div>
              <div class="sub">{x_enhet_use}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
              <h3>LT2 – Puls</h3>
              <div class="value lt2-color">{lt2_hr_str}</div>
              <div class="sub">bpm</div>
            </div>""", unsafe_allow_html=True)

    # ── FATMAX ────────────────────────────────────────────────────
    if fatmax_x is not None:
        st.markdown("---")
        fm_pace = f" ({watt_to_500m_pace(fatmax_x)})" if is_concept2_use else ""
        st.info(f"🔥 **FatMax (Maximal fettförbränning)** inträffar fysiologiskt vid den aeroba tröskeln (LT1). "
                f"Ditt FatMax-värde är därmed satt till **{fatmax_x:.1f} {x_enhet_use}{fm_pace}** "
                f"(Puls: **{fatmax_hr:.0f} bpm**).")

    # ══════════════════════════════════════════════════════════════
    # TRÄNINGSZONER – Manuellt val ELLER AI-optimerat val
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 🏋️ Träningszoner")

    # Bygg val-listor från profil_rader (bara de med beräknat x-värde)
    lt1_options = {m: (x, hr) for m, x, hr in profil_rader if x is not None and "LT1" in m}
    lt2_options = {m: (x, hr) for m, x, hr in profil_rader if x is not None and ("LT2" in m or "FBLA" in m)}

    # Initiera session-state för AI-zonresultat
    if "ai_zon_reasoning" not in st.session_state:
        st.session_state["ai_zon_reasoning"] = ""
    if "ai_zon_lt1" not in st.session_state:
        st.session_state["ai_zon_lt1"] = None
    if "ai_zon_lt2" not in st.session_state:
        st.session_state["ai_zon_lt2"] = None

    # ── Flik-liknande val: Manuellt vs AI ─────────────────────────
    zon_metod = st.radio(
        "Välj metod för zonberäkning",
        options=["📊 Manuellt val", "🤖 Låt AI välja optimala trösklar"],
        horizontal=True,
        key="zon_metod_radio",
    )

    df_zoner = pd.DataFrame()
    ai_zon_reasoning_display = st.session_state.get("ai_zon_reasoning", "")

    if "Manuellt" in zon_metod:
        # ── MANUELLT FLÖDE (befintligt) ──────────────────────────
        st.markdown("Välj vilka tröskelvärden som ska ligga till grund för zonberäkningen:")

        zon_c1, zon_c2 = st.columns(2)
        with zon_c1:
            valt_lt1_namn = st.selectbox(
                "LT1-modell för zoner",
                options=list(lt1_options.keys()),
                index=list(lt1_options.keys()).index("LT1: Baslinje +0.5 mmol/L") if "LT1: Baslinje +0.5 mmol/L" in lt1_options else 0,
                key="zon_lt1_val",
            )
        with zon_c2:
            valt_lt2_namn = st.selectbox(
                "LT2-modell för zoner",
                options=list(lt2_options.keys()),
                index=list(lt2_options.keys()).index("LT2: Modified D-max") if "LT2: Modified D-max" in lt2_options else 0,
                key="zon_lt2_val",
            )

        zon_btn_col, _ = st.columns([2, 3])
        with zon_btn_col:
            berakna_zoner_tryckt = st.button("📊 Beräkna träningszoner", key="berakna_zoner_btn", use_container_width=True)

        if berakna_zoner_tryckt or st.session_state.get("zoner_beraknade", False):
            if berakna_zoner_tryckt:
                st.session_state["zoner_beraknade"] = True
                st.session_state["zon_lt1_cache"] = valt_lt1_namn
                st.session_state["zon_lt2_cache"] = valt_lt2_namn
                # Rensa eventuellt AI-resonemang vid manuellt val
                st.session_state["ai_zon_reasoning"] = ""
                ai_zon_reasoning_display = ""

            _valt_lt1 = st.session_state.get("zon_lt1_cache", valt_lt1_namn)
            _valt_lt2 = st.session_state.get("zon_lt2_cache", valt_lt2_namn)

            if _valt_lt1 in lt1_options and _valt_lt2 in lt2_options:
                z_lt1_x, z_lt1_hr = lt1_options[_valt_lt1]
                z_lt2_x, z_lt2_hr = lt2_options[_valt_lt2]
                df_zoner = berakna_zoner(z_lt1_x, z_lt2_x, z_lt1_hr, z_lt2_hr, x_enhet_use, zon_system_use, is_concept2_use, cs_lac, cs_hr, x_raw)
                st.dataframe(df_zoner, use_container_width=True, hide_index=True)
                st.caption(f"Zoner baserade på **{_valt_lt1}** och **{_valt_lt2}** · Zonsystem: *{zon_system_use}*")
            else:
                st.warning("Någon av de valda modellerna saknar beräknat värde. Kontrollera data.")
        else:
            st.info("👆 Välj LT1- och LT2-modell ovan och tryck **Beräkna träningszoner**.")

    else:
        # ── AI-OPTIMERAT FLÖDE ────────────────────────────────────
        if not gemini_api_key:
            st.warning("⚠️ Fyll i din **Gemini API-nyckel** i sidofältet för att använda AI-optimerade zoner.")
        else:
            st.markdown("AI:n analyserar all rådata och alla tröskelmodeller för att välja de fysiologiskt mest rimliga värdena för LT1 och LT2.")

            ai_zon_btn_col, _ = st.columns([2, 3])
            with ai_zon_btn_col:
                ai_zon_tryckt = st.button(
                    "🤖 Låt AI välja optimala trösklar & Skapa Zoner",
                    key="ai_zon_btn",
                    use_container_width=True,
                )

            if ai_zon_tryckt:
                with st.spinner("AI analyserar tröskelmodeller…"):
                    try:
                        genai.configure(api_key=gemini_api_key)
                        ai_model = genai.GenerativeModel(
                            "gemini-3-flash-preview",
                            generation_config={"response_mime_type": "application/json"},
                        )

                        # Bygg rådata-tabell som text
                        raw_table_lines = [f"{x_enhet_use} | Puls | Laktat | Borg"]
                        for row_i in range(len(x_raw)):
                            raw_table_lines.append(
                                f"{x_raw[row_i]:.1f} | {hr_raw[row_i]:.0f} | {lac_raw[row_i]:.2f} | {df['Borg (6-20)'].iloc[row_i]:.0f}"
                            )
                        raw_table_str = "\n".join(raw_table_lines)

                        # Bygg profilsammanfattning
                        profil_text_ai = "\n".join(
                            f"  {row['Metod']}: {row[x_col_name]}, {row['Puls (bpm)']} bpm"
                            for _, row in df_profil.iterrows()
                        )

                        ai_zon_prompt = (
                            "Du är en expertfysiolog. Analysera denna atlets rådata och de matematiska "
                            "tröskelmodellerna. Din uppgift är att bedöma vilka beräkningar som är "
                            "mest rimliga för LT1 (Aerob tröskel) och LT2 (Anaerob tröskel/MLSS).\n\n"
                            "Returnera DITT VAL som ett strikt JSON-objekt med tre nycklar: "
                            "'best_lt1' (endast siffran för effekt/fart), "
                            "'best_lt2' (endast siffran för effekt/fart), och "
                            "'reasoning' (en kort text på max 100 ord där du motiverar ditt val fysiologiskt).\n\n"
                            f"=== RÅDATA ===\n{raw_table_str}\n\n"
                            f"=== BERÄKNADE MODELLER ===\n{profil_text_ai}\n\n"
                            f"Baslinje (lägsta laktat): {baslinje:.2f} mmol/L\n"
                            f"Enhet: {x_enhet_use}\n"
                        )

                        ai_zon_response = ai_model.generate_content(ai_zon_prompt)
                        ai_zon_json = json.loads(ai_zon_response.text)

                        best_lt1_val = float(ai_zon_json["best_lt1"])
                        best_lt2_val = float(ai_zon_json["best_lt2"])
                        ai_reasoning = str(ai_zon_json.get("reasoning", ""))

                        # Sanitetskontroll: värdena ska ligga inom testintervallet
                        if not (x_raw[0] <= best_lt1_val <= x_raw[-1]):
                            st.warning(f"⚠️ AI:ns LT1 ({best_lt1_val:.1f}) ligger utanför testintervallet. Använder standardvärde.")
                            best_lt1_val = lt1_x if lt1_x is not None else x_raw[0]
                        if not (x_raw[0] <= best_lt2_val <= x_raw[-1]):
                            st.warning(f"⚠️ AI:ns LT2 ({best_lt2_val:.1f}) ligger utanför testintervallet. Använder standardvärde.")
                            best_lt2_val = lt2_x if lt2_x is not None else x_raw[-1]

                        # Hämta HR via spline
                        ai_lt1_hr = float(cs_hr(best_lt1_val))
                        ai_lt2_hr = float(cs_hr(best_lt2_val))

                        # Spara i session state
                        st.session_state["ai_zon_lt1"] = best_lt1_val
                        st.session_state["ai_zon_lt2"] = best_lt2_val
                        st.session_state["ai_zon_lt1_hr"] = ai_lt1_hr
                        st.session_state["ai_zon_lt2_hr"] = ai_lt2_hr
                        st.session_state["ai_zon_reasoning"] = ai_reasoning
                        st.session_state["ai_zoner_beraknade"] = True
                        ai_zon_reasoning_display = ai_reasoning

                    except json.JSONDecodeError:
                        st.error("❌ Kunde inte tolka AI:ns JSON-svar. Försök igen.")
                    except Exception as e:
                        st.error(f"❌ AI-anrop misslyckades: {e}")

            # Visa AI-resultat (aktuellt eller från session state)
            if st.session_state.get("ai_zoner_beraknade", False):
                _ai_lt1 = st.session_state["ai_zon_lt1"]
                _ai_lt2 = st.session_state["ai_zon_lt2"]
                _ai_lt1_hr = st.session_state.get("ai_zon_lt1_hr")
                _ai_lt2_hr = st.session_state.get("ai_zon_lt2_hr")
                _ai_reason = st.session_state["ai_zon_reasoning"]

                st.success(f"🤖 **AI-Coachens bedömning:** {_ai_reason}")

                ai_m1, ai_m2 = st.columns(2)
                with ai_m1:
                    _pace_1 = f" ({watt_to_500m_pace(_ai_lt1)})" if is_concept2_use else ""
                    st.markdown(f"""
                    <div class="metric-card">
                      <h3>AI-valt LT1</h3>
                      <div class="value lt1-color">{_ai_lt1:.1f}{_pace_1}</div>
                      <div class="sub">{x_enhet_use} · {_ai_lt1_hr:.0f} bpm</div>
                    </div>""", unsafe_allow_html=True)
                with ai_m2:
                    _pace_2 = f" ({watt_to_500m_pace(_ai_lt2)})" if is_concept2_use else ""
                    st.markdown(f"""
                    <div class="metric-card">
                      <h3>AI-valt LT2</h3>
                      <div class="value lt2-color">{_ai_lt2:.1f}{_pace_2}</div>
                      <div class="sub">{x_enhet_use} · {_ai_lt2_hr:.0f} bpm</div>
                    </div>""", unsafe_allow_html=True)

                df_zoner = berakna_zoner(
                    _ai_lt1, _ai_lt2, _ai_lt1_hr, _ai_lt2_hr,
                    x_enhet_use, zon_system_use, is_concept2_use, cs_lac, cs_hr, x_raw
                )
                st.dataframe(df_zoner, use_container_width=True, hide_index=True)
                st.caption(f"Zoner baserade på **AI-optimerade trösklar** · Zonsystem: *{zon_system_use}*")
            else:
                st.info("👆 Tryck på knappen ovan för att låta AI analysera och välja optimala trösklar.")

    # ── RÅDATA PREVIEW ────────────────────────────────────────────
    with st.expander("📋 Se inmatad testdata"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── AI-COACH ANALYS ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🤖 AI-Coach Analys")

    if "ai_analys" not in st.session_state:
        st.session_state["ai_analys"] = ""

    if not gemini_api_key:
        st.info("💡 Fyll i din Gemini API-nyckel i sidofältet för att låsa upp AI-analysen.")
    elif gemini_api_key and lt1_x is not None and lt2_x is not None:
        if st.button("✨ Generera AI-Analys (Google Gemini)"):
            with st.spinner("Analyserar din laktatprofil..."):
                try:
                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel("gemini-3-flash-preview")

                    # Bygg en komplett data-summary från df_profil
                    profil_text = "\n".join(
                        f"  {row['Metod']}: {row[x_col_name]}, {row['Puls (bpm)']} bpm"
                        for _, row in df_profil.iterrows()
                    )
                    data_summary = (
                        f"Klient: {klient_namn or 'Okänd'}\n"
                        f"Testdatum: {testdatum}\n"
                        f"Protokoll: {testprotokoll or 'Okänt'}\n\n"
                        f"Testintervall: {x_raw[0]:.0f}–{x_raw[-1]:.0f} {x_enhet_use}, "
                        f"Laktat: {lac_raw[0]:.2f}–{lac_raw[-1]:.2f} mmol/L\n\n"
                        f"=== PRIMÄRA TRÖSKLAR (för zonberäkning) ===\n"
                        f"LT1 (Baslinje +0.5): {_fmt_x(lt1_x)}, {_fmt_hr(lt1_hr)} bpm\n"
                        f"LT2 (Modified D-max): {_fmt_x(lt2_x)}, {_fmt_hr(lt2_hr)} bpm\n"
                        f"FatMax: {_fmt_x(fatmax_x)}, {_fmt_hr(fatmax_hr)} bpm\n\n"
                        f"=== KOMPLETT FYSIOLOGISK PROFIL ===\n{profil_text}\n"
                    )

                    prompt = (
                        "Du är en expert-fysioterapeut och PT med inriktning på konditionsidrott. "
                        "Analysera denna atlets laktatkurva och tröskelvärden.\n\n"
                        f"{data_summary}\n"
                        "Skriv en kort, uppmuntrande och professionell analys (max 200 ord) av deras "
                        "aeroba och anaeroba profil samt ge en konkret rekommendation för träningsfokus framåt. "
                        "Undvik allmänna plattityder och var specifik kring siffrorna."
                    )
                    response = model.generate_content(prompt)
                    st.session_state["ai_analys"] = response.text
                except Exception as e:
                    st.error(f"Kunde inte generera AI-analys. Kontrollera din nyckel. Detaljer: {e}")

    if st.session_state["ai_analys"]:
        st.success(st.session_state["ai_analys"])

    # ── PDF-EXPORT ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📄 Exportera rapport")

    if graf_png and lt1_x is not None:
        pdf_col, _ = st.columns([2, 3])
        with pdf_col:
            try:
                pdf_bytes = generera_pdf(
                    klient=klient_namn or "Okänd klient",
                    testdatum=str(testdatum),
                    protokoll=testprotokoll or "–",
                    coach_kommentar=coach_kommentar,
                    ai_analys=st.session_state["ai_analys"],
                    lt1_x=lt1_x,
                    lt1_y=lt1_y,
                    lt1_hr=lt1_hr,
                    lt2_x=lt2_x,
                    lt2_y=lt2_y,
                    lt2_hr=lt2_hr,
                    fatmax_x=fatmax_x,
                    fatmax_hr=fatmax_hr,
                    x_unit=x_enhet_use,
                    is_concept2=is_concept2_use,
                    df_zoner=df_zoner,
                    df_profil=df_profil,
                    raw_df=df,
                    graf_bytes=graf_png,
                    ai_zon_reasoning=ai_zon_reasoning_display,
                )
                pdf_filnamn = f"Laktattest_{(klient_namn or 'rapport').replace(' ', '_')}_{testdatum}.pdf"
                st.markdown('<div class="pdf-btn">', unsafe_allow_html=True)
                st.download_button(
                    label="📥 Ladda ner testrapport (PDF)",
                    data=pdf_bytes,
                    file_name=pdf_filnamn,
                    mime="application/pdf",
                    use_container_width=True,
                    key="pdf_download"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                st.success(f"✅ Rapport klar: **{pdf_filnamn}**")
            except Exception as e:
                st.error(f"PDF-generering misslyckades: {e}")
    elif graf_png is None:
        st.info("💡 Installera **kaleido** (`pip install kaleido`) för att aktivera PDF-export med graf.")
    elif lt1_x is None:
        st.warning("⚠️ LT1 kunde inte beräknas – PDF inkluderar ej tröskelzoner.")

else:
    # Välkomstskärm – ljust tema
    st.markdown("""
    <div style='text-align:center; padding: 56px 20px 40px;'>
        <div style='font-size:4.5rem; margin-bottom:16px;'>🔬</div>
        <h2 style='color:#4f46e5; font-size:1.9rem; font-weight:700; margin-bottom:12px;'>
            Redo att analysera ditt laktattest
        </h2>
        <p style='color:#6b7280; font-size:1.05rem; max-width:580px; margin:0 auto; line-height:1.7;'>
            Fyll i testmetadata i sidofältet till vänster, mata in stegdata i tabellen ovan
            och tryck på <strong style='color:#4f46e5;'>Beräkna Trösklar &amp; Skapa Grafer</strong>
            för att generera analys och PDF-rapport.
        </p>
        <br><br>
        <div style='display:flex; justify-content:center; gap:20px; flex-wrap:wrap;'>
            <div class='welcome-card'>
                <div style='font-size:2.2rem;'>📐</div>
                <div style='color:#4f46e5; font-weight:600; font-size:0.78rem; margin-top:10px; text-transform:uppercase; letter-spacing:0.5px;'>Kubisk Spline</div>
                <div style='color:#6b7280; font-size:0.8rem; margin-top:4px;'>Interpolation</div>
            </div>
            <div class='welcome-card'>
                <div style='font-size:2.2rem;'>📏</div>
                <div style='color:#4f46e5; font-weight:600; font-size:0.78rem; margin-top:10px; text-transform:uppercase; letter-spacing:0.5px;'>D-max Metoden</div>
                <div style='color:#6b7280; font-size:0.8rem; margin-top:4px;'>för LT2</div>
            </div>
            <div class='welcome-card'>
                <div style='font-size:2.2rem;'>🏃</div>
                <div style='color:#4f46e5; font-weight:600; font-size:0.78rem; margin-top:10px; text-transform:uppercase; letter-spacing:0.5px;'>5 Träningszoner</div>
                <div style='color:#6b7280; font-size:0.8rem; margin-top:4px;'>Puls &amp; Effekt</div>
            </div>
            <div class='welcome-card'>
                <div style='font-size:2.2rem;'>📄</div>
                <div style='color:#4f46e5; font-weight:600; font-size:0.78rem; margin-top:10px; text-transform:uppercase; letter-spacing:0.5px;'>PDF-rapport</div>
                <div style='color:#6b7280; font-size:0.8rem; margin-top:4px;'>med inbäddad graf</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
