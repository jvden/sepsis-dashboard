"""
Sepsis Monitor — Isala IC Dashboard
Data Analytics Opdracht | Windesheim HBO-ICT x Isala Zwolle

Doel: vroegtijdige signalering van sepsis bij IC-patiënten,
      minimaal 6 uur voor klinische diagnose.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_loader import (
    load_sample,
    latest_vitals,
    missing_rate,
    sepsis_onset_hour,
    VITAL_COLS,
    LAB_COLS,
    VITAL_RANGES,
    DEMO_COLS,
    LABEL_COL,
)
from risk_score import compute_score_series, score_contributions, risk_badge

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sepsis Monitor | Isala IC",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .metric-card {
    background: #1e2130; border-radius: 8px; padding: 12px 16px;
    text-align: center; margin: 4px;
  }
  .metric-card .value { font-size: 2rem; font-weight: 700; }
  .metric-card .label { font-size: 0.8rem; color: #aaa; }
  .alert-box {
    border-left: 4px solid #dc3545;
    background: #2d1f1f;
    padding: 10px 14px;
    border-radius: 4px;
    margin: 8px 0;
  }
  .warn-box {
    border-left: 4px solid #fd7e14;
    background: #2d2010;
    padding: 10px 14px;
    border-radius: 4px;
    margin: 8px 0;
  }
  .info-box {
    border-left: 4px solid #0d6efd;
    background: #0d1f35;
    padding: 10px 14px;
    border-radius: 4px;
    margin: 8px 0;
  }
  .disclaimer {
    font-size: 0.72rem; color: #888;
    border-top: 1px solid #333; padding-top: 6px; margin-top: 10px;
  }
</style>
""", unsafe_allow_html=True)

# ── Data laden ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    df = load_sample(n_patients=600)
    return compute_score_series(df)


with st.spinner("Patiëntgegevens laden…"):
    df = get_data()

onset_map = sepsis_onset_hour(df)
last_df = compute_score_series(latest_vitals(df))

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏥 Isala Ziekenhuis")
    st.markdown("## Filters")

    unit_filter = st.multiselect(
        "IC-afdeling", options=["MICU", "SICU", "Overig"],
        default=["MICU", "SICU", "Overig"]
    )
    risk_filter = st.multiselect(
        "Risiconiveau", options=["Laag", "Matig", "Hoog"],
        default=["Laag", "Matig", "Hoog"]
    )
    sepsis_only = st.checkbox("Alleen bewezen sepsis-patiënten")

    st.markdown("---")
    st.markdown("### Drempelwaarden")
    alert_threshold = st.slider(
        "Alarmeer bij risicoscore ≥", min_value=1, max_value=6, value=4,
        help="Patiënten met een score op of boven deze waarde worden als hoog-risico gemarkeerd."
    )
    st.markdown("---")
    st.caption(
        "⚠️ **Klinische disclaimer**: Dit dashboard is een beslissingsondersteunend hulpmiddel. "
        "Klinisch oordeel van de behandelend arts blijft te allen tijde leidend."
    )

# ── Filter toepassen ──────────────────────────────────────────────────────────

filtered = last_df[last_df["Unit"].isin(unit_filter) & last_df["risk_level"].isin(risk_filter)]
if sepsis_only:
    filtered = filtered[filtered["is_sepsis"] == 1]

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🏥 IC Overzicht",
    "📈 Patiënt Monitor",
    "📊 Populatie Inzichten",
    "ℹ️ Over dit Dashboard",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — IC Overzicht
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("IC Overzicht — Sepsis Risicosignalering")

    # KPI-rij
    total = len(filtered)
    high_risk = (filtered["risk_score"] >= alert_threshold).sum()
    sepsis_n = filtered["is_sepsis"].sum()
    avg_score = filtered["risk_score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patiënten (gefilterd)", f"{total}")
    c2.metric("Hoog-risico alerts 🔴", f"{high_risk}", delta=None)
    c3.metric("Sepsis-patiënten", f"{int(sepsis_n)}")
    c4.metric("Gem. risicoscore", f"{avg_score:.1f} / 6")

    st.markdown("---")
    st.subheader("Patiëntoverzicht")

    # Tabel met kleurcodering
    display_cols = ["Patient_ID", "Unit", "ICULOS", "risk_score", "risk_level",
                    "HR", "O2Sat", "Temp", "SBP", "Resp", "is_sepsis"]
    show_df = filtered[display_cols].copy()
    show_df["is_sepsis"] = show_df["is_sepsis"].map({0: "Nee", 1: "Ja ⚠️"})
    show_df = show_df.rename(columns={
        "Patient_ID": "Patiënt ID", "Unit": "Afdeling",
        "ICULOS": "IC-uren", "risk_score": "Score",
        "risk_level": "Niveau", "is_sepsis": "Sepsis"
    }).sort_values("Score", ascending=False)

    def highlight_risk(row):
        score = row["Score"]
        if score >= 5:
            return ["background-color: #3d1515"] * len(row)
        if score >= 3:
            return ["background-color: #3d2a0d"] * len(row)
        return [""] * len(row)

    st.dataframe(
        show_df.style.apply(highlight_risk, axis=1).format({"Score": "{:.0f}"}),
        use_container_width=True, height=420
    )

    # Alarmen
    alarms = filtered[filtered["risk_score"] >= alert_threshold]
    if len(alarms) > 0:
        st.markdown("---")
        st.subheader(f"🚨 Actieve Alarmen ({len(alarms)})")
        for _, row in alarms.iterrows():
            parts = score_contributions(row)
            active = [k for k, v in parts.items() if v > 0]
            st.markdown(
                f'<div class="alert-box">'
                f'<strong>Patiënt {row["Patient_ID"]}</strong> — '
                f'Afdeling: {row["Unit"]} | IC-uur: {int(row["ICULOS"])} | '
                f'Score: {int(row["risk_score"])}/6<br>'
                f'<small>Actieve indicatoren: {", ".join(active)}</small>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Risk distributie
    st.markdown("---")
    st.subheader("Risicodistributie")
    col_a, col_b = st.columns(2)

    with col_a:
        fig_risk = px.histogram(
            filtered, x="risk_score", nbins=7,
            color="risk_level",
            color_discrete_map={"Laag": "#28a745", "Matig": "#fd7e14", "Hoog": "#dc3545"},
            labels={"risk_score": "Risicoscore", "count": "Patiënten"},
            title="Verdeling risicoscores"
        )
        fig_risk.update_layout(bargap=0.1, showlegend=True)
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_b:
        level_counts = filtered["risk_level"].value_counts().reset_index()
        level_counts.columns = ["Niveau", "Aantal"]
        fig_pie = px.pie(
            level_counts, values="Aantal", names="Niveau",
            color="Niveau",
            color_discrete_map={"Laag": "#28a745", "Matig": "#fd7e14", "Hoog": "#dc3545"},
            title="Risiconiveau verdeling"
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Patiënt Monitor
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("Patiënt Monitor")

    patient_ids = sorted(df["Patient_ID"].unique())
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_id = st.selectbox("Selecteer patiënt", options=patient_ids, key="pat_sel")
    with col_sel2:
        st.markdown("<br>", unsafe_allow_html=True)
        highlight_sepsis = st.checkbox("Toon sepsis-venster", value=True)

    pat_df = df[df["Patient_ID"] == selected_id].copy()
    pat_df = compute_score_series(pat_df)
    onset_h = onset_map.get(selected_id)

    # Header info
    demo = pat_df.iloc[-1]
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("Leeftijd", f"{int(demo['Age'])} jr" if not pd.isna(demo['Age']) else "N/B")
    col_i2.metric("Geslacht", "Man" if demo['Gender'] == 1 else "Vrouw" if demo['Gender'] == 0 else "N/B")
    col_i3.metric("Afdeling", demo['Unit'])
    col_i4.metric("IC-verblijf", f"{int(demo['ICULOS'])} uur")

    # Sepsis alert
    if onset_h is not None:
        current_h = int(pat_df["ICULOS"].max())
        if current_h >= onset_h:
            st.markdown(
                f'<div class="alert-box">🚨 <strong>SEPSIS VASTGESTELD</strong> — '
                f'Aanvangsuur: {onset_h} | Huidig IC-uur: {current_h}</div>',
                unsafe_allow_html=True
            )
        else:
            hours_to = onset_h - current_h
            st.markdown(
                f'<div class="warn-box">⚠️ <strong>SEPSIS RISICO</strong> — '
                f'Verwacht sepsis over ~{hours_to} uur (IC-uur {onset_h})</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="info-box">✅ Geen sepsis gedetecteerd in dit opname-traject</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Risicoscore tijdlijn ──
    st.subheader("Risicoscore over tijd")
    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=pat_df["ICULOS"], y=pat_df["risk_score"],
        mode="lines+markers", name="Risicoscore",
        line=dict(color="#fd7e14", width=2),
        fill="tozeroy", fillcolor="rgba(253,126,20,0.15)"
    ))
    fig_score.add_hline(y=alert_threshold, line_dash="dash",
                         line_color="#dc3545", annotation_text=f"Alarmdrempel ({alert_threshold})")
    if highlight_sepsis and onset_h:
        fig_score.add_vrect(x0=max(0, onset_h - 6), x1=onset_h,
                            fillcolor="rgba(220,53,69,0.15)",
                            annotation_text="Vroeg-detectie venster (-6u)",
                            layer="below", line_width=0)
        fig_score.add_vline(x=onset_h, line_dash="solid", line_color="#dc3545",
                            annotation_text="Sepsis-onset")
    fig_score.update_layout(
        xaxis_title="IC-uur", yaxis_title="Score (0–6)",
        yaxis=dict(range=[0, 6.5]), height=280,
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # ── Vitale parameters ──
    st.subheader("Vitale parameters")
    vital_tabs = st.tabs([VITAL_RANGES[v]["label"] for v in VITAL_COLS if v in pat_df.columns])
    for i, vit in enumerate(v for v in VITAL_COLS if v in pat_df.columns):
        with vital_tabs[i]:
            info = VITAL_RANGES[vit]
            series = pat_df[["ICULOS", vit]].dropna(subset=[vit])
            if series.empty:
                st.info("Geen metingen beschikbaar.")
                continue

            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(
                x=series["ICULOS"], y=series[vit],
                mode="lines+markers", name=vit,
                line=dict(color="#4c9be8", width=2)
            ))
            lo_n, hi_n = info["normal"]
            lo_w, hi_w = info["warn"]
            fig_v.add_hrect(y0=lo_n, y1=hi_n, fillcolor="rgba(40,167,69,0.12)",
                            line_width=0, annotation_text="Normaal", annotation_position="top left")
            fig_v.add_hrect(y0=lo_w, y1=lo_n, fillcolor="rgba(253,126,20,0.12)",
                            line_width=0)
            fig_v.add_hrect(y0=hi_n, y1=hi_w, fillcolor="rgba(253,126,20,0.12)",
                            line_width=0)
            if highlight_sepsis and onset_h:
                fig_v.add_vline(x=onset_h, line_dash="dash", line_color="#dc3545",
                                annotation_text="Sepsis")
            latest_val = series[vit].iloc[-1]
            status = "Normaal" if lo_n <= latest_val <= hi_n else ("Let op" if lo_w <= latest_val <= hi_w else "Afwijkend")
            fig_v.update_layout(
                xaxis_title="IC-uur", yaxis_title=f"{vit} ({info['unit']})",
                height=250, margin=dict(t=10, b=30)
            )
            col_val, col_status = st.columns([3, 1])
            with col_val:
                st.plotly_chart(fig_v, use_container_width=True)
            with col_status:
                st.metric(f"Laatste meting", f"{latest_val:.1f} {info['unit']}")
                color = {"Normaal": "🟢", "Let op": "🟠", "Afwijkend": "🔴"}
                st.markdown(f"{color.get(status, '')} **{status}**")

    # ── Laboratorium ──
    st.markdown("---")
    st.subheader("Laboratoriumwaarden (selectie)")
    key_labs = ["Lactate", "WBC", "Creatinine", "BUN", "Bilirubin_total", "Platelets"]
    available_labs = [l for l in key_labs if l in pat_df.columns and pat_df[l].notna().any()]

    if available_labs:
        lab_cols = st.columns(len(available_labs))
        for idx, lab in enumerate(available_labs):
            lab_series = pat_df[["ICULOS", lab]].dropna(subset=[lab])
            if lab_series.empty:
                continue
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(
                x=lab_series["ICULOS"], y=lab_series[lab],
                mode="lines+markers", name=lab,
                line=dict(width=1.5)
            ))
            if highlight_sepsis and onset_h:
                fig_l.add_vline(x=onset_h, line_dash="dash", line_color="#dc3545")
            fig_l.update_layout(
                title=lab, height=180,
                margin=dict(t=30, b=20, l=20, r=10),
                showlegend=False
            )
            with lab_cols[idx]:
                st.plotly_chart(fig_l, use_container_width=True)

    # ── Score breakdown ──
    st.markdown("---")
    st.subheader("Score-onderbouwing (laatste meting)")
    contrib = score_contributions(pat_df.iloc[-1])
    st.markdown(risk_badge(pat_df["risk_score"].iloc[-1]), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    breakdown_df = pd.DataFrame([
        {"Indicator": k, "Score": v, "Status": "✅ Actief" if v > 0 else "—"}
        for k, v in contrib.items()
    ])
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<p class="disclaimer">Disclaimer: de risicoscore is gebaseerd op gepubliceerde klinische criteria '
        '(qSOFA + SIRS). Dit systeem geeft ondersteuning — géén vervanging van klinisch oordeel. '
        'Alle acties worden vastgelegd conform AVG art. 22 (geautomatiseerde besluitvorming).</p>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Populatie Inzichten
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.title("Populatie Inzichten")

    # Sepsis statistieken
    total_pats = df["Patient_ID"].nunique()
    sepsis_pats = df[df["is_sepsis"] == 1]["Patient_ID"].nunique()
    prevalence = sepsis_pats / total_pats * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Totaal patiënten", f"{total_pats}")
    c2.metric("Sepsis-patiënten", f"{sepsis_pats}")
    c3.metric("Prevalentie", f"{prevalence:.1f}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # Vroeg-detectie analyse
    with col_l:
        st.subheader("Vroeg-detectie potentieel")
        onset_hours = []
        for pid, oh in onset_map.items():
            if oh is not None:
                onset_hours.append(oh)
        if onset_hours:
            fig_onset = px.histogram(
                x=onset_hours, nbins=30,
                labels={"x": "Sepsis-onset (IC-uur)", "count": "Patiënten"},
                title="Distributie sepsis-onset tijdstip"
            )
            fig_onset.add_vline(x=np.median(onset_hours), line_dash="dash",
                                annotation_text=f"Mediaan: {int(np.median(onset_hours))}u")
            st.plotly_chart(fig_onset, use_container_width=True)

        # Hoeveel patiënten hadden score >= threshold VOOR onset?
        early_detected = 0
        for pid in [p for p, o in onset_map.items() if o is not None]:
            pat = df[df["Patient_ID"] == pid].copy()
            pat = compute_score_series(pat)
            before_onset = pat[pat["ICULOS"] < (onset_map[pid] or 999)]
            if (before_onset["risk_score"] >= alert_threshold).any():
                early_detected += 1

        if sepsis_pats > 0:
            sensitivity = early_detected / sepsis_pats * 100
            st.metric(
                "Vroeg-detectie (vóór sepsis-onset)",
                f"{early_detected} / {sepsis_pats} patiënten",
                delta=f"{sensitivity:.1f}% sensitiviteit"
            )

    # Leeftijd en geslacht
    with col_r:
        st.subheader("Demografische verdeling")
        demo_df = df.groupby("Patient_ID")[["Age", "Gender", "is_sepsis"]].first().reset_index()
        demo_df["Geslacht"] = demo_df["Gender"].map({0: "Vrouw", 1: "Man"})
        demo_df["Sepsis"] = demo_df["is_sepsis"].map({0: "Geen sepsis", 1: "Sepsis"})

        fig_age = px.box(
            demo_df, x="Sepsis", y="Age", color="Geslacht",
            labels={"Age": "Leeftijd", "Sepsis": ""},
            title="Leeftijdsverdeling naar sepsis-status en geslacht",
            color_discrete_map={"Vrouw": "#e377c2", "Man": "#4c9be8"}
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Missing data analyse
    st.markdown("---")
    st.subheader("Datakwaliteit — Missende waarden")
    st.markdown(
        "Inzicht in missende waarden is essentieel voor betrouwbaar gebruik van het systeem. "
        "Hoge missende percentages beïnvloeden de risicoscore-berekening."
    )
    miss = missing_rate(df, VITAL_COLS + ["Lactate", "WBC", "Creatinine", "BUN"])
    miss_df = pd.DataFrame({"Variabele": miss.index, "Missend (%)": (miss.values * 100).round(1)})
    fig_miss = px.bar(
        miss_df, x="Missend (%)", y="Variabele", orientation="h",
        color="Missend (%)",
        color_continuous_scale=["#28a745", "#fd7e14", "#dc3545"],
        title="Percentage missende waarden per variabele"
    )
    fig_miss.update_layout(yaxis=dict(autorange="reversed"), height=350)
    st.plotly_chart(fig_miss, use_container_width=True)

    # Fairness check: score per geslacht
    st.markdown("---")
    st.subheader("Fairness analyse — Risicoscore per groep")
    st.markdown(
        "Het systeem wordt gecontroleerd op bias: krijgen vrouwen en mannen, "
        "en jongere vs oudere patiënten, vergelijkbare scores bij dezelfde klinische toestand?"
    )
    last_fair = compute_score_series(latest_vitals(df))
    last_fair["Geslacht"] = last_fair["Gender"].map({0: "Vrouw", 1: "Man"})
    last_fair["Leeftijdsgroep"] = pd.cut(
        last_fair["Age"], bins=[0, 50, 65, 80, 120],
        labels=["< 50", "50–65", "65–80", "> 80"]
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fig_f1 = px.box(
            last_fair.dropna(subset=["Geslacht"]),
            x="Geslacht", y="risk_score", color="Geslacht",
            labels={"risk_score": "Risicoscore"},
            title="Risicoscore naar geslacht",
            color_discrete_map={"Vrouw": "#e377c2", "Man": "#4c9be8"}
        )
        st.plotly_chart(fig_f1, use_container_width=True)
    with col_f2:
        fig_f2 = px.box(
            last_fair.dropna(subset=["Leeftijdsgroep"]),
            x="Leeftijdsgroep", y="risk_score", color="Leeftijdsgroep",
            labels={"risk_score": "Risicoscore"},
            title="Risicoscore naar leeftijdsgroep"
        )
        st.plotly_chart(fig_f2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Over dit Dashboard
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.title("Over dit Dashboard")

    st.markdown("""
    ## Doel
    Dit dashboard is ontwikkeld voor artsen en verpleegkundigen op de Intensive Care van
    **Isala Ziekenhuis Zwolle** om sepsis **vroegtijdig** (≥ 6 uur vóór klinische diagnose)
    te kunnen signaleren.

    ## Businessvraag
    > *"Hoe kan het IC-team van Isala ondersteund worden bij het tijdig herkennen van sepsis
    > bij opgenomen patiënten, zodat mortaliteit en behandelvertraging worden gereduceerd?"*

    ## KPI's
    | KPI | Streefwaarde |
    |-----|-------------|
    | Sensitiviteit (detectie vóór onset) | ≥ 75% |
    | False Positive Rate | ≤ 25% |
    | Vroeg-detectievenster | ≥ 6 uur vóór klinische diagnose |
    | Paginalaadtijd | < 3 seconden |
    | Beschikbaarheid | 24/7, 99.5% uptime |

    ## Risicoscore — methode
    De risicoscore is gebaseerd op **qSOFA** (Quick Sequential Organ Failure Assessment)
    aangevuld met SIRS-criteria en laboratoriumindicatoren:
    - Ademfrequentie ≥ 22/min
    - Systolische BD ≤ 100 mmHg
    - Temperatuur < 36.0°C of > 38.3°C
    - Hartfrequentie > 90 bpm
    - Lactaat > 2.0 mmol/L
    - WBC < 4 of > 12 ×10³/µL

    **Bewuste keuze**: een transparant regelgebaseerd model in plaats van een model
    dat zijn redenering verbergt, omdat de **EU AI Act** (art. 13) uitlegbaarheid
    verplicht en klinisch personeel de score per indicator moet kunnen begrijpen.

    ## Juridische context
    - **AVG (GDPR)**: Patiëntgegevens worden alleen lokaal verwerkt; geen export naar derde partijen.
      Het dashboard toont geen naam/BSN (pseudonimisering). Art. 22 AVG: geautomatiseerde besluitvorming
      vereist menselijk toezicht — de klinische disclaimer is hierop gebaseerd.
    - **EU AI Act**: Dit systeem valt onder **hoog-risico AI** (Bijlage III, medische apparatuur).
      Vereisten: technische documentatie, menselijk toezicht, transparantie, auditlog.
    - **Wet BIG / Wkkgz**: Het systeem ondersteunt klinische besluitvorming; aansprakelijkheid
      blijft bij de behandelend arts.

    ## Data
    Bron: **PhysioNet/CinC Challenge 2019** — 40.336 IC-patiënten uit drie ziekenhuissystemen.
    Gebruik van de data is gelicenseerd onder PhysioNet Open Data Commons.

    ## Aanbevelingen voor implementatie bij Isala
    1. Koppeling met het patiëntensysteem (HiX) van Isala voor live data per patiënt.
    2. Aanvullen met Isala-eigen data zodat de scores beter aansluiten op de lokale populatie.
    3. Testperiode: minimaal 3 maanden naast het bestaande protocol draaien vóór klinisch gebruik.
    4. Jaarlijkse hervalidatie conform de EU AI Act.
    5. Training voor IC-medewerkers over het omgaan met onterechte alarmen en de interpretatie van scores.
    """)
