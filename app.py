from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st


st.set_page_config(page_title="İzmir İhracatı ve İngiltere İş Çevrimi", page_icon="📈", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "Izmir_UK_NBER_raw_1864_1914.csv"
QUARTER_COLUMNS = ["NBER_Q1_raw", "NBER_Q2_raw", "NBER_Q3_raw", "NBER_Q4_raw"]
REQUIRED_COLUMNS = ["year", "Izmir_exports_GBP_raw", *QUARTER_COLUMNS]


@st.cache_data(show_spinner=False)
def read_data(file_bytes: bytes | None, default_path: str) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(file_bytes)) if file_bytes is not None else pd.read_csv(default_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("CSV’de gerekli sütunlar eksik: " + ", ".join(missing))
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    for column in ["Izmir_exports_GBP_raw", *QUARTER_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("year").reset_index(drop=True)


def construct_series(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float]]:
    # Raw exports / linear trend × 100; then two centered 3-year moving averages.
    izmir = frame.dropna(subset=["Izmir_exports_GBP_raw"])[["year", "Izmir_exports_GBP_raw"]].copy()
    slope, intercept = np.polyfit(
        izmir["year"].to_numpy(), izmir["Izmir_exports_GBP_raw"].to_numpy(), 1
    )
    izmir["Izmir_linear_trend_GBP"] = slope * izmir["year"] + intercept
    izmir["Izmir_ratio_index"] = 100 * izmir["Izmir_exports_GBP_raw"] / izmir["Izmir_linear_trend_GBP"]
    izmir["Izmir_Kurmus_index"] = (
        izmir["Izmir_ratio_index"]
        .rolling(3, center=True, min_periods=3).mean()
        .rolling(3, center=True, min_periods=3).mean()
    )
    out = frame.copy()
    out["NBER_quarter_count"] = out[QUARTER_COLUMNS].notna().sum(axis=1)
    out["NBER_annual_mean"] = out[QUARTER_COLUMNS].mean(axis=1)
    out = out.merge(izmir.drop(columns=["Izmir_exports_GBP_raw"]), on="year", how="left")
    return out, (float(slope), float(intercept))


st.title("İzmir İhracatı ve İngiltere İş Çevrimi")
st.caption(
    "İzmir’in ham ihracat serisini doğrusal trend ve iki kez uygulanan merkezlenmiş "
    "3 yıllık hareketli ortalamayla dönüştürür; NBER İngiltere endeksiyle karşılaştırır."
)

with st.sidebar:
    st.header("Veri")
    uploaded = st.file_uploader("Birleşik ham CSV dosyası", type=["csv"])
    st.caption("Dosya yüklemezseniz uygulama klasöründeki varsayılan CSV kullanılır.")

try:
    raw = read_data(uploaded.getvalue() if uploaded else None, str(DEFAULT_DATA))
    data, trend_params = construct_series(raw)
except Exception as exc:
    st.error(f"Veri okunamadı: {exc}")
    st.stop()

valid = data.dropna(subset=["Izmir_Kurmus_index", "NBER_annual_mean"]).copy()
valid = valid[valid["NBER_quarter_count"] == 4]
if valid.empty:
    st.warning("İki serinin karşılaştırılabileceği tam yıllık gözlem bulunamadı.")
    st.stop()

min_year, max_year = int(valid["year"].min()), int(valid["year"].max())
default_end = min(1895, max_year)
with st.sidebar:
    st.header("Dönem seçimi")
    start_year, end_year = st.select_slider(
        "Analiz aralığı",
        options=list(range(min_year, max_year + 1)),
        value=(min_year, default_end),
    )
    st.caption("Pozitif gecikme, İngiltere endeksinin İzmir ihracatından önce geldiğini ifade eder.")

sample = valid[valid["year"].between(start_year, end_year)].copy().sort_values("year")
if len(sample) < 5:
    st.warning("İstatistiksel karşılaştırma için en az 5 ortak yıllık gözlem seçin.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ortak yıllık gözlem", f"{len(sample)}")
c2.metric("İzmir endeksi ortalaması", f"{sample['Izmir_Kurmus_index'].mean():.2f}")
c3.metric("NBER endeksi ortalaması", f"{sample['NBER_annual_mean'].mean():.2f}")
c4.metric("Dönem", f"{start_year}–{end_year}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sample["year"], y=sample["NBER_annual_mean"], mode="lines+markers",
    name="NBER İngiltere endeksi (yıllık ortalama)", line=dict(color="#2767a8", width=3),
    hovertemplate="Yıl: %{x}<br>NBER: %{y:.2f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=sample["year"], y=sample["Izmir_Kurmus_index"], mode="lines+markers",
    name="İzmir ihracatı (Kurmuş yöntemiyle endeks)", line=dict(color="#c6532f", width=3),
    hovertemplate="Yıl: %{x}<br>İzmir endeksi: %{y:.2f}<extra></extra>",
))
fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.7)
fig.update_layout(
    title="İngiltere iş çevrimi ve İzmir ihracatı",
    xaxis_title="Yıl", yaxis_title="Endeks (100 = normal / trend)",
    hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=520, margin=dict(l=20, r=20, t=80, b=20),
)
st.plotly_chart(fig, use_container_width=True)

levels_r = sample["NBER_annual_mean"].corr(sample["Izmir_Kurmus_index"])
diff_sample = sample[["NBER_annual_mean", "Izmir_Kurmus_index"]].diff().dropna()
diff_r = diff_sample["NBER_annual_mean"].corr(diff_sample["Izmir_Kurmus_index"])
metrics = st.columns(2)
metrics[0].metric("Düzeylerde korelasyon", f"{levels_r:.3f}" if pd.notna(levels_r) else "—")
metrics[1].metric("Yıllık değişimlerde korelasyon", f"{diff_r:.3f}" if pd.notna(diff_r) else "—")

st.subheader("Gecikme analizi")
lag_rows = []
uk_series = sample.set_index("year")["NBER_annual_mean"]
iz_series = sample.set_index("year")["Izmir_Kurmus_index"]
for lag in range(-4, 5):
    # lag > 0 compares Izmir(t) with UK(t-lag): UK leads by lag years.
    pair = pd.concat([iz_series.rename("izmir"), uk_series.shift(lag).rename("uk_lagged")], axis=1).dropna()
    lag_rows.append({"Gecikme (yıl)": lag, "Korelasyon": pair["izmir"].corr(pair["uk_lagged"]), "N": len(pair)})
lag_table = pd.DataFrame(lag_rows)
st.dataframe(lag_table.style.format({"Korelasyon": "{:.3f}"}), use_container_width=True, hide_index=True)

two_year = sample.set_index("year").copy()
two_year["NBER_lag2"] = two_year["NBER_annual_mean"].shift(2)
two_year = two_year.dropna(subset=["Izmir_Kurmus_index", "NBER_lag2"])
if len(two_year) >= 5:
    model = sm.OLS(
        two_year["Izmir_Kurmus_index"], sm.add_constant(two_year["NBER_lag2"])
    ).fit(cov_type="HAC", cov_kwds={"maxlags": min(3, len(two_year) - 1)})
    st.write(
        f"**İki yıllık gecikmeli regresyon (HAC standart hataları):** "
        f"β = {model.params['NBER_lag2']:.3f}, SE = {model.bse['NBER_lag2']:.3f}, "
        f"p = {model.pvalues['NBER_lag2']:.3f}, R² = {model.rsquared:.3f}."
    )

with st.expander("Dönüştürme ve veri notları"):
    slope, intercept = trend_params
    st.markdown(
        "**İzmir dönüşümü:** Ham ihracat değerlerine tüm mevcut İzmir yılları üzerinden "
        "doğrusal trend tahmin edilir. Ham değer / tahmini trend × 100 oranına, "
        "merkezlenmiş 3 yıllık hareketli ortalama iki kez uygulanır. Uç yıllar kaybolduğu "
        "için düzeltilmiş İzmir endeksi 1866–1910 aralığındadır."
    )
    st.write(f"Tahmini doğrusal trend eğimi: £{slope:,.2f}/yıl; kesişim: £{intercept:,.2f}.")
    st.markdown(
        "**NBER:** Çeyreklik değerler CSV’deki haliyle korunur; yıllık ortalama dört "
        "çeyrekten hesaplanır. NBER endeksi İngiltere GSYH’si değildir. 1914 yalnızca "
        "iki çeyrek içerdiğinden tam yıl ortalaması sayılmaz."
    )
    st.markdown(
        "**Yorum sınırı:** İzmir endeksi yumuşatılmıştır; seri korelasyonu bulunabilir. "
        "Korelasyon ve gecikmeli regresyon nedensellik kanıtı olarak yorumlanmamalıdır."
    )

download_columns = [
    "year", "Izmir_exports_GBP_raw", "Izmir_source_footnote_marker",
    *QUARTER_COLUMNS, "NBER_quarter_count", "NBER_annual_mean",
    "Izmir_linear_trend_GBP", "Izmir_ratio_index", "Izmir_Kurmus_index",
]
download_data = data[data["year"].isin(sample["year"])][download_columns]
st.download_button(
    "Seçili dönemin verisini indir",
    data=download_data.to_csv(index=False).encode("utf-8"),
    file_name=f"izmir_nber_analysis_{start_year}_{end_year}.csv",
    mime="text/csv",
)

st.caption(
    "Kaynaklar: İzmir ihracat değerleri Orhan Kurmuş’un tezindeki Appendix I tablosu; "
    "İngiltere iş çevrimi serisi NBER Macrohistory Database / FRED, Q12010GBQ511SNBR."
)
