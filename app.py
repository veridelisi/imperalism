from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
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
    # Raw exports / linear trend × 100; then one centered 3-year moving average.
    izmir = frame.dropna(subset=["Izmir_exports_GBP_raw"])[["year", "Izmir_exports_GBP_raw"]].copy()
    slope, intercept = np.polyfit(
        izmir["year"].to_numpy(), izmir["Izmir_exports_GBP_raw"].to_numpy(), 1
    )
    izmir["Izmir_linear_trend_GBP"] = slope * izmir["year"] + intercept
    izmir["Izmir_ratio_index"] = 100 * izmir["Izmir_exports_GBP_raw"] / izmir["Izmir_linear_trend_GBP"]
    izmir["Izmir_Kurmus_index"] = (
        izmir["Izmir_ratio_index"]
        .rolling(3, center=True, min_periods=3).mean()
    )
    out = frame.copy()
    out["NBER_quarter_count"] = out[QUARTER_COLUMNS].notna().sum(axis=1)
    out["NBER_annual_mean"] = out[QUARTER_COLUMNS].mean(axis=1)
    out = out.merge(izmir.drop(columns=["Izmir_exports_GBP_raw"]), on="year", how="left")
    return out, (float(slope), float(intercept))


st.title("İzmir İhracatı ve İngiltere İş Çevrimi")
st.caption(
    "İzmir’in ham ihracat serisini doğrusal trend ve merkezlenmiş "
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

st.subheader("1. Zaman serisi analizi")
st.markdown(
    "Önce endekslerin durağanlığı ve genel birlikte hareketi inceleniyor. "
    "ADF testinde sıfır hipotezi serinin birim kök içerdiğidir."
)

def adf_row(series, name):
    values = series.dropna().astype(float)
    if len(values) < 12:
        return {
            "Seri": name, "N": len(values), "Düzey ADF p": np.nan,
            "İlk fark ADF p": np.nan, "ADF yorumu": "Gözlem sayısı yetersiz"
        }
    try:
        level_test = adfuller(values, regression="c", autolag="AIC")
        diff_test = adfuller(values.diff().dropna(), regression="c", autolag="AIC")
        level_p, diff_p = float(level_test[1]), float(diff_test[1])
        if level_p < 0.05:
            verdict = "Düzeyde durağanlık lehine"
        elif diff_p < 0.05:
            verdict = "Düzeyde birim kök reddedilemedi; ilk fark durağan"
        else:
            verdict = "Birim kök sonucu belirsiz"
        return {
            "Seri": name, "N": len(values), "Düzey ADF p": level_p,
            "İlk fark ADF p": diff_p, "ADF yorumu": verdict
        }
    except Exception as exc:
        return {
            "Seri": name, "N": len(values), "Düzey ADF p": np.nan,
            "İlk fark ADF p": np.nan, "ADF yorumu": f"Test hesaplanamadı: {exc}"
        }

uk_level = sample.set_index("year")["NBER_annual_mean"]
iz_level = sample.set_index("year")["Izmir_Kurmus_index"]
uk_adf = adf_row(uk_level, "NBER İngiltere endeksi")
iz_adf = adf_row(iz_level, "İzmir ihracat endeksi")
adf_table = pd.DataFrame([uk_adf, iz_adf])
st.dataframe(
    adf_table.style.format({"Düzey ADF p": "{:.3f}", "İlk fark ADF p": "{:.3f}"}),
    use_container_width=True, hide_index=True
)

level_pair = pd.concat([uk_level.rename("uk"), iz_level.rename("izmir")], axis=1).dropna()
diff_pair = level_pair.diff().dropna()
level_r, level_p = pearsonr(level_pair["uk"], level_pair["izmir"])
diff_r, diff_p = pearsonr(diff_pair["uk"], diff_pair["izmir"])
c1, c2 = st.columns(2)
c1.metric("Düzeylerde korelasyon", f"r = {level_r:.3f}", delta=f"p = {level_p:.3f}")
c2.metric("Yıllık değişimlerde korelasyon", f"r = {diff_r:.3f}", delta=f"p = {diff_p:.3f}")

delta_fig = go.Figure()
delta_fig.add_trace(go.Scatter(
    x=diff_pair.index, y=diff_pair["uk"], mode="lines+markers",
    name="NBER endeksinin yıllık değişimi", line=dict(color="#2767a8", width=2)
))
delta_fig.add_trace(go.Scatter(
    x=diff_pair.index, y=diff_pair["izmir"], mode="lines+markers",
    name="İzmir endeksinin yıllık değişimi", line=dict(color="#c6532f", width=2)
))
delta_fig.add_hline(y=0, line_dash="dot", line_color="gray")
delta_fig.update_layout(
    title="Endekslerde yıllık değişimler",
    xaxis_title="Yıl", yaxis_title="Endeks puanı değişimi",
    hovermode="x unified", height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=20, r=20, t=70, b=20),
)
st.plotly_chart(delta_fig, use_container_width=True)

both_level_stationary = (
    pd.notna(uk_adf["Düzey ADF p"]) and pd.notna(iz_adf["Düzey ADF p"])
    and uk_adf["Düzey ADF p"] < 0.05 and iz_adf["Düzey ADF p"] < 0.05
)
both_diff_stationary = (
    pd.notna(uk_adf["İlk fark ADF p"]) and pd.notna(iz_adf["İlk fark ADF p"])
    and uk_adf["İlk fark ADF p"] < 0.05 and iz_adf["İlk fark ADF p"] < 0.05
)
if both_level_stationary:
    st.info("ADF sonuçları iki endeksin de bu dönemde düzeyde durağan olduğu yönünde; düzey ilişkisi incelenebilir.")
elif both_diff_stationary:
    st.info("ADF, düzeylerde birim kökü reddedemiyor; ilk farklar durağan görünüyor. Gecikme testinde değişimler esas alınacak.")
else:
    st.warning("ADF sonuçları ortak bir durağanlık yapısını netleştirmiyor. Düzey ve değişim testleri keşifsel olarak birlikte gösterilecek.")

st.subheader("2. İki yıllık gecikme iddiasının ayrı testi")
st.markdown(
    "Önceden belirlenen iddia şu şekilde sınanıyor: İngiltere iş çevrimi endeksi iki yıl önceki "
    "değeriyle İzmir ihracat endeksi arasında pozitif ilişki taşır. Katsayı için H₀: β = 0 "
    "test edilir; standart hatalar HAC/Newey–West ile düzeltilir."
)

all_years = data.set_index("year").sort_index()
uk_all = all_years["NBER_annual_mean"]
iz_all = all_years["Izmir_Kurmus_index"]
uk_lag2 = uk_all.shift(2)
level_test_data = pd.concat(
    [iz_all.rename("izmir"), uk_lag2.rename("uk_t_minus_2")], axis=1
).loc[start_year:end_year].dropna()

uk_change = uk_all.diff()
iz_change = iz_all.diff()
change_test_data = pd.concat(
    [iz_change.rename("delta_izmir"), uk_change.shift(2).rename("delta_uk_t_minus_2")],
    axis=1
).loc[start_year:end_year].dropna()

def hac_result(frame, y_name, x_name):
    if len(frame) < 8:
        return None
    fit = sm.OLS(frame[y_name], sm.add_constant(frame[x_name])).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(3, len(frame) - 1)}
    )
    ci = fit.conf_int().loc[x_name]
    return {
        "Gözlem": len(frame), "β": fit.params[x_name], "HAC SE": fit.bse[x_name],
        "p-değeri": fit.pvalues[x_name], "Alt %95": ci.iloc[0],
        "Üst %95": ci.iloc[1], "R²": fit.rsquared
    }

level_result = hac_result(level_test_data, "izmir", "uk_t_minus_2")
change_result = hac_result(change_test_data, "delta_izmir", "delta_uk_t_minus_2")
test_rows = []
if level_result:
    test_rows.append({"Model": "Endeks düzeyleri: İzmir(t) ~ İngiltere(t−2)", **level_result})
if change_result:
    test_rows.append({"Model": "Yıllık değişimler: Δİzmir(t) ~ Δİngiltere(t−2)", **change_result})
if test_rows:
    result_table = pd.DataFrame(test_rows)
    st.dataframe(
        result_table.style.format({
            "β": "{:.3f}", "HAC SE": "{:.3f}", "p-değeri": "{:.3f}",
            "Alt %95": "{:.3f}", "Üst %95": "{:.3f}", "R²": "{:.3f}"
        }),
        use_container_width=True, hide_index=True
    )
    if both_level_stationary and level_result:
        chosen = level_result
        chosen_label = "düzey modelinde"
    elif both_diff_stationary and change_result:
        chosen = change_result
        chosen_label = "yıllık değişim modelinde"
    else:
        chosen = None
        chosen_label = ""
    if chosen:
        if chosen["β"] > 0 and chosen["p-değeri"] < 0.05:
            st.success(
                f"Seçilen durağanlık yapısına uygun {chosen_label} iki yıllık gecikme için "
                f"pozitif katsayı istatistiksel olarak anlamlı (β={chosen['β']:.3f}, "
                f"p={chosen['p-değeri']:.3f})."
            )
        else:
            st.info(
                f"Seçilen durağanlık yapısına uygun {chosen_label} iki yıllık gecikme iddiası "
                f"%5 düzeyinde desteklenmiyor (β={chosen['β']:.3f}, p={chosen['p-değeri']:.3f})."
            )
    else:
        st.caption("Durağanlık sonucu net olmadığı için iki yıllık gecikme katsayıları iki modelde de raporlanıyor; sonuç kesin kanıt sayılmamalı.")
else:
    st.warning("Seçilen dönem iki yıllık gecikme regresyonu için yeterli gözlem içermiyor.")

with st.expander("Dönüştürme ve veri notları"):
    slope, intercept = trend_params
    st.markdown(
        "**İzmir dönüşümü:** Ham ihracat değerlerine tüm mevcut İzmir yılları üzerinden "
        "doğrusal trend tahmin edilir. Ham değer / tahmini trend × 100 oranına, "
        "merkezlenmiş 3 yıllık hareketli ortalama bir kez uygulanır. Bu nedenle kullanılabilir "
        "İzmir endeksi 1865–1911 aralığındadır."
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
