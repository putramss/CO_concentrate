import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Path file di GitHub
november_data_path = "https://raw.githubusercontent.com/putramss/CO_concentrate/refs/heads/main/november_data.csv"
december_data_path = "https://raw.githubusercontent.com/putramss/CO_concentrate/refs/heads/main/december_data.csv"
december_predictions_path = "https://raw.githubusercontent.com/putramss/CO_concentrate/refs/heads/main/december_data_predicted.csv"

# Baca file CSV
try:
    november_data = pd.read_csv(november_data_path, encoding='utf-8')
except UnicodeDecodeError:
    november_data = pd.read_csv(november_data_path, encoding='ISO-8859-1')

try:
    december_data = pd.read_csv(december_data_path, encoding='utf-8')
except UnicodeDecodeError:
    december_data = pd.read_csv(december_data_path, encoding='ISO-8859-1')

try:
    december_predictions = pd.read_csv(december_predictions_path, encoding='utf-8')
except UnicodeDecodeError:
    december_predictions = pd.read_csv(december_predictions_path, encoding='ISO-8859-1')

# Konversi kolom Timestamp ke Date
def convert_to_date(df):
    if 'Timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    else:
        df['Date'] = np.nan
    return df

november_data = convert_to_date(november_data)
december_data = convert_to_date(december_data)
december_predictions = convert_to_date(december_predictions)

november_data.dropna(subset=["Date"], inplace=True)
december_data.dropna(subset=["Date"], inplace=True)

# Konfigurasi Streamlit
st.title("Industrial Carbon Monoxide Concentrate in Urban Areas")
st.markdown("---")

# Statistik CO
nov_avg_co = november_data["CO"].mean()
dec_avg_co = december_data["CO"].mean()

col1, col2 = st.columns(2)
col1.metric("Rata-rata CO November (ppm)", f"{nov_avg_co:.2f}", help="Konsentrasi CO rata-rata di bulan November")
col2.metric("Rata-rata CO Desember (ppm)", f"{dec_avg_co:.2f}", help="Konsentrasi CO rata-rata di bulan Desember")

st.markdown("---")

# **1. Perbandingan CO Bulan November dan Desember (Plotly Scatter Plot)**
nov_daily_avg = november_data.groupby("Date")["CO"].mean().reset_index()
dec_daily_avg = december_data.groupby("Date")["CO"].mean().reset_index()

fig_nov_dec = go.Figure()

fig_nov_dec.add_trace(go.Scatter(
    x=nov_daily_avg["Date"],
    y=nov_daily_avg["CO"],
    mode='lines+markers',
    name='November',
    marker=dict(color='blue')
))

fig_nov_dec.add_trace(go.Scatter(
    x=dec_daily_avg["Date"],
    y=dec_daily_avg["CO"],
    mode='lines+markers',
    name='Desember',
    marker=dict(color='red')
))

fig_nov_dec.update_layout(
    title="Perbandingan Konsentrasi CO Bulan November dan Desember",
    xaxis_title="Tanggal",
    yaxis_title="Konsentrasi CO (ppm)",
    legend_title="Bulan",
    template="plotly_white"
)

st.plotly_chart(fig_nov_dec, use_container_width=True)
st.markdown("---")

# **2. Perbandingan CO Hasil Pengukuran dan Prediksi (Plotly Scatter Plot)**
# Pastikan data CO dan Predicted_CO adalah numerik
december_predictions["CO"] = pd.to_numeric(december_predictions["CO"], errors="coerce")
december_predictions["Predicted_CO"] = pd.to_numeric(december_predictions["Predicted_CO"], errors="coerce")

# Filter nilai yang masuk akal dalam satuan ppm (misalnya, 300-500 ppm)
december_predictions = december_predictions[
    (december_predictions["CO"] >= 300) & (december_predictions["CO"] <= 500)
]
december_predictions = december_predictions[
    (december_predictions["Predicted_CO"] >= 300) & (december_predictions["Predicted_CO"] <= 500)
]

# Plot ulang grafik dengan sumbu x untuk tanggal dan sumbu y untuk konsentrasi CO
fig = go.Figure()

# Tambahkan data hasil pengukuran
fig.add_trace(go.Scatter(
    x=december_predictions["Date"],
    y=december_predictions["CO"],
    mode='lines+markers',
    name='Pengukuran',
    marker=dict(color='orange')
))

# Tambahkan data hasil prediksi
fig.add_trace(go.Scatter(
    x=december_predictions["Date"],
    y=december_predictions["Predicted_CO"],
    mode='lines+markers',
    name='Prediksi',
    marker=dict(color='green')
))

# Update layout dengan sumbu yang benar
fig.update_layout(
    title="Perbandingan Konsentrasi CO Hasil Pengukuran dan Prediksi",
    xaxis_title="Tanggal/Waktu",
    yaxis_title="Konsentrasi CO (ppm)",
    xaxis=dict(title='Tanggal/Waktu', type='date', tickformat='%b %d'),
    yaxis=dict(title='Konsentrasi CO (ppm)'),
    legend_title="Kategori",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# **3. Korelasi Konsentrasi CO dan Faktor Meteorologis (Plotly Scatter Plot)**
available_columns = ["CO", "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)"]
valid_columns = [col for col in available_columns if col in november_data.columns]

selected_columns = st.multiselect(
    "Pilih faktor meteorologis untuk korelasi dengan CO:", 
    options=valid_columns, 
    default=valid_columns[1:]
)

if selected_columns:
    fig_correlation = go.Figure()

    for column in selected_columns:
        sorted_data = november_data.sort_values(by=column)
        linear_data = sorted_data.iloc[::len(sorted_data)//50][["CO", column]].dropna()

        fig_correlation.add_trace(go.Scatter(
            x=linear_data[column],
            y=linear_data["CO"],
            mode='markers',
            name=column,
            marker=dict(size=8, opacity=0.7)
        ))

    fig_correlation.update_layout(
        title="Korelasi Konsentrasi CO dan Faktor Meteorologis",
        xaxis_title="Nilai Faktor",
        yaxis_title="Konsentrasi CO (ppm)",
        legend_title="Faktor Meteorologis",
        template="plotly_white"
    )

    st.plotly_chart(fig_correlation, use_container_width=True)

st.markdown("---")

# **4. Evaluasi Model**
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", "0.923")
col2.metric("MSE", "0.851")
col3.metric("MAE", "0.745")
col4.metric("R-squared", "0.079")
st.markdown("---")

# Copyright
st.caption('Copyright (C) 2025, [Putra Ramdhani](https://www.linkedin.com/in/putramdhani/)')
