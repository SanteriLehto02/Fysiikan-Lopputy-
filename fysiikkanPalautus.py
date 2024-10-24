import pandas as pd
import numpy as np
import streamlit as st
from scipy.signal import butter, filtfilt, find_peaks,welch
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from geopy.distance import geodesic

df = pd.read_csv('Linear Acceleration.csv')
dfmap = pd.read_csv('Location.csv')

df = df.drop(index=df.index[:25]).reset_index(drop=True)
dfmap = dfmap.drop(index=dfmap.index[:25]).reset_index(drop=True)

askeleet = df['Linear Acceleration z (m/s^2)']

def butter_lowpass_filter(data, cutoff, order, fs):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

T = df['Time (s)'].iloc[-1] - df['Time (s)'].iloc[0]
n = len(df['Time (s)'])
fs = n / T
cutoff = 0.5
order = 1

filtered_signal = butter_lowpass_filter(askeleet, cutoff, order, fs)

peaks, _ = find_peaks(filtered_signal, height=0)
st.header("Fysiikan lopputyö")
st.subheader("Suodatettu data jota käytettiin askeleiden määrittämiseen")
plt.figure()
plt.plot(filtered_signal, label='Suodatettu data')
plt.legend()
st.pyplot(plt.gcf())

fft_signal = np.fft.fft(askeleet)
freq_domain = np.fft.fftfreq(n, d=1/fs)
dominant_freq = freq_domain[np.argmax(np.abs(fft_signal[:n // 2]))]
total_steps = dominant_freq * T

st.write(f"Askeleet: {len(peaks)}")
st.write(f"Askeleet käyttäen Fourier analysis: {int(total_steps)}")

total_distance = 0.0
for i in range(1, len(dfmap)):
    coords_1 = (dfmap['Latitude (°)'][i-1], dfmap['Longitude (°)'][i-1])
    coords_2 = (dfmap['Latitude (°)'][i], dfmap['Longitude (°)'][i])
    total_distance += geodesic(coords_1, coords_2).meters

time_first = dfmap['Time (s)'].iloc[0]
time_last = dfmap['Time (s)'].iloc[-1]
time_difference = time_last - time_first

step_length = total_distance / len(peaks)
average_speed = (total_distance / time_difference) * 3.6

st.write(f"Kävelty matka {total_distance:.2f} meters")
st.write(f"Askeleen pituus: {step_length:.2f} meters")
st.write(f"Keskinopeus: {average_speed:.2f} km/h")

lat_mean = dfmap['Latitude (°)'].mean()
long_mean = dfmap['Longitude (°)'].mean()

my_map = folium.Map(location=[lat_mean, long_mean], zoom_start=14)
folium.PolyLine(dfmap[['Latitude (°)', 'Longitude (°)']].values, color='red', opacity=1).add_to(my_map)

st.subheader("Kartta kävellystä reitistä")
folium_static(my_map)

frequencies, psd = welch(askeleet, fs, nperseg=1024)
dominant_freq = frequencies[np.argmax(psd)]
plt.figure()
plt.semilogy(frequencies, psd)
plt.title('Tehospektritiheys (PSD) kiihtyvyysdatasta')
plt.xlabel('Taajuus (Hz)')
plt.ylabel('Teho (V^2/Hz)')
plt.legend()
st.pyplot(plt.gcf())