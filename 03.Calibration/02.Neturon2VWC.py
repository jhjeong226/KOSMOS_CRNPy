## This code was developed by Jaehwan Jeong for the general processing of CRNP data in the KOSMOS project
## Unauthorized modification and distribution of this code are prohibited.
## Draft version: 18 October 2024
## Last updated: 18 October 2024
## This code uses functions from CRNPy (Peraza Rud et al., 2024).
import crnpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 결과를 저장할 폴더 경로
output_folder = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\02.Output\02.VWC"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
#----------------------------------------------------------------------------------

parameters_file =r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\02.Output\01.Preprocessed\Parameters.xlsx"
df_params = pd.read_excel(parameters_file)
lat = df_params['lat'].iloc[0]
lon = df_params['lon'].iloc[0]
N0_rdt = df_params['N0_rdt'].iloc[0]
Pref = df_params['Pref'].iloc[0]
Aref = df_params['Aref'].iloc[0]
clay_content = df_params['clay_content'].iloc[0]
soil_bulk_density = df_params['soil_bulk_density'].iloc[0]

# Site specific setting (Suwon, Pyeonchang, Hongcheon, etc.) *Calibration 끝냈을 때 사용!
# lat = 37.7049111
# lon = 128.0316412
# # Altitude = 444.3027
# soil_bulk_density = 1.44
# N0_rdt = 1757.8647
# Pref = 962.9302
# Aref = 12.5694
# clay_content = 0.35
z_surface = 144 # Average depth in mm obtained from previous cell using crnpy.sensing_depth()
z_subsurface = 350 # Arbitrary subsurface depth in mm

#----------------------------------------------------------------------------------
# Load observations from a stationary detector
CRNP_datapath = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\02.Output\01.Preprocessed\HC_CRNP_input.xlsx"
df_crnp = pd.read_excel(CRNP_datapath, names=['Timestamp', 'RN', 'Ta', 'RH', 'Pa', 'WS', 'WS_max', 'WD_VCT', 'N_counts'])
df_crnp['timestamp'] = pd.to_datetime(df_crnp['Timestamp'], errors='coerce')

# Ensure 'timestamp' column has valid values
if df_crnp['timestamp'].isna().all():
    raise ValueError("All timestamp values are invalid. Please check the 'Timestamp' format in your CRNP data.")
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# Compute total counts - from single detector
def remove_outliers(df, column, threshold=3):
    median = np.median(df[column])
    mad = np.median(np.abs(df[column] - median))
    mad_scaled = np.abs(df[column] - median) / (mad + 1e-6)  # Avoid division by zero
    return df[mad_scaled < threshold]

df_crnp['total_raw_counts'] = df_crnp['N_counts']#----------------------------------------------------------------------------------
df_crnp = remove_outliers(df_crnp, 'total_raw_counts', threshold=3)

# Plot total counts
# Create a new figure and plot the data
plt.figure(figsize=(15,7))
plt.plot(df_crnp['timestamp'], df_crnp['total_raw_counts'], label='Raw Counts', color='black', linewidth=.8)
plt.xlabel("Date")
plt.ylabel("Total Raw Counts")
plt.legend()
plt.title('Total Raw Counts Over Time')
plt.savefig(os.path.join(output_folder, "Total_Raw_Counts_Over_Time.png"))
plt.close()

#----------------------------------------------------------------------------------
# Neutron count correction
# Define study start and end dates
start_date = df_crnp.iloc[0]['timestamp']
end_date = df_crnp.iloc[-1]['timestamp']

#----------------------------------------------------------------------------------

# 차단 강도 계산 및 중성자 모니터 검색
# cutoff_rigidity = crnpy.cutoff_rigidity(lat, lon)
# print(f"Cutoff Rigidity at location ({lat}, {lon}): {cutoff_rigidity}")

# Find stations with cutoff rigidity similar to estimated by lat,lon
# crnpy.find_neutron_monitor(crnpy.cutoff_rigidity(lat, lon), start_date=start_date, end_date=end_date, verbose=False)

# 중성자 모니터 검색 결과 출력
# neutron_monitors = crnpy.find_neutron_monitor(cutoff_rigidity, start_date=start_date, end_date=end_date, verbose=True)
# print("Available neutron monitors:", neutron_monitors)

# Download incoming neutron flux data from the Neutron Monitor Database (NMDB).


# Use utc_offset for Central Standard Time.
nmdb = crnpy.get_incoming_neutron_flux(start_date, end_date, station="ATHN", utc_offset=9)
# Interpolate incoming neutron flux to match the timestamps in our station data
df_crnp['incoming_flux'] = crnpy.interpolate_incoming_flux(nmdb['timestamp'], nmdb['counts'], df_crnp['timestamp'])
# Compute correction factor for incoming neutron flux
df_crnp['fi'] = crnpy.correction_incoming_flux(incoming_neutrons=df_crnp['incoming_flux'],
                                          incoming_Ref=df_crnp['incoming_flux'].iloc[0])
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Atmospheric correction
# Fill NaN values in atmospheric data and neutron counts
df_crnp[['Pa', 'RH', 'Ta']] = df_crnp[['Pa', 'RH', 'Ta']].apply(pd.to_numeric, errors='coerce')
df_crnp[['Pa', 'RH', 'Ta']] = df_crnp[['Pa', 'RH', 'Ta']].interpolate(method='pchip', limit=24, limit_direction='both')

# Compute the pressure correction factor 
df_crnp['fp'] = crnpy.correction_pressure(pressure=df_crnp['Pa'], Pref=Pref, L=130)

# Calculate the absolute humidity (g cm^-3) and the vapor pressure correction factor
df_crnp['abs_humidity'] = crnpy.abs_humidity(df_crnp['RH'], df_crnp['Ta'])
df_crnp['fw'] = crnpy.correction_humidity(abs_humidity=df_crnp['abs_humidity'], Aref=Aref)


# Plot all the correction factors
plt.figure(figsize=(15, 7))
plt.plot(df_crnp['timestamp'], df_crnp['fp'], label='Pressure', color='tomato', linewidth=1)
plt.plot(df_crnp['timestamp'], df_crnp['fw'], label='Humidity', color='navy', linewidth=1)
plt.plot(df_crnp['timestamp'], df_crnp['fi'], label='Incoming Flux', color='olive', linewidth=1)
plt.xlabel("Date")
plt.ylabel('Correction Factor')
plt.legend()
plt.title('Correction Factors for Pressure, Humidity, and Incoming Flux')
plt.savefig(os.path.join(output_folder, "correction_factors.png"))
plt.close()


# Apply correction factors
df_crnp['total_corrected_neutrons'] = df_crnp['total_raw_counts'] * df_crnp['fw'] / (df_crnp['fp'] * df_crnp['fi'])


plt.figure(figsize=(15, 7))
plt.plot(df_crnp['timestamp'], df_crnp['total_raw_counts'], label='Raw Counts', color='black', linestyle='dashed', linewidth=.8)
plt.plot(df_crnp['timestamp'], df_crnp['total_corrected_neutrons'], label='Corrected Counts', color='teal', linewidth=.8)
plt.xlabel("Date")
plt.ylabel('Counts')
plt.legend()
plt.title('Raw and Corrected Counts')
plt.savefig(os.path.join(output_folder, "raw_corrected_counts.png"))
plt.close()
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# df를 일자별로 그룹핑하여 평균값 계산
df_daily_data = df_crnp.resample('D', on='timestamp').mean(numeric_only=True)


# Estimate lattice water (%) based on texture
lattice_water = crnpy.lattice_water(clay_content=clay_content)


# Drop NaN values in the daily resampled neutron data
# df_daily_data = df_daily_data.dropna(subset=['total_corrected_neutrons', 'Pa', 'fp', 'fi', 'fw'])


df_daily_data['VWC'] = crnpy.counts_to_vwc(df_daily_data['total_corrected_neutrons'], N0=N0_rdt, 
                                               bulk_density=soil_bulk_density, Wlat=lattice_water, Wsoc=0.01)

# Drop any NaN values
# df_daily_data = df_daily_data.dropna(subset=['VWC'])

# # Filter using the Savitzky-Golay filter, drop NaN values and timestamps
# df_crnp['VWC'] = crnpy.smooth_1d(df_crnp['VWC'], window=11, order=3, method="savitzky_golay")


# Plot VWC over time and save the plot
plt.figure(figsize=(15, 7))
plt.plot(df_daily_data.index, df_daily_data['VWC'], color='teal', linewidth=1.0)
plt.xlabel("Date")
plt.ylabel(r"Daily Average Volumetric Water Content (m3⋅m−3)")
plt.title('Daily Average CRNP Soil Moisture')
plt.savefig(os.path.join(output_folder, "daily_avg_soil_moisture.png"))
plt.close()

#----------------------------------------------------------------------------------
# Estimate sensing depth
df_daily_data['sensing_depth'] = crnpy.sensing_depth(df_daily_data['VWC'], df_daily_data['Pa'], 
                                                         df_daily_data['Pa'].mean(), 
                                                         bulk_density=soil_bulk_density, Wlat=lattice_water, method="Franz_2012")
print(f"Average sensing depth: {np.round(df_daily_data['sensing_depth'].mean(), 2)} cm")

# Compute the storage using the exponential filter
surface = df_daily_data['VWC']
subsurface = crnpy.exp_filter(df_daily_data['VWC'])
df_daily_data['storage'] = np.sum([surface*z_surface, subsurface*z_subsurface], axis=0)


# Plot the obtained time series of soil water storage
# Create a new figure and plot the data
plt.figure(figsize=(15, 7))
plt.plot(df_daily_data.index, df_daily_data['storage'], color='teal', linewidth=1.0)
plt.xlabel("Date")
plt.ylabel("Storage (mm)")
plt.title('Daily Average Soil Water Storage')
plt.savefig(os.path.join(output_folder, "daily_avg_soil_water_storage.png"))
plt.close()
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# Estimate the uncertainty of the volumetric water content
df_daily_data['sigma_VWC'] = crnpy.uncertainty_vwc(df_daily_data['total_raw_counts'], N0=N0_rdt, bulk_density=soil_bulk_density, fp=df_daily_data['fp'], fi=df_daily_data['fi'], fw=df_daily_data['fw'])

# Plot the VWC with uncertainty as a shaded area
plt.figure(figsize=(15, 7))
plt.plot(df_daily_data.index, df_daily_data['VWC'], color='black', linewidth=1.0)
plt.fill_between(df_daily_data.index, df_daily_data['VWC'] - df_daily_data['sigma_VWC'], df_daily_data['VWC'] + df_daily_data['sigma_VWC'], color='teal', alpha=0.5, label="Standard Deviation")
plt.xlabel("Date")
plt.ylabel(r"Volumetric Water Content (m3⋅m−3)")
plt.title('Volumetric Water Content with Uncertainty')
plt.legend()
plt.savefig(os.path.join(output_folder, "vwc_with_uncertainty.png"))
plt.close()

# 날짜 범위를 설정하여 NaN을 포함한 빈 날짜를 채움
all_dates = pd.date_range(start=df_daily_data.index.min(), end=df_daily_data.index.max(), freq='D')

# 누락된 날짜를 포함하도록 전체 데이터프레임을 다시 인덱싱
df_daily_data = df_daily_data.reindex(all_dates)

# 최종 엑셀 파일에 저장할 때도 누락된 날짜를 포함한 상태로 저장
output_excel_path = os.path.join(output_folder, "CRNP_SM.xlsx")
df_daily_data.to_excel(output_excel_path, index_label="Date")

print(f"Data and plots have been successfully saved to {output_folder}")