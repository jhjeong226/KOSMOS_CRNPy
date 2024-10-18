## This code was developed by Jaehwan Jeong for the general processing of CRNP data in the KOSMOS project
## Unauthorized modification and distribution of this code are prohibited.
## Draft version: 14 October 2024
## Last updated: 18 October 2024
## This code uses functions from CRNPy (Peraza Rud et al., 2024).
import crnpy
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Define geographic coordinates: type Station information
lat = 37.7049111
lon = 128.0316412
soil_bulk_density = 1.44
clay_content = 0.35

# Define Calibration period - Daily conversion 시 온전할 수 있도록, 시작시간과 끝 시간은 00시와 23시로 사용하는 것을 권장. 
calibration_start = pd.to_datetime("2023-03-26 00:00")
calibration_end = pd.to_datetime("2023-11-04 23:00")

# Function to remove outliers using Median Absolute Deviation (MAD)
def remove_outliers(df, column, threshold=3):
    median = np.median(df[column])
    mad = np.median(np.abs(df[column] - median))
    mad_scaled = np.abs(df[column] - median) / (mad + 1e-6)
    return df[mad_scaled < threshold]

## =============================================================================================================================================
## =========================================================== Load Data  =============================================================
## 전처리가 완료된 데이터 읽어오기

# Load the in-situ soil data
FDR_datapath = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\02.Output\01.Preprocessed\HC_FDR_input.xlsx"
df_soil = pd.read_excel(FDR_datapath)

# Create an ID for each soil profile using their respective latitude and longitude
df_soil['ID'] = df_soil['latitude'].astype(str) + '_' + df_soil['longitude'].astype(str)

# Load the CRNP station data
CRNP_datapath = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\02.Output\01.Preprocessed\HC_CRNP_input.xlsx"
df_crnp = pd.read_excel(CRNP_datapath, names=['Timestamp', 'RN', 'Ta', 'RH', 'Pa', 'WS', 'WS_max', 'WD_VCT', 'N_counts'])
df_crnp['timestamp'] = pd.to_datetime(df_crnp['Timestamp'], errors='coerce')

# Ensure 'timestamp' column has valid values
if df_crnp['timestamp'].isna().all():
    raise ValueError("All timestamp values are invalid. Please check the 'Timestamp' format in your CRNP data.")


# Filter data within the calibration period
idx_period = (df_crnp['timestamp'] >= calibration_start) & (df_crnp['timestamp'] <= calibration_end)
df_crnp = df_crnp[idx_period]

# We use only one detector
df_crnp['total_raw_counts'] = df_crnp['N_counts']

## =============================================================================================================================================
## =========================================================== Neutron correction  =============================================================

# Atmospheric corrections
df_crnp[['Pa', 'RH', 'Ta']] = df_crnp[['Pa', 'RH', 'Ta']].apply(pd.to_numeric, errors='coerce')
df_crnp[['Pa', 'RH', 'Ta']] = df_crnp[['Pa', 'RH', 'Ta']].interpolate(method='pchip', limit=24, limit_direction='both')

df_crnp['abs_humidity'] = crnpy.abs_humidity(df_crnp['RH'], df_crnp['Ta'])

Pref = df_crnp['Pa'].mean()
Aref = df_crnp['abs_humidity'].mean()

# Absolute humidity and correction factors
df_crnp['fp'] = crnpy.correction_pressure(pressure=df_crnp['Pa'], Pref=Pref, L=130)
df_crnp['fw'] = crnpy.correction_humidity(abs_humidity=df_crnp['abs_humidity'], Aref=Aref)

# Neutron flux correction
nmdb = crnpy.get_incoming_neutron_flux(calibration_start, calibration_end, station="ATHN", utc_offset=9)
df_crnp['incoming_flux'] = crnpy.interpolate_incoming_flux(nmdb['timestamp'], nmdb['counts'], df_crnp['timestamp'])
df_crnp['fi'] = crnpy.correction_incoming_flux(incoming_neutrons=df_crnp['incoming_flux'], incoming_Ref=df_crnp['incoming_flux'].iloc[0])

# Apply correction factors to neutron counts
df_crnp['total_corrected_neutrons'] = df_crnp['total_raw_counts'] * df_crnp['fw'] / (df_crnp['fp'] * df_crnp['fi'])

# ** Remove outliers based on total_corrected_neutrons using MAD **
df_crnp = remove_outliers(df_crnp, 'total_corrected_neutrons', threshold=3)

# Daily average of neutron counts
df_crnp['date'] = df_crnp['timestamp'].dt.date
daily_neutron_avg = df_crnp.groupby('date')['total_corrected_neutrons'].mean()


## ==========================================================================================================================================

results = pd.DataFrame(columns=['date', 'Daily_N', 'Field_SM'])


for single_date in pd.date_range(start=calibration_start, end=calibration_end, freq='D'):
    daily_crnp = df_crnp[df_crnp['date'] == single_date.date()]
    if not daily_crnp.empty:
        daily_soil = df_soil[df_soil['Date'].dt.date == single_date.date()]

        if not daily_soil.empty:

            # Field_SM, _ = crnpy.nrad_weight(df_crnp['abs_humidity'].mean(),
            #                                      df_soil['theta_v'], df_soil['distance_from_station'],
            #                                      df_soil['FDR_depth'], rhob=df_soil['bulk_density'].mean(), method="Kohli_2015")
            
            Field_SM, _ = crnpy.nrad_weight(daily_crnp['abs_humidity'].mean(),
                                                daily_soil['theta_v'],
                                                daily_soil['distance_from_station'],
                                                daily_soil['FDR_depth'],
                                                profiles=daily_soil['ID'],
                                                rhob=daily_soil['bulk_density'].mean(),
                                                p=daily_crnp['Pa'].mean(),
                                                method = "Schron_2017")
            
            # Calculate Daily_N for the day
            Daily_N = daily_crnp['total_corrected_neutrons'].mean()

            # Append the results for the day
            results = pd.concat([results, pd.DataFrame({
                'date': [single_date],
                'Field_SM': [Field_SM],
                'Daily_N': [Daily_N]
            })])

# Estimate lattice water (%) based on texture
lattice_water = crnpy.lattice_water(clay_content=clay_content)

# Define RMSE function and optimize N0 to minimize RMSE between Field_SM and calculated VWC from neutron counts
def objective(N0):
    results['CRNP_SM'] = crnpy.counts_to_vwc(results['Daily_N'], N0, bulk_density=soil_bulk_density, Wlat=lattice_water, Wsoc=0.01)
    return np.sqrt(np.mean((results['CRNP_SM'] - results['Field_SM']) ** 2))

# Minimize the RMSE
result = minimize(objective, x0=1000, method='Nelder-Mead')
N0_opt = result.x[0]

# Print the optimized N0
print(f"Optimized N0: {N0_opt}")
print(f"Pref : {Pref}")
print(f"Aref : {Aref}")

# Save the results with the estimated theta_v
results['CRNP_SM'] = crnpy.counts_to_vwc(results['Daily_N'], N0_opt, bulk_density=soil_bulk_density, Wlat=0.03, Wsoc=0.01)
output_file_with_estimates = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\02.Output\01.Preprocessed\Daily_Calibration.xlsx"
results.to_excel(output_file_with_estimates, index=False)
print(f"Daily results with estimates saved to {output_file_with_estimates}")