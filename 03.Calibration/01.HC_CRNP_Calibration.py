import crnpy
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import root

# Function to remove outliers using Median Absolute Deviation (MAD)
def remove_outliers(df, column, threshold=3):
    median = np.median(df[column])
    mad = np.median(np.abs(df[column] - median))
    mad_scaled = np.abs(df[column] - median) / (mad + 1e-6)
    return df[mad_scaled < threshold]

## =============================================================================================================================================
## ===========================================================  Field Survey Data  =============================================================

# Load the soil samples data and the CRNP dataset using pandas
FDR_datapath = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\02.Output\01.Preprocessed\HC_FDR_input.xlsx"
df_soil = pd.read_excel(FDR_datapath)

# Define start and end of field survey calibration
calibration_start = pd.to_datetime("2023-03-26 00:00")
calibration_end = pd.to_datetime("2023-11-04 23:00")

# Create an ID for each soil profile using their respective latitude and longitude
df_soil['ID'] = df_soil['latitude'].astype(str) + '_' + df_soil['longitude'].astype(str)



## =============================================================================================================================================


## =============================================================================================================================================
## ===========================================================  Station CRNP Data  =============================================================

# Load the station data
CRNP_datapath = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\99.Data\01.HC\02.Output\01.Preprocessed\HC_CRNP_input.xlsx"
df_crnp = pd.read_excel(CRNP_datapath, names=['Timestamp', 'RN', 'Ta', 'RH', 'Pa', 'WS', 'WS_max', 'WD_VCT', 'N_counts'])

# Define geographic coordinates
lat = 37.7049111
lon = 128.0316412
soil_bulk_density = 1.44

# Parse dates
df_crnp['timestamp'] = pd.to_datetime(df_crnp['Timestamp'], errors='coerce')

# Ensure 'timestamp' column has valid values
if df_crnp['timestamp'].isna().all():
    raise ValueError("All timestamp values are invalid. Please check the 'Timestamp' format in your CRNP data.")

# Filter data within the calibration period
idx_period = (df_crnp['timestamp'] >= calibration_start) & (df_crnp['timestamp'] <= calibration_end)
df_crnp = df_crnp[idx_period]

# Compute total neutron counts by adding the counts from both probe detectors
df_crnp['total_raw_counts'] = crnpy.total_raw_counts(df_crnp[['N_counts']])

# Atmospheric corrections
df_crnp[['Pa', 'RH', 'Ta']] = df_crnp[['Pa', 'RH', 'Ta']].apply(pd.to_numeric, errors='coerce')
df_crnp[['Pa', 'RH', 'Ta']] = df_crnp[['Pa', 'RH', 'Ta']].interpolate(method='pchip', limit=24, limit_direction='both')

# Absolute humidity and correction factors
df_crnp['abs_humidity'] = crnpy.abs_humidity(df_crnp['RH'], df_crnp['Ta'])
df_crnp['fp'] = crnpy.correction_pressure(pressure=df_crnp['Pa'], Pref=df_crnp['Pa'].mean(), L=130)
df_crnp['fw'] = crnpy.correction_humidity(abs_humidity=df_crnp['abs_humidity'], Aref=df_crnp['abs_humidity'].mean())

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


# Weight field average soil moisture
field_theta_v, w = crnpy.nrad_weight(df_crnp['abs_humidity'].mean(), df_soil['theta_v'], df_soil['distance_from_station'], 
                                     df_soil['FDR_depth'], rhob=df_soil['bulk_density'].mean(), method="Kohli_2015")

field_theta_v, w = crnpy.nrad_weight(df_crnp['abs_humidity'].mean(), df_soil['theta_v'], df_soil['distance_from_station'], 
                                     df_soil['FDR_depth'], profiles=df_soil['ID'], rhob=df_soil['bulk_density'].mean(),
                                     p=df_crnp['Pa'].mean(), method = "Schron_2017")


# Determine the mean corrected counts during the calibration survey
idx_cal_period = (df_crnp['timestamp'] >= calibration_start) & (df_crnp['timestamp'] <= calibration_end)
mean_cal_counts = df_crnp.loc[idx_cal_period, 'total_corrected_neutrons'].mean()

print(f"Mean volumetric Water content during calibration survey: {round(field_theta_v,3)}")
print(f"Mean corrected counts during calibration: {round(mean_cal_counts)} counts")

# Define the function for which we want to find the roots
VWC_func = lambda N0 : crnpy.counts_to_vwc(mean_cal_counts, N0, bulk_density=soil_bulk_density, Wlat=0.03, Wsoc=0.01) - field_theta_v

# Make an initial guess for N0
N0_initial_guess = 1000

# Find the root
sol = int(root(VWC_func, N0_initial_guess).x[0])

# Print the solution
print(f"The solved value for N0 is: {sol}")



# # Daily average of soil moisture data
# df_soil['timestamp'] = pd.to_datetime(df_soil['Date'], errors='coerce')
# df_soil['date'] = df_soil['timestamp'].dt.date
# daily_soil_avg = df_soil.groupby('date')['theta_v'].mean()

# # Combine the daily averages into a single DataFrame
# daily_data = pd.DataFrame({
#     'date': daily_neutron_avg.index,
#     'daily_neutron_avg': daily_neutron_avg.values,
#     'daily_soil_avg': daily_soil_avg.reindex(daily_neutron_avg.index).values
# }).dropna()





# # Objective function to minimize the difference between the estimated and actual soil moisture
# def objective(N0):
#     estimated_theta_v = crnpy.counts_to_vwc(daily_data['daily_neutron_avg'], N0, bulk_density=soil_bulk_density, Wlat=0.03, Wsoc=0.01)
#     return np.sum((estimated_theta_v - daily_data['daily_soil_avg'])**2)

# # Minimize the objective function to find the optimal N0
# result = minimize(objective, x0=1000, method='Nelder-Mead')
# N0_opt = result.x[0]

# # Calculate the estimated soil moisture (theta_v) using the optimized N0
# daily_data['estimated_theta_v'] = crnpy.counts_to_vwc(daily_data['daily_neutron_avg'], N0_opt, bulk_density=soil_bulk_density, Wlat=0.03, Wsoc=0.01)

# # Display results
# print(f"Optimized N0: {N0_opt}")

# print(f"Pref: {df_crnp['Pa'].mean()}")

# print(f"Aref: {df_crnp['abs_humidity'].mean()}")

# # Save daily corrected neutron, soil moisture, and estimated theta_v to Excel
# output_file = r"C:\Users\USER\Desktop\Workbox\00.KIHS_CRNP\10.Data\01.CRNPy\01.HC\00.CRNPy_output\HC_Calibration_Daily.xlsx"
# daily_data.to_excel(output_file, index=False)

# print(f"Daily calibration data saved to {output_file}")
