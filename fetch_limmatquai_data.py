import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns

def download_year_data(year):
    url = f"https://data.stadt-zuerich.ch/dataset/ted_taz_verkehrszaehlungen_werte_fussgaenger_velo/download/{year}_verkehrszaehlungen_werte_fussgaenger_velo.csv"
    filename = f"{year}_data.csv"
    
    print(f"Downloading data for {year}...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
        else:
            print(f"Failed to download data for {year}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading data for {year}: {e}")
        return None

def process_year_data(filename, limmatquai_ids, station_names):
    try:
        df = pd.read_csv(filename)
        
        # Filter for pedestrian counting stations only (33 and 3279)
        pedestrian_stations = ['33', '3279']
        df = df[df["FK_STANDORT"].astype(str).isin(pedestrian_stations)]
        
        # Convert timestamp
        df["DATUM"] = pd.to_datetime(df["DATUM"])
        
        # Filter for time between Dec 31 17:00 and Jan 1 07:00
        mask = ((df["DATUM"].dt.month == 12) & (df["DATUM"].dt.day == 31) & (df["DATUM"].dt.hour >= 17)) | \
               ((df["DATUM"].dt.month == 1) & (df["DATUM"].dt.day == 1) & (df["DATUM"].dt.hour <= 7))
        df = df[mask]
        
        # Drop rows where both FUSS_IN and FUSS_OUT are NaN
        df = df.dropna(subset=['FUSS_IN', 'FUSS_OUT'], how='all')
        
        return df
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None
    finally:
        # Clean up downloaded file
        if os.path.exists(filename):
            os.remove(filename)

def plot_hourly_heatmap(df):
    # Create hour column (0-23)
    df['Hour'] = df['DATUM'].dt.hour
    
    # Create year column that represents the New Year's Eve (e.g., "2014-15")
    df['NYE'] = df.apply(lambda x: f"{x['DATUM'].year}-{str(x['DATUM'].year + 1)[-2:]}" 
                        if x['DATUM'].month == 12 
                        else f"{x['DATUM'].year-1}-{str(x['DATUM'].year)[-2:]}", axis=1)
    
    # Calculate total pedestrians per hour
    hourly_data = df.groupby(['NYE', 'Hour'])[['FUSS_IN', 'FUSS_OUT']].sum().sum(axis=1).reset_index()
    
    # Create a complete range of hours for each NYE period
    all_hours = pd.DataFrame({'Hour': list(range(17, 24)) + list(range(0, 8))})
    all_nyes = pd.DataFrame({'NYE': sorted(df['NYE'].unique())})
    complete_grid = all_nyes.assign(key=1).merge(all_hours.assign(key=1), on='key').drop('key', axis=1)
    
    # Merge with actual data, filling NaN with 0
    hourly_data = complete_grid.merge(hourly_data, on=['NYE', 'Hour'], how='left').fillna(0)
    
    # Pivot the data for the heatmap
    pivot_data = hourly_data.pivot(index='NYE', columns='Hour', values=0)
    
    # Sort index to ensure chronological order
    pivot_data = pivot_data.sort_index()
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Create heatmap with integer formatting for annotations
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.0f', 
                cbar_kws={'label': 'Total Pedestrians'})
    
    # Customize the plot
    plt.title('Hourly Pedestrian Traffic on New Year\'s Eve at Limmatquai\n(17:00-07:00)', 
             fontsize=14, pad=20)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('New Year\'s Eve', fontsize=12)
    
    # Adjust hour labels
    hours = list(range(17, 24)) + list(range(0, 8))
    plt.xticks(range(len(hours)), hours, rotation=0)
    
    # Add some padding to the layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('limmatquai_hourly_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nPlot saved as 'limmatquai_hourly_heatmap.png'")
    
    # Print peak hours for each year
    print("\nPeak Hours by Year:")
    print("==================")
    for nye in pivot_data.index:
        hour_data = pivot_data.loc[nye]
        peak_hour = hour_data.idxmax()
        peak_count = hour_data.max()
        formatted_hour = f"{peak_hour:02d}:00"
        print(f"{nye}: {formatted_hour} ({int(peak_count):,} pedestrians)")

def main():
    # Read station metadata from local CSV
    stations_df = pd.read_csv("taz.view_eco_standorte.csv")
    print("\nAvailable stations with 'Limmatquai' in name:")
    print("===========================================")
    limmatquai_stations = stations_df[stations_df["bezeichnung"].str.contains("Limmatquai", na=False)]
    print(limmatquai_stations[["id1", "bezeichnung", "abkuerzung"]])
    
    # Get list of Limmatquai station IDs
    limmatquai_ids = limmatquai_stations["id1"].astype(str).tolist()
    
    station_names = {}
    if not stations_df.empty:
        # Create a mapping of station IDs to names
        station_names = dict(zip(stations_df["id1"].astype(str), stations_df["bezeichnung"]))
    
    # Process data for each year from 2014 to present
    all_data = []
    current_year = datetime.now().year
    
    for year in range(2014, current_year + 1):
        filename = download_year_data(year)
        if filename:
            df = process_year_data(filename, limmatquai_ids, station_names)
            if df is not None and not df.empty:
                all_data.append(df)
    
    if all_data:
        # Combine all years of data
        df = pd.concat(all_data, ignore_index=True)
        
        # Convert numeric columns
        numeric_cols = ["VELO_IN", "VELO_OUT", "FUSS_IN", "FUSS_OUT"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by station and datetime
        df = df.sort_values(["FK_STANDORT", "DATUM"])
        
        # Save to CSV
        output_file = "limmatquai_new_year_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        
        # Create heatmap of hourly data
        plot_hourly_heatmap(df)
        
        # Print overall summary
        print("\nOverall Summary:")
        print("===============")
        print(f"Years covered: {df['DATUM'].dt.year.min()}-{df['DATUM'].dt.year.max()}")
        print(f"Total records: {len(df)}")
        total_pedestrians = df[["FUSS_IN", "FUSS_OUT"]].sum().sum()
        print(f"Total pedestrian count: {int(total_pedestrians):,}")
        
        # Calculate busiest hour overall
        df['Hour'] = df['DATUM'].dt.hour
        hourly_totals = df.groupby('Hour')[['FUSS_IN', 'FUSS_OUT']].sum().sum(axis=1)
        busiest_hour = hourly_totals.idxmax()
        busiest_count = hourly_totals.max()
        print(f"Busiest hour overall: {busiest_hour:02d}:00 (average of {int(busiest_count/len(df['DATUM'].dt.year.unique())):,} pedestrians per year)")
    else:
        print("\nNo data found for any year.")

if __name__ == "__main__":
    main()
