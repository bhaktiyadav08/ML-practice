import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dialysis_dataset(filename='dialysis_data.csv'):
    """
    Generates COMPLETE dialysis dataset and saves as CSV
    """
    # Create sample data
    num_samples = 10000  # 10,000 data points

    # Generate timestamps (every 2 seconds for 5.5 hours)
    start_time = datetime(2024, 1, 15, 9, 0, 0)
    timestamps = [start_time + timedelta(seconds=i*2) for i in range(num_samples)]

    # Generate sensor data
    data = {
        'timestamp': timestamps,
        'pressure_arterial': np.random.normal(130, 8, num_samples).clip(100, 200),
        'pressure_venous': np.random.normal(60, 5, num_samples).clip(40, 100),
        'blood_flow': np.random.normal(300, 10, num_samples).clip(250, 350),
        'dialysate_flow': np.random.normal(500, 20, num_samples).clip(450, 550),
        'conductivity': np.random.normal(14.0, 0.3, num_samples).clip(13.0, 15.0),
        'temperature': np.random.normal(37.0, 0.2, num_samples).clip(36.5, 37.5),
        'ultrasonic': np.random.normal(550, 30, num_samples).clip(400, 700),
        'tds_value': np.random.normal(2100, 100, num_samples).clip(1900, 2300),
        'failure_type': ['normal'] * num_samples
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"✅ Dataset saved as: {filename}")
    print(f"📊 Size: {len(df)} rows × {len(df.columns)} columns")
    print(f"💾 File size: {len(df.to_csv(index=False)) / 1024:.1f} KB")

    return df

# RUN THIS TO CREATE CSV
dataset = generate_dialysis_dataset('my_dialysis_data.csv')
