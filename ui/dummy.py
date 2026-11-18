import pandas as pd
import numpy as np

# Generate 20 dummy GPS points around a location
lon = np.linspace(77.5946, 77.5956, 20)
lat = np.linspace(12.9716, 12.9726, 20)
time = np.arange(20)

df = pd.DataFrame({'lon': lon, 'lat': lat, 'time': time})
df.to_csv('dummy.track', sep=' ', index=False, header=False)
print("Created dummy.track successfully!")
