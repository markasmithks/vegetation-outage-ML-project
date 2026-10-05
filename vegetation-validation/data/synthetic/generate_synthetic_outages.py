"""Generate fictional event records from aggregate-calibrated distributions.
Requires Python, numpy and pandas. Contains no source records or identity lookup.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PARAMETERS = json.loads(r'''{"month_weights": [34, 24, 28, 66, 93, 94, 302, 106, 51, 42, 35, 14], "types": {"Feeder": {"weight": 13, "mean": [4.815, 5.338], "cov": [[0.611, 0.08], [0.08, 0.265]]}, "Line Section": {"weight": 551, "mean": [4.897, 2.695], "cov": [[0.898, 0.043], [0.043, 1.234]]}, "Meter": {"weight": 141, "mean": [4.487, 0.693], "cov": [[2.313, 0.0], [0.0, 0.03]]}, "No Outage": {"weight": 20, "mean": [1.27, 0.0], "cov": [[5.281, 0.0], [0.0, 0.03]]}, "Substation": {"weight": 4, "mean": [5.162, 6.747], "cov": [[0.34, 0.105], [0.105, 0.261]]}, "Transformer": {"weight": 100, "mean": [4.823, 0.831], "cov": [[0.871, -0.009], [-0.009, 0.086]]}}, "weather": {"1": {"temperature_mean": -0.3, "temperature_sd": 2.4, "wind_mean": 21.0, "wind_sd": 8.0, "wet_probability": 0.77, "rain_mean": 0.91}, "2": {"temperature_mean": 9.0, "temperature_sd": 8.3, "wind_mean": 24.3, "wind_sd": 7.8, "wet_probability": 0.48, "rain_mean": 0.98}, "3": {"temperature_mean": 15.3, "temperature_sd": 6.2, "wind_mean": 26.7, "wind_sd": 10.4, "wet_probability": 0.12, "rain_mean": 0.2}, "4": {"temperature_mean": 18.4, "temperature_sd": 6.3, "wind_mean": 27.0, "wind_sd": 11.1, "wet_probability": 0.22, "rain_mean": 4.15}, "5": {"temperature_mean": 21.6, "temperature_sd": 3.7, "wind_mean": 18.0, "wind_sd": 9.7, "wet_probability": 0.32, "rain_mean": 1.07}, "6": {"temperature_mean": 24.6, "temperature_sd": 3.8, "wind_mean": 13.7, "wind_sd": 8.5, "wet_probability": 0.32, "rain_mean": 1.0}, "7": {"temperature_mean": 24.8, "temperature_sd": 3.3, "wind_mean": 10.3, "wind_sd": 4.8, "wet_probability": 0.48, "rain_mean": 2.71}, "8": {"temperature_mean": 26.5, "temperature_sd": 3.8, "wind_mean": 13.8, "wind_sd": 6.4, "wet_probability": 0.17, "rain_mean": 2.27}, "9": {"temperature_mean": 22.9, "temperature_sd": 5.6, "wind_mean": 13.6, "wind_sd": 6.9, "wet_probability": 0.25, "rain_mean": 7.43}, "10": {"temperature_mean": 21.5, "temperature_sd": 5.4, "wind_mean": 22.3, "wind_sd": 11.0, "wet_probability": 0.36, "rain_mean": 1.67}, "11": {"temperature_mean": 11.4, "temperature_sd": 7.7, "wind_mean": 24.5, "wind_sd": 15.2, "wet_probability": 0.34, "rain_mean": 1.55}, "12": {"temperature_mean": 4.5, "temperature_sd": 7.0, "wind_mean": 14.7, "wind_sd": 10.3, "wet_probability": 0.27, "rain_mean": 0.65}}}''')


def generate(output_dir=None, seed=20261005):
    output_dir = Path(output_dir or Path(__file__).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Fictional inventory: 24 substations, each with three feeders.
    inventory = [(s, f) for s in range(1, 25) for f in range(1, 4)]
    feeder_weights = rng.lognormal(0, 0.6, len(inventory))
    feeder_weights /= feeder_weights.sum()
    types = list(PARAMETERS['types'])
    type_weights = np.array([PARAMETERS['types'][t]['weight'] + 2 for t in types], float)
    type_weights /= type_weights.sum()
    month_weights = np.array(PARAMETERS['month_weights'], float)
    month_weights /= month_weights.sum()
    outputs = []
    for year, n in [(2023, 350), (2024, 479)]:
        rows = []
        for _ in range(n):
            month = int(rng.choice(np.arange(1, 13), p=month_weights))
            substation, feeder = inventory[int(rng.choice(len(inventory), p=feeder_weights))]
            event_type = str(rng.choice(types, p=type_weights))
            p = PARAMETERS['types'][event_type]
            if event_type == 'No Outage':
                duration, customers = 0.0, 0
            else:
                sample = rng.multivariate_normal(p['mean'], p['cov'])
                duration = round(float(np.clip(np.expm1(sample[0]), 1, 2400)), 2)
                customers = int(np.clip(np.rint(np.expm1(sample[1])), 1, 2500))
            w = PARAMETERS['weather'][str(month)]
            temperature = round(float(np.clip(rng.normal(w['temperature_mean'], w['temperature_sd']), -20, 45)), 1)
            wind = round(float(np.clip(rng.normal(w['wind_mean'], w['wind_sd']), 0, 80)), 1)
            rain = round(float(min(rng.gamma(0.8, w['rain_mean'] / 0.8), 50)), 1) if rng.random() < w['wet_probability'] else 0.0
            rows.append([year, month, substation, feeder, duration, customers,
                         int(round(duration * customers)), event_type, temperature, rain, wind])
        frame = pd.DataFrame(rows, columns=['Year', 'Month', 'Substation_ID', 'Feeder_ID',
            'Duration_Minutes', 'Customers_Out', 'Customer_Minutes', 'Event_Type',
            'Temperature_C', 'Precipitation_mm', 'Wind_Speed_kmh'])
        assert len(frame) == n and not frame.isna().any().any()
        assert (frame.Customer_Minutes == (frame.Duration_Minutes * frame.Customers_Out).round().astype(int)).all()
        zero = frame.Event_Type == 'No Outage'
        assert (frame.loc[zero, ['Duration_Minutes', 'Customers_Out', 'Customer_Minutes']] == 0).all().all()
        frame.to_csv(output_dir / f'{year}_synthetic_outages.csv', index=False, encoding='utf-8-sig')
        outputs.append(frame)
    return outputs


if __name__ == '__main__':
    frames = generate()
    for frame in frames:
        print(f"{int(frame.Year.iloc[0])}: {len(frame)} synthetic rows")
