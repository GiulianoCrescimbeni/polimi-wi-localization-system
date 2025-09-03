import os
import pandas as pd
import numpy as np

def fill_and_filter_csv(folder, sa_list, output_folder, fill_value=-100, min_samples=50, augment_std=1.0):
    os.makedirs(output_folder, exist_ok=True)
    target_sas = [sa.lower() for sa in sa_list]

    for file in os.listdir(folder):
        if file.endswith('.csv') and "(" in file and ")" in file:
            try:
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path)

                df['wlan.sa'] = df['wlan.sa'].astype(str).str.strip().str.lower()

                df_filtered = df[df['wlan.sa'].isin(target_sas)].copy()
                output_rows = []

                for sa in target_sas:
                    sa_df = df_filtered[df_filtered['wlan.sa'] == sa].copy()

                    if sa_df.empty:
                        for _ in range(min_samples):
                            dummy_row = {
                                "frame.number": -1,
                                "frame.time": "dummy",
                                "wlan.sa": sa,
                                "wlan.seq": -1,
                                "frame.len": -1,
                                "radiotap.dbm_antsignal": fill_value,
                                "is_synthetic": True
                            }
                            output_rows.append(dummy_row)
                        print(f"{file} - SA {sa} missing, added 50 dummy")
                    else:
                        count = len(sa_df)
                        output_rows.extend(sa_df.to_dict(orient='records'))

                        if count < min_samples:
                            n_extra = min_samples - count
                            mean_rssi = sa_df["radiotap.dbm_antsignal"].astype(float).mean()
                            synthetic_rssi = np.random.normal(loc=mean_rssi, scale=augment_std, size=n_extra)

                            for rssi in synthetic_rssi:
                                synth_row = {
                                    "frame.number": -2,
                                    "frame.time": "synthetic",
                                    "wlan.sa": sa,
                                    "wlan.seq": -1,
                                    "frame.len": -1,
                                    "radiotap.dbm_antsignal": rssi
                                }
                                output_rows.append(synth_row)
                            print(f"{file} - SA {sa} augmented by {n_extra} entries")

                df_output = pd.DataFrame(output_rows)
                output_path = os.path.join(output_folder, file)
                df_output.to_csv(output_path, index=False)
                print(f"Processed: {file}, total rows: {len(df_output)}")

            except Exception as e:
                print(f"Error processing {file}: {e}")

target_sas = [
    "f2:42:89:7e:73:bd",
    "40:ee:dd:0e:38:54",
    "3a:07:16:cb:a7:f4",
    "c4:ea:1d:53:75:c3"
]

fill_and_filter_csv(
    folder="SampleData/",
    sa_list=target_sas,
    output_folder="ProcessedData/",
    fill_value=-100,
    min_samples=50,
    augment_std=1.0
)