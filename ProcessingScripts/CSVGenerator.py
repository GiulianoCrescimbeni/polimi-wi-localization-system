import os
import subprocess

def estrai_csv_da_pcapng(file_path, output_csv):
    fields = [
        "frame.number",
        "frame.time",
        "wlan.sa",
        "wlan.seq",
        "frame.len",
        "radiotap.dbm_antsignal"
    ]

    cmd = [
        "tshark",
        "-r", file_path,
        "-Y", "wlan.fc.type_subtype == 8",
        "-T", "fields",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-E", "occurrence=f"
    ]
    
    for field in fields:
        cmd += ["-e", field]

    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Creato {output_csv}")
    except subprocess.CalledProcessError as e:
        print(f"Errore su {file_path}: {e}")

cartella_pcap = "SampleData/"
for file in os.listdir(cartella_pcap):
    if file.endswith(".pcapng"):
        nome_base = os.path.splitext(file)[0]
        input_path = os.path.join(cartella_pcap, file)
        output_path = os.path.join(cartella_pcap, f"{nome_base}.csv")
        estrai_csv_da_pcapng(input_path, output_path)