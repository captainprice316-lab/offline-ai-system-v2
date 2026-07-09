import pyarrow.parquet as pq

pf = pq.read_table(
    r"E:\hf_ks_temp\hub\datasets--ai4bharat--indicvoices_r\snapshots"
    r"\5f4495c91d500742a58d1be2ab07d77f73c0acf8\Kashmiri\test-00000-of-00002.parquet",
    memory_map=True,
)
print("Schema:")
print(pf.schema)
print("Num rows:", len(pf))
print("Columns:", pf.column_names)

row = pf.slice(0, 1).to_pydict()
audio0 = row["audio"][0]
print("\nAudio type:", type(audio0).__name__)
if isinstance(audio0, dict):
    print("Audio keys:", list(audio0.keys()))
    for k, v in audio0.items():
        t = type(v).__name__
        ln = len(v) if hasattr(v, "__len__") else "n/a"
        print(f"  {k}: type={t} len={ln}")
elif isinstance(audio0, bytes):
    print("Audio is raw bytes, length:", len(audio0))

# Check normalized column
if "normalized" in row:
    print("\nnormalized[0]:", row["normalized"][0])
elif "transcription" in row:
    print("\ntranscription[0]:", row["transcription"][0])
