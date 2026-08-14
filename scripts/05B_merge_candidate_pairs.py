import os
import pandas as pd

print("=" * 70)
print("MERGING CANDIDATE PAIR CHUNKS")
print("=" * 70)

CHUNK_DIR = "processed_data/candidate_chunks"

files = sorted(
    f for f in os.listdir(CHUNK_DIR)
    if f.endswith(".pkl")
)

print(f"Chunk files found: {len(files)}")

frames = []

for i, file in enumerate(files, start=1):

    path = os.path.join(CHUNK_DIR, file)

    df = pd.read_pickle(path)

    frames.append(df)

    print(f"[{i}/{len(files)}] Loaded {file} ({len(df):,} rows)")

candidate_pairs = pd.concat(
    frames,
    ignore_index=True
)

print("\nFinal dataset")

print(candidate_pairs.shape)

candidate_pairs.to_pickle(
    "processed_data/candidate_pairs.pkl"
)

candidate_pairs.to_csv(
    "processed_data/candidate_pairs.csv",
    index=False
)

print("\nSaved:")

print("processed_data/candidate_pairs.pkl")

print("processed_data/candidate_pairs.csv")

print("\nDONE")