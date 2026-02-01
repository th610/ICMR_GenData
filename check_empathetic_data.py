import pandas as pd

df = pd.read_parquet('data/external/empathetic_train.parquet')

print(f"Total rows (utterances): {len(df)}")
print(f"Unique conversations: {df['conv_id'].nunique()}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nSample conversation structure:")
sample_conv = df[df['conv_id'] == df['conv_id'].iloc[0]]
print(f"Conversation ID: {sample_conv['conv_id'].iloc[0]}")
print(f"Number of utterances: {len(sample_conv)}")
print(sample_conv[['utterance_idx', 'speaker_idx', 'utterance']])
