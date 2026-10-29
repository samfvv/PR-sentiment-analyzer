import pandas as pd
from datasets import load_dataset

print("--- STEP 1: Fetching Social Media Data from Hugging Face ---")
# Load the specific dataset you chose
split = {'train': 'train_df.csv', 'validation': 'val_df.csv', 'test': 'test_df.csv'}
hf_data = load_dataset('Sp1786/multiclass-sentiment-analysis-dataset', split='train')

# Convert it into a standard Pandas DataFrame
df_social = hf_data.to_pandas()

# Keep only the columns we need
df_social = df_social[['text', 'label']]
print(f"Loaded {len(df_social)} social media rows.")


print("\n--- STEP 2: Loading Corporate Data from Kaggle ---")
# 1. Read the CSV, tell it there are no headers, assign our own column names, and fix the text encoding!
df_corp = pd.read_csv(
    'financial.csv', 
    header=None, 
    names=['label', 'text'], 
    encoding='latin1'
)

# 2. Crucial step: Clean up the label words to numbers so they match Hugging Face exactly:
# (0 = Negative, 1 = Neutral, 2 = Positive)
corp_label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df_corp['label'] = df_corp['label'].map(corp_label_map)

print(f"Loaded {len(df_corp)} corporate financial news rows.")

# Update this section inside your load_and_mix.py file

print("\n--- STEP 2.5: Injecting Manual Twitter Slang Data ---")
manual_data = [
    # --- NEGATIVE OVERTONES (0) ---
    {"text": "this whole situation is so messy ngl, massive L for the brand", "label": 0},
    {"text": "they really thought they did something here... its giving nothing burger", "label": 0},
    {"text": "the new layout is so mid, nobody asked for this", "label": 0},
    {"text": "cringe corporate response, they are completely missing the point", "label": 0},
    {"text": "the fall off must be studied", "label": 0},
    {"text": "gatekeeping features behind a paywall is a terrible look", "label": 0},
    {"text": "the comms team is scrambling, absolute clown behavior", "label": 0},

    # --- NEUTRAL / INDIFFERENT (1) ---
    {"text": "idk i'm completely indifferent about the redesign tbh", "label": 1},
    {"text": "just saw the announcement, standard behavior i guess", "label": 1},
    {"text": "it is what it is, doesn't really affect how I use the app", "label": 1},
    {"text": "maybe its the friends we make along the way", "label": 1},
    {"text": "unpopular opinion but the new patch is just okay", "label": 1},

    # --- POSITIVE OVERTONES (2) ---
    {"text": "the PR team completely cooked with this launch video", "label": 2},
    {"text": "huge W for the engineering team today, this feature rules", "label": 2},
    {"text": "they actually listened to feedback! absolute dub", "label": 2},
    {"text": "this marketing campaign is honestly top tier", "label": 2},
    {"text": "the world gagged", "label": 2},
    {"text": "clutch update, fixed all my issues immediately", "label": 2}
]

df_manual = pd.DataFrame(manual_data)
print(f"Loaded {len(df_manual)} custom manual slang rows.")

print("\n--- STEP 3: Blending and Shuffling Data ---")
# Combine both datasets into one single master DataFrame
master_dataset = pd.concat([df_social, df_corp, df_manual], ignore_index=True)

# Drop any row that happens to have missing/blank text
master_dataset = master_dataset.dropna()

# Shuffle the rows completely so the AI doesn't learn patterns based on order
master_dataset = master_dataset.sample(frac=1, random_state=42).reset_index(drop=True)


print("\n--- STEP 4: Saving Master Dataset ---")
# Save the final product as a clean CSV file
master_dataset.to_csv('pr_master_dataset.csv', index=False)
print(f"🎉 Success! 'pr_master_dataset.csv' created with {len(master_dataset)} total mixed rows.")