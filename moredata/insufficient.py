import pandas as pd

file_path = "masterdataframe.csv" 
df = pd.read_csv(file_path)

fighter_name_columns = ["fighter", "opponent"]  

columns_to_keep = fighter_name_columns + list(df.select_dtypes(include=["number"]).columns)

df_cleaned = df[columns_to_keep]
df_cleaned = df_cleaned[df_cleaned.isnull().sum(axis=1) <= 30]

cleaned_file_path = "lowdata.csv"  
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to: {cleaned_file_path}")
