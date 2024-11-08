import pandas as pd

file_path = 'ufc-master.csv' 
df = pd.read_csv(file_path)

columns_to_keep = [
    'RedFighter', 'BlueFighter', 'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue', 'Winner',
    'RedCurrentLoseStreak', 'RedCurrentWinStreak', 'RedDraws', 'RedAvgSigStrLanded', 'RedAvgSigStrPct',
    'RedAvgSubAtt', 'RedAvgTDLanded', 'RedAvgTDPct', 'RedLongestWinStreak', 'RedLosses', 'RedTotalRoundsFought',
    'RedTotalTitleBouts', 'RedWinsByDecisionMajority', 'RedWinsByDecisionSplit', 'RedWinsByDecisionUnanimous',
    'RedWinsByKO', 'RedWinsBySubmission', 'RedWinsByTKODoctorStoppage', 'RedWins', 'RedStance', 'RedHeightCms',
    'RedReachCms', 'RedWeightLbs', 'RedAge', 'BlueCurrentLoseStreak', 'BlueCurrentWinStreak', 'BlueDraws',
    'BlueAvgSigStrLanded', 'BlueAvgSigStrPct', 'BlueAvgSubAtt', 'BlueAvgTDLanded', 'BlueAvgTDPct', 
    'BlueLongestWinStreak', 'BlueLosses', 'BlueTotalRoundsFought', 'BlueTotalTitleBouts',
    'BlueWinsByDecisionMajority', 'BlueWinsByDecisionSplit', 'BlueWinsByDecisionUnanimous', 'BlueWinsByKO',
    'BlueWinsBySubmission', 'BlueWinsByTKODoctorStoppage', 'BlueWins', 'BlueStance', 'BlueHeightCms', 
    'BlueReachCms', 'BlueWeightLbs', 'BlueAge'
]

# Filter the DataFrame to keep only the selected columns
df_filtered = df[columns_to_keep]

output_path = 'ufc-cleaned.csv' 
df_filtered.to_csv(output_path, index=False)

print("Data cleaning completed. Filtered file saved to:", output_path)
