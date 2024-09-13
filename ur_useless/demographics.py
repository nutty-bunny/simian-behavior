import pandas as pd 

mkid_rhesus_file = '/Users/similovesyou/Desktop/qts/mkid_rhesus.csv'
df_mkid_rhesus = pd.read_csv(mkid_rhesus_file)
df_mkid_rhesus.dropna(subset=['Age'], inplace=True)

print(df_mkid_rhesus)
print()

mkid_misc_file = '/Users/similovesyou/Desktop/qts/mkid_misc.csv'
df_mkid_misc = pd.read_csv(mkid_misc_file)
df_mkid_misc.dropna(subset=['Age'], inplace=True)
df_mkid_misc.drop(columns=['specie'], inplace=True)
df_mkid_misc.rename(columns={'group': 'specie'}, inplace=True)

print(df_mkid_misc)
print()

rhesus_index = df_mkid_misc[df_mkid_misc['specie'] == 3]
mkid_rhesus = pd.concat([df_mkid_rhesus, rhesus_index], ignore_index=True)
mkid_tonkean = df_mkid_misc[df_mkid_misc['specie'] != 3]

print(mkid_rhesus)
print()
print(mkid_tonkean)

mkid_rhesus_file = '/Users/similovesyou/Desktop/qts/demographics/mkid_rhesus.csv'
mkid_tonkean_file = '/Users/similovesyou/Desktop/qts/demographics/mkid_tonkean.csv'

mkid_rhesus.to_csv(mkid_rhesus_file, index=False)
mkid_tonkean.to_csv(mkid_tonkean_file, index=False)
