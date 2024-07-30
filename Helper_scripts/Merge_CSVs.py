import pandas as pd
import os

#ImageJ batch processing outputs an individual CSV for each image
#This script merges all results into one CSV

#Parameters

#Change this variable the folder that includes all the CSVs
foldername = 'C:/Path/to/folder/csvs'
output_csv_name = 'merged_output.csv'

#Run

#Shouldn't need to change anything below this line
files_to_merge = []
for f in os.listdir(foldername):
    if f.endswith('.csv'):
        csv = pd.read_csv(f'{foldername}/{f}')
        csv['Image_Name'] = f.split('.')[0]
        files_to_merge.append(csv)
new_output = pd.concat(files_to_merge)

#Save output in the same folder as the defined foldername variable above
new_output.to_csv(f'{foldername}/{output_csv_name}')