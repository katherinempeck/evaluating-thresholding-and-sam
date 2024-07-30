import pandas as pd
import random

#This script is specific to our image spreadsheet and to those column names
#To randomly select images for analysis, we filtered this spreadsheet to a single site based on the unique ID associated with that site ('KK' for Kin Kletso)
#We also filtered to only images taken at the same objective, using the unique string associated with those photos ('10x2')

#Parameters
#To resample, change these parameters:

#Replace string below with path to actual CSV
in_csv_path = 'path_to_csv.csv'
#Replace string below with desired output path for sampled CSV
out_csv_path = 'path_to_desired_output.csv'
#Replace string below with site ID for sampled site
siteid = 'KK'
#Replace string below with objective type
objective = '10x2'
#Replace integer below with desired number of images to sample
sample_size = 20

#Run
#Should not need to change anything below this line
#Read in photo log CSV
photos = pd.read_csv(in_csv_path)
#Select only sherds from desired site (i.e. if PhotoFileName contains site string)
only_one_site = photos[photos['PhotoFileName'].str.contains(siteid, na = False)]
#Drop photos taken of rims
#Rims have too much resin around the edges and are not appropriate for this kind of analysis
#If the PhotoType column contains rim, remove that row
no_rim = only_one_site[only_one_site['PhotoType'].str.contains('rim') == False]
#Drop if notes have blue (i.e. where there's lots of blue at edges - this happens ocassionally with som ebody sherds, and will cause both methods to over-report voids)
to_drop = [i for i,r in no_rim.iterrows() if 'blue' in str(r['Notes'])]
no_edge = no_rim.drop(index = to_drop)
obj10x2 = no_edge[no_edge['PhotoFileName'].str.contains(objective)]
#Get unique sample ids from the column DissID
group = obj10x2.groupby('DissID')
df2 = group.apply(lambda x: x['DissID'].unique())
#Turn that into a list
unique_ids = [i[0] for i in df2]
#Randomly sample 20 IDs from that list
sample20 = random.sample(unique_ids, sample_size)
#Define an empty list to hold the results
photo_samples = []
#Iterate through the randomly selected sample IDs
for id in sample20:
    #Get only the rows in the dataframe from that site
    sliced = obj10x2[obj10x2['DissID'].str.contains(id)]
    #Make that into a list
    filelist = sliced['PhotoFileName'].tolist()
    #Randomly select a photo from that list
    photo = random.sample(filelist, 1)
    #Append that to the photo_samples list
    #Slicing the dataframe above puts each value in a list, so get the actual string by grabbing the first object in the list
    photo_samples.append(photo[0])
#Query the original dataframe and only select files which have a PhotoFileName in the sampled list
sampled_dataframe = obj10x2.query("PhotoFileName.isin(@photo_samples)")
#Save this to a new CSV so we have all photo information
sampled_dataframe.to_csv(out_csv_path)