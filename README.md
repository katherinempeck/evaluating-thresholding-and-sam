# Evaluating image thresholding and computer vision methods for ceramic fabric analysis
This repository contains code and demo notebooks for "Evaluating image thresholding and computer vision methods for ceramic fabric analysis" poster presented by Genevieve Woodhead, Katherine Peck, and Saige Young at the 2024 Pecos Conference, Friday, August 2nd.

**Abstract:** 
Ceramic fabrics—the specific combinations of clay, inclusions, and voids that make up ceramic vessel bodies—can help answer broad questions about potters’ traditions, decision-making, and communities of practice. Archaeologists identify fabric groups by spotting meaningful differences in petrographic thin sections. To do this, they often focus on characterizing (e.g., by type, size, sorting) and estimating the area percentage of inclusions and voids. Image thresholding algorithms can assist in deriving these metrics by masking particles out from the background clay matrix according to hue, saturation, and brightness. Conversely, computer vision models like Segment Anything (SAM) mask discrete objects by drawing on their training dataset of segmented images. These binary masks can then be used to generate useful metrics for characterizing ceramic fabrics. In this poster, we present a feasibility study that compares image thresholding and computer vision (SAM). We analyzed 20 petrographic images of Kin Kletso (Chaco Canyon) pottery using ImageJ’s default thresholding method to determine the percent area of voids and particles in each slide. We segmented the same slides using SAM, via Python, and derived the same metrics. We then assessed if there were statistically significant differences between each method’s outputs.

## Repository Contents

The computer vision tests were conducted using Segment Anything (SAM), a computer vision model developed by Kirrilov et al. (2023) based on a training dataset of segmented images. We ran the procedure in Google Colaboratory to make use of its virtual GPUs, so some of the setup is specific to Colab (these lines are noted with comments). The folder **Computer_vision** contains the Python functions used to segment the input images and calculate metrics as well as a demo Jupyter notebook. 

We conducted the thresholding using ImageJ's (Schnieder et al. 2012) default thresholding algorithm. We did automatic batch thresholding and manual thresholding (where ecah image threshold value was defined manually using the slider tool). The folder **Image_thresholding** contains ImageJ macro code we used for the automatic batch thresholding.

The folder **Helper_scripts** contains some additional Python scripts we used to randomly select images for analysis and combine output CSVs.

The folder **Results_and_stats** contains a Jupyter Notebook with the statistical analysis we conducted on the method outputs.

## References Cited
Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, et al. 2023. “Segment Anything.” In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4015–26.  
<br>
Schneider, Caroline A., Wayne S. Rasband, and Kevin W. Eliceiri. "NIH Image to ImageJ: 25 years of image analysis." *Nature Methods* 9, no. 7 (2012): 671-675.
