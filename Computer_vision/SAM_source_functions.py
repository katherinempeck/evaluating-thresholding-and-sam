#Import necessary libraries

#Import SAM functions
#This needs to be cloned and installed from GitHub
#e.g. pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
#Import other necessary libraries
import os
#Open and convert images
import cv2
#Display images and plots
import matplotlib.pyplot as plt
#Work with masks (which are created as numpy arrays)
import numpy as np
#For the purposes of this analysis, we're treating the masks as geographic features
#This allows us to vectorize the masks and make area and shape calculations easily
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import rasterio
import rasterio.features
#Calculations
import math

def get_masks(image_path, model_type, checkpoint):
    """Return SAM masks from image

    Args:
        image_path (str): image file path
        model_type (str): SAM model type, e.g. 'vit_h'
        checkpoint (str): filepath to model .pth file (downloaded from the FB ai public files repo)
    """
    image = cv2.imread(image_path)
    w = image.shape[0]
    h = image.shape[1]
    image = cv2.resize(image, (int(h/2), int(w/2)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Segment image
    sam = sam_model_registry[model_type](checkpoint = checkpoint)
    #Necessary to make sure Colab is actually using the virtual GPU
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    return masks

def remove_biggest_mask(mask_array):
    """Given SAM mask outputs, remove the largest mask (usually the "background" mask)

    Args:
        mask_array (numpy array): output from get_masks() defined above

    Returns:
        new_masks: array of masks with larges mask dropped
    """
    areas = [mask_array[i]['area'] for i in range(0, len(mask_array))]
    new_masks = [x for x in mask_array if not (max(areas) == x.get('area'))]
    return new_masks

def separate_voids_particles(image_path:str, mask_array:np.array) -> list:
    """Given a petrographic slide and masks, separate masks into particles and voids

    Args:
        image_path (str): original image
        mask_array (np.array): SAM mask associated with image (output of get_masks())

    Returns:
        list: list of masks in order index 0: particles, index 1: voids
    """
    image = cv2.imread(image_path)
    w = image.shape[0]
    h = image.shape[1]
    image = cv2.resize(image, (int(h/2), int(w/2)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(image)
    #This is probably the line that's making it run opposite the way I think it should
    #But now it's hard-coded that way...
    im_list = [r, g, b]
    void_masks = []
    particle_masks = []
    for m in mask_array:
        result = []
        for band in im_list:
            masked = band * m['segmentation']
            arr = np.where(masked==0, np.nan, masked)
            av = np.nanmean(arr)
            result.append(av)
            # print(result)
        max_index = result.index(max(result))
        if max_index == 2:
            particle_masks.append(m['segmentation'])
        else:
            void_masks.append(m['segmentation'])
    return [particle_masks, void_masks]

def vectorize(array:np.array) -> gpd.GeoDataFrame:
    """Vectorize masks to an unprojected GeoDataFrame

    Args:
        array (np.array): output of SAM (ideally, after separation)

    Returns:
        gpd.GeoDataFrame: GeoDataFrame representing SAM masks
    """
    features = []
    for m in array:
        m = np.float32(m)
        mask = m < 255
        shapeDict = rasterio.features.shapes(m, mask = mask)
        feats = []
        geoms = []
        for key, value in shapeDict:
            feats.append(value)
            geoms.append(shape(key))
        features_gdf = gpd.GeoDataFrame({'feats': feats, 'geometry': geoms})
        #It was also rasterizing the whole boundary, so we can drop that
        features_gdf = features_gdf.drop(features_gdf[features_gdf.feats == 0.0].index)
        #Add a (dimensionless) area column so we can drop
        features_gdf['area'] = features_gdf['geometry'].area
        #Can get individual geometries from this list if you want using features[n].loc[0, 'geometry']
        features.append(features_gdf)
    vector_features = pd.concat(features)
    return vector_features

def calculate_metrics(folder_path:str, model_type:str, checkpoint:str) -> pd.DataFrame:
    """For each image in a folder, segment, and calculate total area (as a percent) of voids, particles, and the two combined.
    Args:
        folder_path (str): Folder path for each image
        model_type (str): SAM model type e.g. vit_h
        checkpoint (str): Path to SAM model weights (.pth file)

    Returns:
        pd.DataFrame: DataFrame of metrics, by image name
    """
    results = []
    for filename in os.listdir(folder_path):
        im_path = f'{folder_path}/{filename}'
        image = cv2.imread(im_path)
        #Reduce image size for quicker processing
        w = image.shape[0]
        h = image.shape[1]
        image = cv2.resize(image, (int(h/2), int(w/2)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = get_masks(im_path, model_type, checkpoint)
        masks = remove_biggest_mask(masks)
        particles, voids = separate_voids_particles(im_path, masks)
        particles = vectorize(particles)
        voids = vectorize(voids)
        image_pixels = image.shape[0]*image.shape[1]
        total_mask_area = sum([m['area'] for m in masks])
        void_and_particle = total_mask_area/image_pixels*100
        particle_area = sum(list(particles['area']))
        void_area = sum(list(voids['area']))
        particle_percent = (particle_area/image_pixels)*100
        void_percent = (void_area/image_pixels)*100
        results.append([filename, void_and_particle, void_percent, particle_percent])
        print(f'{filename} processed')
    cols = ['image_name', 'Void + NotVoid%', 'Void%', 'NotVoid%']
    df = pd.DataFrame(results, columns = cols)
    return df

def calculate_metrics_with_images(folder_path:str, model_type:str, checkpoint:str, image_fpath:str) -> list:
    """For each image in a folder, segment, and calculate total area (as a percent) of voids, particles, and the two combined.
       This function also saves a binary .tif and a .tif with each segmented object circled and saves the masks/vectorized masks for later use.

    Args:
        folder_path (str): Folder path for each image
        model_type (str): Chosen SAM model e.g. vit_h
        checkpoint (str): Path to SAM model weights (.pth file)
        image_fpath (str): Output folder for saving segmented images

    Returns:
        list: [DataFrame of metrics, by image name, list of masks, list of vectorized masks]
    """
    results = []
    saved_masks = []
    saved_geoms = []
    for filename in os.listdir(folder_path):
        im_path = f'{folder_path}/{filename}'
        image = cv2.imread(im_path)
        #Reduce image size for quicker processing
        w = image.shape[0]
        h = image.shape[1]
        image = cv2.resize(image, (int(h/2), int(w/2)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = get_masks(im_path, model_type, checkpoint)
        masks = remove_biggest_mask(masks)
        particles, voids = separate_voids_particles(im_path, masks)
        particles = vectorize(particles)
        voids = vectorize(voids)
        image_pixels = image.shape[0]*image.shape[1]
        total_mask_area = sum([m['area'] for m in masks])
        void_and_particle = total_mask_area/image_pixels*100
        particle_area = sum(list(particles['area']))
        void_area = sum(list(voids['area']))
        particle_percent = (particle_area/image_pixels)*100
        void_percent = (void_area/image_pixels)*100
        results.append([filename, void_and_particle, void_percent, particle_percent])
        saved_masks.append([filename, masks])
        saved_geoms.append([filename, particles['geometry'].tolist(), voids['geometry'].tolist()])
        #Save an image with each object outlined (combined)
        fig, ax = plt.subplots()
        ax.imshow(image)
        particles.plot(ax = ax, facecolor = 'none', edgecolor = 'white')
        voids.plot(ax = ax, facecolor = 'none', edgecolor = 'white')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(f'{image_fpath}/{filename.split(".")[0]}_segmented.tif', dpi = 300)
        plt.close()
        #Save an image with each object outlined, voids in blue and particles in white
        fig, ax = plt.subplots()
        ax.imshow(image)
        particles.plot(ax = ax, facecolor = 'none', edgecolor = 'white')
        voids.plot(ax = ax, facecolor = 'none', edgecolor = 'blue')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(f'{image_fpath}/{filename.split(".")[0]}_segmented_color_coded.tif', dpi = 300)
        plt.close()
        #Merge the masks and save as a binary image
        segs = [m['segmentation'] for m in masks]
        s = sum(segs)
        combined = np.where(s > 0, 1, s)
        plt.imsave(f'{image_fpath}/{filename.split(".")[0]}_binary.tiff', combined, cmap='Greys')
    cols = ['image_name', 'Void + NotVoid%', 'Void%', 'NotVoid%']
    df = pd.DataFrame(results, columns = cols)
    return df, saved_masks, saved_geoms