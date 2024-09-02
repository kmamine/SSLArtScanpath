import cv2
import os 
from tqdm.auto import tqdm
Path = '/media/amine/DataDisk8T/amine/wikiart_dataset_reduced/'
for im in tqdm(os.listdir(Path)):
    
    image1  = cv2.imread('/media/amine/DataDisk8T/amine/wikiart_augmented/abstraction/'+im)
    image2  = cv2.imread('/media/amine/DataDisk8T/amine/wikiart_augmented/cubism__Picasso/'+im)
    image3  = cv2.imread('/media/amine/DataDisk8T/amine/wikiart_augmented/magnolia/'+im)
    image4  = cv2.imread('/media/amine/DataDisk8T/amine/wikiart_augmented/Starry_Night/'+im)
    image5  = cv2.imread('/media/amine/DataDisk8T/amine/wikiart_augmented/Tsunami/'+im)
    if (image1 is not None) and (image2 is not None) and (image3 is not None) and (image4 is not None) and (image5 is not None):
        image1 = cv2.resize(image1,(230,230))
        image2 = cv2.resize(image2,(230,230))
        image3 = cv2.resize(image3,(230,230))
        image4 = cv2.resize(image4,(230,230))
        image5 = cv2.resize(image5,(230,230))

        cv2.imwrite('/media/amine/DataDisk8T/maroine/BarlowTwins/Dataset/Style1/'+im,image1)
        cv2.imwrite('/media/amine/DataDisk8T/maroine/BarlowTwins/Dataset/Style2/'+im,image2)    
        cv2.imwrite('/media/amine/DataDisk8T/maroine/BarlowTwins/Dataset/Style3/'+im,image3)
        cv2.imwrite('/media/amine/DataDisk8T/maroine/BarlowTwins/Dataset/Style4/'+im,image4)
        cv2.imwrite('/media/amine/DataDisk8T/maroine/BarlowTwins/Dataset/Style5/'+im,image5)