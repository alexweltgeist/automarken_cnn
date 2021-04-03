# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 10:47:12 2021

@author: christian/alex
"""

import os, shutil, glob
import random

# Download the sourcefile with all named pictures in one folder 'Datensatz'
#Source file 
sourcefile = 'C:/Users/alex/CAS_ML_local/B_Deeplearning/03_Project/Datensatz'

# split the images in folders based on the image name (here: we use brand)
for filename in os.listdir(sourcefile):
    #new_dir = filename.find('_')
    brand = filename.rsplit('_', 17)[0]
    # If folder does not exist try making new one
    try:
        os.mkdir(os.path.join(sourcefile, brand))
    # except error then pass
    except WindowsError:
        pass
    # Move the images from file to new folder based on image name
    shutil.move(os.path.join(sourcefile, filename), 
                os.path.join(sourcefile, brand, filename))

# Now split the data into train, valid, test dirctories
# 60% goes to train, 20% each to validation and test
all_subdirs = os.listdir(sourcefile)
len(all_subdirs)

for subdirs in all_subdirs:
    os.chdir(sourcefile)
    if os.path.isdir('train/'+subdirs) is False:
        #print(type(str(subdirs+'*')))
        os.makedirs('train/'+subdirs)
        os.makedirs('valid/'+subdirs)
        os.makedirs('test/'+subdirs)
        
        sample_anz = len(os.listdir(sourcefile+'/'+subdirs))
        
        for c in random.sample(glob.glob(str(subdirs+'/'+subdirs+'*')),
                               sample_anz // 10 * 6):
            shutil.copy(c, 'train/'+subdirs)
        for c in random.sample(glob.glob(str(subdirs+'/'+subdirs+'*')),
                               sample_anz // 10 * 2):
            shutil.copy(c, 'valid/'+subdirs)
        for c in random.sample(glob.glob(str(subdirs+'/'+subdirs+'*')),
                               sample_anz // 10 * 2):
            shutil.copy(c, 'test/'+subdirs)

