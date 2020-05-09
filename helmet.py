"""
Mask R-CNN based Motorcyclist and Helmet detection
Train on the Helmet dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Customized by : Sudalaiandi Raja Sudalaimuthu & 
                Jayaraman Revathi
Date          : 10th May 2020
------------------------------------------------------------

Usage: import the module (see Jupyter notebook helmet_detect.ipynb for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python helmet.py train --dataset=/path/to/helmet/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python helmet.py train --dataset=/path/to/helmet/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python helmet.py train --dataset=/path/to/helmet/dataset --weights=imagenet

    # Apply color splash to an image
    python helmet.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python helmet.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize_ext as visualize

import matplotlib.pyplot as plt

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Keep the label as 'Motorcyclist' and 'Helmet'
# Set to False if needs to be 'Motorcyclist_with_Helmet' or 'without_Helmet' instead
DEFAULT_LABEL_HELMET_SEPARATE = True 
DEBUG = False
NBR_OF_EPOCHS = 100

############################################################
#  Configurations
############################################################

class HelmetConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "helmet"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1+2 if DEFAULT_LABEL_HELMET_SEPARATE else 1+3  

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class HelmetDataset(utils.Dataset):

    def load_helmet(self, dataset_dir, subset):
        """Load a subset of the Helmet dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "annotations-json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # For collecting statistics 
        number_of_images = 0
        
        # Preset class_names and instance_count ** To-be-checked
        if DEFAULT_LABEL_HELMET_SEPARATE:
            class_names = {'Motorcyclist': 1, 
                           'Helmet': 2}
            instance_count = {'Motorcyclist': 0, 
                           'Helmet': 0}
        else :
            class_name = {'Motorcyclist_without_helmet': 1, 
                          'Helmet': 2, 
                          'Motorcyclist_with_helmet': 3}
            instance_count = {'Motorcyclist_without_helmet': 0, 
                          'Helmet': 0, 
                          'Motorcyclist_with_helmet': 0}
        
        def isSegmentConnector(r) :
            """

            Parameters
            ----------
            r : region type with format :
                {
                	"shape_attributes": {
                		"name": "polygon",
                		"all_points_x": [
                			229,231,
                		],
                		"all_points_y": [
                			272,268,
                		]
                	},
                	"region_attributes": {
                		"name": "Motorcyclist",
                		"segment_connector": " "
                	}
                    "ignore" : False
                    "with_helmet" : False
                }
                
                DESCRIPTION : checks if the given region is Connected Segment or not.

            Returns
            -------
            True if it the given region is a valid connected segment
            False if it is a stand-alone segment

            """
            if ('segment_connector' in r['region_attributes']  and
            r['region_attributes']['segment_connector'].strip()) :
                return True
            else:
                return False
        
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        
        nbr_of_ignored = 0
        if DEBUG :
            temp_r = [] # for debugging purpose
        
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                regions = a['regions'].values()
            else :
                regions = a['regions']
            
            # Connect the regions which belong to the same instance
            reg_len = len(regions)
            for i in range(reg_len):
                regions[i]['ignore'] = False
                regions[i]['with_helmet'] = False
                regions[i]['helmet_annotation'] = "NA"
                regions[i]['shape_attributes'] = [regions[i]['shape_attributes']]
            
            # Merge connected regions by checking SegmentConnector region attribute
            # Identify Motorcyclist if it has connected Helment annotation to flag accordingly
            for i, r in enumerate(regions):
                
                assert r['region_attributes']['name'] in ["Motorcyclist", "Helmet"]
                
                if not isSegmentConnector(r):
                    # check for 'Helmet' annotation found with no SegmentConnector.
                    # this should be corrected in annotation.
                    if r['region_attributes']['name'] == 'Helmet' :
                        print('Helmet annotation found without SegmentConnector : ',
                              'filename : {} annotation# : {}'.format(a['filename'], i+1))
                    continue
                # Check if there are linked instances and combine the regions accordingly
                # Check if the Helmet instance belongs to a motorcyclist to mark 'with_helmet' flag
                for j, rg in zip(range(i+1,reg_len), regions[i+1:]):
                    
                    if (not isSegmentConnector(rg) or 
                        rg['ignore'] or 
                        (rg['with_helmet'] and rg['region_attributes']['name'] == 'Helmet') or
                        (regions[i]['with_helmet'] and r['region_attributes']['name'] == 'Helmet')):
                        continue
                    
                    if (r['region_attributes']['segment_connector'].strip() == 
                        rg['region_attributes']['segment_connector'].strip()) :
                        
                        if (r['region_attributes']['name'] == 
                            rg['region_attributes']['name']) :
                            regions[i]['shape_attributes'].extend(rg['shape_attributes']) 
                            regions[j]['ignore'] = True
                            
                        elif (r['region_attributes']['name'] == 'Motorcyclist' and 
                            rg['region_attributes']['name'] == 'Helmet') : 
                            regions[i]['with_helmet'] = True
                            regions[j]['with_helmet'] = True
                            regions[i]['helmet_annotation'] = str(j)
                            
                        elif (r['region_attributes']['name'] == 'Helmet' and 
                            rg['region_attributes']['name'] == 'Motorcyclist') : 
                            regions[i]['with_helmet'] = True
                            regions[j]['with_helmet'] = True
                            regions[j]['helmet_annotation'] = str(i)
                
            polygons = []
            
            for i, r in enumerate(regions) :
                
                if r['ignore']:
                    nbr_of_ignored += 1
                    continue 
                
                class_name = r['region_attributes']['name']
                
                # Modify class_name to mark with_helmet for combined helmet detection 
                if not DEFAULT_LABEL_HELMET_SEPARATE :
                    if r['with_helmet'] and class_name == 'Motorcyclist':
                        class_name = class_name + "_with_helmet"
                    elif class_name == 'Motorcyclist':
                        class_name = class_name + "_without_helmet"
                    
                    if class_name == 'Helmet' and not r['with_helmet'] :
                        print('Helmet annotation found without matching SegmentConnector : ', 
                              'filename : {} annotation# : {}'.format(a['filename'], i+1))
                    
                if class_name not in class_names.keys() :
                    if len(class_names.keys()) > 0 :
                        class_names[class_name] = max(class_names.values()) + 1
                    else :
                        class_names[class_name] = 1
                        
                    instance_count[class_name] = 0
                    
                instance_count[class_name] += 1
                
                polygon = {
                    "polygon": r['shape_attributes'],
                    "catId" : class_names[class_name]
                    }
                polygons.append(polygon)
                
                # For debugging purpose
                if DEBUG :
                    temp_r.append( {
                        "file_name" : a['filename'],
                        "name" : class_name,
                        'annotation_nbr' : str(i),
                        "with_helmet" : r['with_helmet'],
                        "helmet_annotation_nbr" : r['helmet_annotation']
                        })

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "helmet",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)
            number_of_images += 1
            
        for class_name in list(class_names.keys()):
            self.add_class("helmet", class_names[class_name], class_name)
        
        # Statistics on datasets
        print ('\n{:*^50}'.format('Dataset statistics for ' + subset))
        print ('{:<20} {}'.format('Number of Images', number_of_images))
        print ('{:<20} {}'.format('class info', class_names))
        print ('{:<20} {}'.format('Nbr of Instances', instance_count))
        print ('{:<20} {}\n'.format('Nbr of ignored Instances', nbr_of_ignored))
        
        if DEBUG :
            import pandas as pd
            df = pd.DataFrame(data=temp_r)
            df.to_excel("DEBUG_temp.xlsx", index=False)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a helmet dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "helmet":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        
        class_ids = []
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            for each in p['polygon'] :
                rr, cc = skimage.draw.polygon(each['all_points_y'], each['all_points_x'])
                mask[rr, cc, i] = 1
            class_ids.append(p['catId'])

        # Return mask, and array of class IDs of each instance. Since we have
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "helmet":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, base_dataset_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = HelmetDataset()
    dataset_train.load_helmet(base_dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = HelmetDataset()
    dataset_val.load_helmet(base_dataset_dir, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=NBR_OF_EPOCHS,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    if DEFAULT_LABEL_HELMET_SEPARATE :
        class_names = ['BG', 'Motorcyclist', 'Helmet']
    else :
        class_names = ['BG', 'Motorcyclist_without_Helmet', 
                       'Helmet', 'Motorcyclist_with_Helmet']
        
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        # print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        ### visualize the results
        #print ('rois, class_ids, scores ', 
        #       r['rois'], r['class_ids'], r['scores'])
        print (f"{'rois':<15} - {r['rois']} \n{'class_ids':<15} - {r['class_ids']}")
        #print (f"{'scores':<15} - {r['scores']} \n{'keypoints':<15} - {r['keypoints']}")
        
        _, ax = plt.subplots(1, 1, figsize=(20, 20))
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, 
                            r['scores'], ax=ax)
        ####
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        #ax.savefig(file_name, bbox_inches='tight')
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect helmets.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/helmet/dataset/",
                        help='Directory of the Helmet dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = HelmetConfig()
    else:
        class InferenceConfig(HelmetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
