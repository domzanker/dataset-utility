
# DATASET_UTILITIES
This repo includes utility classes that is are needed in order to create camera BEV, LiDAR gridmaps and road boundary labels.

## Installation
You can install this package by running   
`pip install [-e] dataset-utilities`  
from the repos path.
##### camera.py
Includes several classes for handeling sensor data processing.  

* Sensor  
   base class

* Camera  
   inherits from *Sensor*  
   transforms images points to ground and vice-versa. 
   Handles automatic cropping of horizon.

* BirdEyeView  
   inherits from *Camera*  
   adds BEVTransformations to Camera Class

* Lidar  
   inherits from *Sensor*  
   Keep track of reference frame for point-cloud.  
   Export point-cloud as pcd file.

##### bev_compositor.py  
Stitch several BirdEyeView Cameras together and create a unified top view.

##### grid_map.py
Create a simple grid map from LiDAR point-clouds using ray casting.

##### transformation.py
Provide a simple class for isometric transformation.
