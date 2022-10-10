### Introduction
This is the experimental code for multi-frame multi-scale depth representation network at video-level .
Under the spatial feature extractor, we add the ConvGRU to extract the temporal information 
between different but adjacent frame images at the same video level. On the one hand, we 
designed a spatial gradient feature module to obtain the multi-scale spatial feature and predicted
depth map. On the other hand, we also utilized the ConvGRU to extract the temporal information by the 
spatial depth map between multi-frame at the same video. By them, we extract the spatial and temporal
feature and fuse them to better detect the living and spoofing face.

At the same time, we use the depth map to auxiliary supervision to better train the deep network.
By redesigning the adjacent depth loss, we train the corresponding network with more efficient information.

### Architecture
![MMDR_with_ConvGRU architecture](:ConvGRU.png)