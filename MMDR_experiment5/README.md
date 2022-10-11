### Introduction
This is the experimental code for multi-frame multi-scale depth representation network by adversarial learning.
Under the feature generation, the spatial texture information divides into the content feature and style feature.
Content feature is represented by common factors in FAS, mainly including semantic features and physical attributes. 
Style feature describes some discriminative cues that be divided into two parts: domain-specific and liveness-related 
style. The content feature from various domains and environments has small distribution discrepancies because they 
share a common semantic feature space. Therefore, the style feature is vital and discriminative for FAS task. At the 
same time, we need to separate the domain-specific feature and liveness-related feature to avoid the influence of various 
environments and domains. The adversarial network will improve the generation by random shuffled and fuse the domain feature
 and liveness-related feature to synthesis more various samples for simulating complex real-life environments.

For the adversarial learning, we need to make generated content feature indistinguishable for different domain. 
The parameters of the content feature generator are optimized by maximizing the adversarial loss function, while 
the parameters of domain discriminator are optimized by minimizing it.

Specifically, the feature generators extract multi-scale low-level information. For the style feature, we obtain multi-layer
features along with the hierarchical structure in a pyramid approach to match the multi-scale low-level information and multi-scale 
texture differences. 

### Architecture
