### Overview
This is a repo of the experimental models and deep networks for face antispoofing task.
This repo will add some novel and non-trivial FAS models to improve the performance and generation
of FAS deep networks. In the meantime, this repo also will add some personal thoughts and understand for 
FAS task to build optimal FAS model.

### Source Models
1. DRIT
2. MUNIT
3. STDN

### DRN: Disentangled Representations Network
The DRN is a deep network based on disentangled representations for FAS task. The network add some 
change and novel thoughts.

### MMDR: Multi-dataset Multi-class Disentangled Representations
There are many textural differences between real and spoof faces, which can 
be partitioned into multiple levels based on scales: low-level, mid-level 
and high-level. Low-level textural differences consist of smooth content 
patterns, for example image noises in paper attacks and specular 
highlights in replay attacks. Mid-level differences include sharp patterns 
and texture, such as Moiré patterns in video attacks. High-level 
differences include chromatic balance bias and range bias, which are 
created by the interaction between PAs and capture cameras. Therefore, 
three level texture features are extracted to cover these differences 
based on multiple scales and get the fine-grained texture features for 
FAS task.