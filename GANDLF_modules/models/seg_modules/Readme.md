### Description
Link to the paper: https://arxiv.org/pdf/1907.02110.pdf. 

This folder contains the smaller modules that are integrated to form the larger Inc U Net arch.

Notes:

- Architecture is defined on Page 5 Figure 1 of the paper. 
- This is the module implementation on Page 6 Figure 2 (diagram on the right) of the above mentioned paper. 
- In summary, this consists of 4 parallel pathways each with f/4 feature maps (f is the number of feature maps of the input to the InceptionModule. These 4 feature maps (or channels) are concatenated after being processed by the Inception Module)