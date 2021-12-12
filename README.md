# Single-Image-Reflection-Removal
This Repository contains the codes for Image Reflection Removal Techniques

<hr/>

We surveyed following techniques and evaluated many results over the same. We have implemented the best two techniques to do fast single image reflection removal

1. Averaging
2. Independent Component Analysis
3. Layer Separation using Relative Layer Smoothness
4. Reflection Supression using convex optimization
5. Bidirectional Estimation using Neural Nets

Here we have followed Single Image Layer Separation using Relative Layer Smoothness[1] and Fast Single Image Reflection Suppression via Convex Optimzation[2] papers. These are independent Python implementation of both techniques and we have combined them together to produce better results.


References:
1. Li, Y., & Brown, M. S. (2014). Single image layer separation using relative smoothness. 2014 IEEE Conference on Computer Vision and Pattern Recognition. https://doi.org/10.1109/cvpr.2014.346
2. Yang, Y., Ma, W., Zheng, Y., Cai, J.-F., & Xu, W. (2019). Fast single image reflection suppression via convex optimization. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2019.00833
