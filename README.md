# Single-Image-Reflection-Removal
This Repository contains the codes for Image Reflection Removal Techniques

<hr/>

We surveyed following techniques and evaluated many results over the same. We have implemented the best two techniques to do fast single image reflection removal out of these following techniques:

1. Averaging
2. Independent Component Analysis
3. Layer Separation using Relative Layer Smoothness
4. Reflection Supression using convex optimization
5. Bidirectional Estimation using Neural Nets

Here we have followed Single Image Layer Separation using Relative Layer Smoothness[1] and Fast Single Image Reflection Suppression via Convex Optimzation[2] papers. These are independent Python implementation of both techniques and we have combined them together to produce better results. Let's have a look at few examples with our code:

Sign Board ***with*** Reflection |  Sign Board ***without*** Reflection
:-------------------------:|:-------------------------:
![](https://github.com/Ratansingh648/Single-Image-Reflection-Removal/blob/main/test%20data/sign_board.jpg)  |  ![](https://github.com/Ratansingh648/Single-Image-Reflection-Removal/blob/main/results/sign_board_result.png)

Suitcase ***with*** Reflection |  Suitcase ***without*** Reflection
:-------------------------:|:-------------------------:
![](https://github.com/Ratansingh648/Single-Image-Reflection-Removal/blob/main/test%20data/suitcase.png)  |  ![](https://github.com/Ratansingh648/Single-Image-Reflection-Removal/blob/main/results/suitcase_result.png)

You can notice few remenant reflections near by the edges of the suitcase. This is an expected behavior due to the assumptions that we have made in our modelling. The reflection near high frequency edges are prone to escape from suppression and removal. Apart from those, almost all of the reflection has been removed from the image. 

</hr>

Our code works very well in following scenarios due to assumptions made in modelling the solution:

1. When the actual image is on focus, not the reflection. When reflection is the main focus, the actual image is blurred which denigrates the output quality in general.
2. When reflection in image is not too dominant. If reflection is very bright, it is considered as part of actual image itself.

For example in this image, the reflection of tubelight is too strong in image and hence it is not considered for the removal. However, the reflection of passengers, grab handles etc. being not too dominant are removed by our code.

Train Window ***with*** Reflection |  Train Window ***without*** Reflection
:-------------------------:|:-------------------------:
![](https://github.com/Ratansingh648/Single-Image-Reflection-Removal/blob/main/test%20data/train_window.jpg)  |  ![](https://github.com/Ratansingh648/Single-Image-Reflection-Removal/blob/main/results/train_window.png)

These assumptions are true in most of the cases. In few cases, we have noticed that overall color of image is impacted which can be fixed by color equalization.


References:
1. Li, Y., & Brown, M. S. (2014). Single image layer separation using relative smoothness. 2014 IEEE Conference on Computer Vision and Pattern Recognition. https://doi.org/10.1109/cvpr.2014.346
2. Yang, Y., Ma, W., Zheng, Y., Cai, J.-F., & Xu, W. (2019). Fast single image reflection suppression via convex optimization. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2019.00833
