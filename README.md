# Image-Reflection-Removal
This Repository contains the codes for Image Reflection Removal Techniques

<hr/>

###  Akhil - Seeing Deeply and Bidirectionally: A deep learning approach for Single Image Reflection Removal

#### Main Idea - 

***Estimate Reflection (R) image and utilize it to estimate Background (B) image using a cascaded Deep Neural Network.***

#### Background - 

Traditionally, special patterns such as ghost cue etc. have been used as Prior knowledge to help isolate relfection element from the overall image. Drawing motivation from the same - this paper,

- Utilizes Strong Edges to identify the background scene.
- Trained on images synthesized with highly blurred reflection layers.

#### Highlight:

Instead of training a network to estimate B along from I, ***authors show that estimating not only B, but also the reflection (a seemingly unnecessary step), can significantly improve the quality of reflection removal.***

The network in this paper has been trainied in order to reconstruct the scenes on both sides of the reflection surface - and in cascade they use B to estimate R, and R to estimate B - ***therefore the network is called Bidirectional.***

Where 

**I** : Original Image

**B** : Background of Original Image

**R** : Reflection of Original Image

<hr/>


### Osama's Paper idea and implementation details


<hr/>


### Ratan - Single Image Layer Separation using Relative Smoothness

This paper is based on idea that an input image can be written as

**I = L<sub>1</sub> + L<sub>2</sub>**

<br/>

Where 

L<sub>1</sub> : Image with long tail distribution of gradients (original image)

L<sub>2</sub> : Image with short tail distribution of gradients (Reflection)



<hr/>


### Rishabh's paper ideas and implementation details



Contributors: Akhil Shukla, Osama Trabelsi, Ratan Singh, Rishabh Patel
