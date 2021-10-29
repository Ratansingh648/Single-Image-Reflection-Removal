# Image-Reflection-Removal
This Repository contains the codes for Image Reflection Removal Techniques

<hr/>

###  Akhil - Seeing Deeply and Bidirectionally: A deep learning approach for Single Image Reflection Removal

#### LONG story SHORT:

I = B +  R

Training dataset has I as input and B and R as ground truths available.

Testing dataset has I as input and only B as groundtruth (no R available)

1) I -> B     (learn a mapping function to estimate B)
2) B -> R    (use the learnt mapping function to estimate R)
3) R -> B    (use the learnt mapping function to estimate B again - thereby making B prediction more accurate)

Evaluation is done in SSIM and PSNR


#### Main Idea - 

***Estimate Reflection (R) image and utilize it to estimate Background (B) image using a cascaded Deep Neural Network.***

- Instead of training a network to estimate B along from I, ***authors show that estimating not only B, but also the reflection (a seemingly unnecessary step), can significantly improve the quality of reflection removal.***

- The network in this paper has been trainied in order to reconstruct the scenes on both sides of the reflection surface - and in cascade they use B to estimate R, and R to estimate B - ***therefore the network is called Bidirectional.***

#### Background - 

Traditionally, special patterns such as ghost cue etc. have been used as prior knowledge to help isolate relfection element from the overall image. 
Drawing motivation from the same - this paper,
- Utilizes Strong Edges to identify the background scene.
- Trained on images synthesized with highly blurred reflection layers.

#### Final Goal:
Final goal is to learn a mapping function F(·) to predict the background image bB = F(I) from an observed image I. Instead of training only on the image pairs (I,B)’s, we impose the ground truth reflection layers R’s to boost the training of F(·) by training on a set of triplets {(It,Bt,Rt)}<sup>N</sup><sub>t=1</sub>. Note that R<sub>t</sub>’s are only used in training, not in testing.

<hr/>

### Osama's Paper idea and implementation details

***Single Image Reflection Separation with Perceptual Losses:*** 

#### Summary: 

We can separate reflection from a single image using a deep neural network with perceptual losses that exploit the low-level and high-level information of an image.  

Key points of the process:  

We need to train a network with low-level and high-level image features. 

To do this We create a dataset of real-world images with reflection and corresponding ground-truth transmission layers for quantitative evaluation and model training. 

The loss function includes two perceptual losses: 

a feature loss from a visual perception network 

an adversarial loss that encodes characteristics of images in the transmission layers 

 What makes this paper innovative is adding a new type of losss which is an exclusion loss that enforces pixel-level layer separation 

#### Implementation: 
![Lossimplementation](https://user-images.githubusercontent.com/64674291/139358248-dd54581d-b69c-455b-b458-24eda3920be4.PNG)
 

#### Results: 

We validate our method through comprehensive quantitative experiments and show that our approach outperforms state-of-the-art reflection removal methods in PSNR, SSIM, and perceptual user study. 

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
Fast Deep Embedded Single Image Reflection Removal

<hr/>

Contributors: Akhil Shukla, Osama Trabelsi, Ratan Singh, Rishabh Patel
