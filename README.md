# ImageClassification-CNNModel
### ABSTRACT

In this paper, we propose a deep learning approach for detecting and classifying nuclei in histology images of colon cancer. Deep Learning approaches have been proven to produce encouraging results on histology images as per various studies. The objective of our study is to efficiently detect cell nucleus at center and classify cell nuclei into a class based on the location of nucleus on the cell. For Detection, a stacked autoencoder based approach is also been proposed with unsupervised fusion using Softmax classifier. For Classification, The image dataset will be trained using a Sequential model and will be measured on Accuracy and F-1 score.

### INTRODUCTION

Colorectal cancer is the third most commonly diagnosed cancer and the second leading cause of cancer death in men and women combined in the US. The American Cancer Society estimates that this year 95,520 people will be diagnosed with colon cancer, and 50,260 will die from this disease. On an average, the lifetime risk of developing colon cancer is one in 23 for men and women combined. There has been a rapid increase in screening for colorectal cancer (CRC) over the past several years in North America. This could paradoxically lead to worsening outcomes if the system is not adapted to deal with the increased demand. Wait times for cancer diagnosis and treatment are a persistent concern to the public and have received increasing attention in many countries over the past decade. Hence arises a need to develop an efficient technique to diagnose cancer based on cell structure. The massive urge behind us picking this experiment also falls along the same line of thought, to make diagnosis agile and accurate. However, it comes with its own challenges. Few factors that hinder the automatic approach for cell nuclei detection and classification could be cell heterogeneity, complex tissue structure, image quality. By implementing right measures in our models, we can overcome a few hindrances if not all. The qualitative and quantitative approaches analysis of different types of tumors at cellular level speaks volumes in terms of better understanding of tumor but also explore various options for cancer treatment.

Our approach is segmented int o 2 tasks – Detection and Classification. For Detection we have used an approach that generates a map which points to the nucleus center, referred to a PMap. The detection quality has a greater impact after processing PMap. The motive of detection is to detect all nuclei in an image by location center position regardless of class labels. For classification, the objective is to classify the nuclei into one of the 4 labeled classes: Epithelial, Fibroblast, Inflammatory, Others. This can be achieved by referring to the nuclei coordinates provided in the dataset and masking them with the already available images. The image dataset will be trained using a Sequential model and will be measured on Accuracy and F-1 score.
The dataset consists of 100 H&E stained histology images of colorectal adenocarcinomas. The images have a common size of 500 x 500 pixels. A total number of 29,756 nuclei were marked on the cells. The cell nuclei are categorized into 4 classes. In total, there are 7,722 epithelial, 5,712 fibroblast, 6,971 inflammatory, 2,039 others. 

### BACKGROUND

Several researches are being carried out in digital pathology field, trying to improve the task of nuclei segmentation, detection and classification due to the inherent complex nature of the images. We are considering 2 of these tasks and treating them as separate problems. In this decade many proposed nuclei detection techniques considering morphological features such as symmetry & stability of nuclear region as features with graph cut, Difference of Gaussian and other methods.   But many of these methods fail to address some of the factors which hinder automatic detection & classification such as the inherent cellular heterogeneity (irregular chromatin structure, cluttered nuclei) and inferior image quality due to poor tissue preparation process. In paper, focus was on nuclei segmentation considering intensity, morphology & texture as features with AdaBoost classifier. And some works were focused on nuclei segmentation which is not straightforward due to complex tissue architecture. 

There have been many deep learning methods proposed in recent years, especially from 2016. Which produce promising results in nuclei classification for large H&E stained images. In 2016, proposed spatially constrained neural networks for nuclei detection considering probability mask of adjacent pixels of the nuclei and ensemble CNN for classification, resulting highest average F1 score. In 2017, [5] used SVM, AdaBoost & DCNN on different histology image datasets of prostate, breast, renal clear cell, and renal papillary cell cancer, producing 91.2% testing accuracy.   
For nuclei detections, there are mainly 2 approaches. One using density-based methods without any prior segmentation, and other requires prior detection or segmentation. In [4], regression model is used with fully convolutional networks and patch-based input is used instead of end-to-end input images. A stacked autoencoder based approach is also been proposed with unsupervised fusion using Softmax classifier. Many CNN based models have shown significant improvement in performance.    

### APPROACH

1.Detection

Nuclei Detection

For cell detection, we have used single pixel annotated input. Where each pixel co-ordinate represents center of the nuclei. We have constructed a circle of radius 3 pixels around this center given in input dataset mat file. We are using patches of image rather than entire 500 x 500 image to train faster. We have randomly split the input images into 74 train images, 10 validation, and 16 test images. Then we have constructed 16 128 x 128 pixels out of a single input image and used as input to the network.
We are considering a U-net based architecture, which is most powerful architecture for image segmentation task. Since our nuclei detection task shares the characteristics of segmentation, we have considered this model. 
The building blocks are
-	Down Sampling: BatchNormalization  Conv2D  Conv2D  MaxPooling  Dropout 
-	Up Sampling: BatchNormalization  Conv2D  Conv2D  UpSampling  Dropout 
-	With 3 x 3 kernels and stride = 1, stride = 2 for pooling layers
Since the number of training images are less for deep networks, we have considered data augmentation of random rotation up to 50 degrees, shear range of 0.5, image is horizontally & vertically flipped. We have used Adam Optimizer with 0.001 learning rate and Binary Cross Entropy loss function

2.Classification

For classification, first the 100-image dataset is loaded into Classification Folder. Each image folder has a .bmp image file and a .mat nuclei co-ordinates file one for each class(total 4 .mat files).
 
Subsequently, test train and validation folders have been created by allotting 70 images to train, 20 images to test and 10 images to validation folders respectively.
After loading the images. One of the image,
 
Masking: With respect to the nuclei coordinates given in the mat files, each image must be masked. In plain words, masking is hiding what’s not needed and highlighting what’s needed for our classification. We have set the pixel value to 1 to the image. This has been done to images for each class.

Model: Sequential model with conv2D layers and ReLU activation function.

Layers:
•	Convolution: Convolutional layers convolve around the image to detect edges, lines, blobs of colors and other visual elements. Convolutional layers hyperparameters are the number of filters, filter size, stride, padding and activation functions for introducing non-linearity.

•	MaxPooling: Pooling layers reduces the dimensionality of the images by removing some of the pixels from the image. Maxpooling replaces a n x n area of an image with the maximum pixel value from that area to downsample the image.

•	Dropout: Dropout is a simple and effective technique to prevent the neural network from overfitting during the training. Dropout is implemented by only keeping a neuron active with some probability p and setting it to 0 otherwise. This forces the network to not learn redundant information.

•	Flatten: Flattens the output of the convolution layers to feed into the Dense layers.

•	Dense: Dense layers are the traditional fully connected networks that maps the scores of the convolutional layers into the correct labels with some activation function (SoftMax used here)

Activation functions:

Activation layers apply a non-linear operation to the output of the other layers such as convolutional layers or dense layers.

•	ReLU Activation: ReLU or Rectified Linear Unit computes the function $f(x)=max(0,x) to threshold the activation at 0.

•	SoftMax Activation: SoftMax function is applied to the output layer to convert the scores into probabilities that sum to 1.

Optimizers:

•	Adam: Adam (Adaptive moment estimation) is an update to RMSProp optimizer in which the running average of both the gradients and their magnitude is used. In practice Adam is currently recommended as the default algorithm to use, and often works slightly better than RMSProp. 

•	Categorical Cross entropy: It is a SoftMax activation plus a Cross-Entropy loss. It is used for multi-class classification.

### RESULTS

1.Detection

Even though the loss seems to be reducing over epochs, the network is not learning anything over multiple epochs, as precision & recall reduces.
 
We tried Res-net as well as U-net networks, but none seems to learn the problem. Initially we tried entire 500 x 500 resolution images, as network didn’t learn, we cropped and created 128 x 128 resolution images. We tried both RGB and grayscale images and none seems to help in learning.

Future work - Detection

For detection task, we have imbalanced classes as one 500 x 500 pixels image on an average has only 50 to 500 positive classes (center of nuclei) and remaining 249k+ classes as negative class (not nuclei). This makes the network hard to learn anything. As our next step we would like to explore patch-based models and kernel density estimation as our output layer from which we can make inference.  

2.Classification

This study involves 100 H&E stained histology images of colorectal adenocarcinomas. All images have a common size of 500 × 500 pixels. The types of nuclei that were labeled as inflammatory include lymphocyte plasma, neutrophil and eosinophil. The nuclei that do not fall into the first three categories (i.e., epithelial, inflammatory, and fibroblast) such as adipocyte, endothelium, mitotic figure, nucleus of necrotic (i.e., dead) cell, etc. are labeled as miscellaneous. In total, there are 7, 722 epithelial, 5, 712 fibroblast, 6, 971 inflammatory, and 2, 039 miscellaneous nuclei. Figuratively speaking,
 
Model summary is as follows
 
The performance metrics that we analyzed were Accuracy,AUC,F-1 score. After training the model with batch_size = 16 and epochs = 20, train accuracy has been plotted against epochs. The max accuracy achieved is 78.03%.
 
AUC is 0.7 which tells us that the model is capable of class separability
 
And F-1 score (Micro avg) is 0.25. Micro avg is preferred over macro avg for multiclass classification.

### CONCLUSION

We have developed models to detect and classify classes accordingly by feeding through the colon cancer histology images. For Detection, we have used UNet and ResNet and for classification we used Sequential Model to serve the purpose.
Future Directions

In addition to the cellularity features studied here, other features may be calculated using deep learning. Such features include tumor budding which is the presence of single tumor cells or small clusters of up to five cells in the stroma and which is associated with aggressive cancer 

We can also look at classifying colorec tal cancers according to molecular features, observing that they are related to morphological features such as the number of tumor infiltrating lymphocytes, differentiation, presence of dirty necrosis, serration, tumor budding, mucinous/not mucinous, and presence of an expanding invasive margin. Deep learning has recently been used to predict diagnostic molecular features from morphology, e.g., for lung cancer and breast cancer. It is to be expected that future work with deep learning will enable morphological, clinical and molecular data to be linked.

### REFERENCES

[3]Sharma, Harshita, et al. "A Multi-resolution Approach for Combining Visual Information using Nuclei Segmentation and Classification in Histopathological Images." VISAPP (3). 2015. 

[4]Sirinukunwattana, Korsuk, et al. "Locality sensitive deep learning for detection and classification of nuclei in routine colon cancer histology images." IEEE Trans. Med. Imaging 35.5 (2016): 1196-1206.

[5]Cruz-Roa, Angel Alfonso, et al. "A deep learning architecture for image representation, visual interpretability and automated basal-cell carcinoma cancer detection." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Berlin, Heidelberg, 2013. 


