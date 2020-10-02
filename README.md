# WSIRegistration

This repository contains the c++ code for the paper:
*"Regional Registration of Whole Slide Image Stacks Containing Highly Deformed Artefacts"* 
submitted to BMC Bioinformatics Journal. Opencv-3.4.1 and Opencv_contrib3.4.1 libraries were used for this implementation. 
The gray-levels file is the executable file for the Mumford-Shah code which we call using system call from the code.

# Motivation:
High resolution 2D whole slide imaging provides rich information about the tissue structure. This information can be a lot richer if these 2D images can be stacked into a 3D tissue volume. A 3D analysis, however, requires accurate reconstruction of the tissue volume from the 2D image stack. This task is not trivial due to the distortions that each individual tissue slice experiences while cutting and mounting the tissue on the glass slide. Performing registration for the whole tissue slices may be adversely affected by the deformed tissue regions. Consequently, regional registration is found to be more effective. 

In this repository, we have included our implementation of the proposed regional registration algorithm for whole slide images which incrementally focuses registration on the area around the region of interest. We provide a brief description of different stages of the proposed algorithm together with an example.

In order to register a target blood vessel in the whole slide images, three steps are carried out as follows: 1) Preprocessing, to remove extra stains and artifacts around the tissue of interest, 2) Whole tissue registration, to approximately align the whole tissue in consecutive whole slide images, 3) Target blood vessel registration, to register the blood vessel of interest. Finally, fine registration is carried out to improve the registration for the blood vessel of interest.

# 1) Preprocessing
Extra stains and artifacts around the tissue can affect the registration outcome. To remove these artifacts, each image is converted to the gray scale and smoothed using a Gaussian filter. The smoothed image is then thresholded. Since an accurate segmentation of the tissue from the surrounding artifacts cannot be achieved merely by thresholding, an opening and later an closing morphological operation was applied on the output mask from thresholdingto get a mask that covers the artifacts and extra stains around the tissue. The final segmentation mask is then applied to the image to remove the surrounding artifacts. Contours in the new image are then detected. The contours which are closer to the center of the image and surround the largest area in the image are identified. Extra tissue and stains outside the convex hull of the selected contours are removed, resulting in a cleaned tissue image. 

Original Image             |  Thresholded Image        | Selected Edges            | Convex Hull of Edges      | Cleaned Image
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="original.jpg" width="150"> |  <img src="thresholded.jpg" width="150">|  <img src="overlay.jpg" width="150">|  <img src="overlay_hull.jpg" width="150">|  <img src="clean.jpg" width="150">

# 2) Whole tissue registration
In this stage, relative rotations or displacements in the location of the tissue across consecutive virtual slides are corrected. The cleaned image, is segmented using a multi-resolution Monte Carlo method of Sashida. Next, each consecutive pair of Mumford-Shah segmented images are registered independently. For each pair of images, a combination of varying translation $(dx,dy)$ and rotation ($\theta$) transformations are applied to the second (moving) image to find the rotation and translation parameters which make $T_{\theta,dx,dy}[\tilde{I}_{i+1}^{MS}(x,y)]$ most similar to $\tilde{I}_i^{MS}(x,y)$.
The $\{\theta, dx, dy\}$ triplet which gives the least sum of squared difference is chosen and its corresponding transformation matrix is applied to the moving image: ${I'}_{i+1} \leftarrow  T_{\theta,dx,dy}[\tilde{I}_{i+1}]$. These two steps roughly align the images in consecutive image slides. 

