# WSIRegistration

This repository contains the c++ code for the paper:
*"Regional Registration of Whole Slide Image Stacks Containing Highly Deformed Artefacts"* 
submitted to Bioinformatics Journal. Opencv-3.4.1 and Opencv_contrib3.4.1 libraries were used for this implementation. 

# Motivation:
High resolution 2D whole slide imaging provides rich information about the tissue structure. Moreover, this information can be a lot richer if these 2D images can be stacked into a 3D tissue volume. A 3D analysis, however, requires accurate reconstruction of the tissue volume from the 2D image stack. This task is not trivial due to the distortions that each individual tissue slice experiences while cutting and mounting the tissue on the glass slide. Performing registration for the whole tissue slices may be adversely affected by the deformed tissue regions. Consequently, regional registration is found to be more effective. 

In this repository, we have included our implementation of the proposed regional registration algorithm for whole slide images which incrementally focuses registration on the area around the region of interest. 

<<<<<<< HEAD

=======
>>>>>>> f147fe4874693ec56a30050cc538528ac32f3edf
