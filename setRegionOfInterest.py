import cv2
import os
import selectinwindow
import sys

import sys
# Set recursion limit
sys.setrecursionlimit(10 ** 9)

#Image Folder Path
path_folder = "/home/images/"
image_folder = ['Case 1'
save_directory = '/home/images'
#Width of slideshow
window_width_downsample = 1.0/pow(2,0)
#Height of slideshow
window_height_downsample = 1.0/pow(2,0)
#Transition time slideshow
slideshow_trasnition_time = 2
#Image stable time
slideshow_img_time = 10
#Window Name
window_name="Image Slide Show"
#Supoorted formats tuple
supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.dib', '.jpe', '.jp2', '.pgm', '.tiff', '.tif', '.ppm')
#Escape ASCII Keycode
esc_keycode=27
#slide calibration
transit_slides = 10
#minimum weight
min_weight = 0
#maximum weight 
max_weight = 1

if not os.path.exists(save_directory):
    os.mkdir(save_directory)

#Range function with float 
def range_step(start, step, stop):
	range = start
	while range < stop:
		yield range
		range += step

#Wait Key function with escape handling		
def wait_key(time_seconds):
	#state False if no Esc key is pressed
	state = False
	#Check if any key is pressed. second multiplier for millisecond: 1000
	k = cv2.waitKey(int(time_seconds * 1000))
	#Check if ESC key is pressed. ASCII Keycode of ESC=27
	if k == esc_keycode:  
		#Destroy Window
		cv2.destroyWindow(window_name)
		#state True if Esc key is pressed
		state = True
	#return state	
	return state	
	
#Load image path of all images		
def load_img_path(pathFolder):
	#empty list
	_path_image_list = []
	#Loop for every file in folder path
	for filename in os.listdir(pathFolder):
		#Image Read Path
		_path_image_read = os.path.join(pathFolder, filename)
		#Check if file path has supported image format and then only append to list
		if _path_image_read.lower().endswith(supported_formats):
			_path_image_list.append(_path_image_read)
	#Return image path list
	return sorted(_path_image_list)

def load_img_names(pathFolder):
	#empty list
	_image_list = []
	#Loop for every file in folder path
	for filename in os.listdir(pathFolder):
		#Image Read Path
		_image_name = filename
		#Check if file path has supported image format and then only append to list
		if _image_name.lower().endswith(supported_formats):
			_image_list.append(_image_name)
	#Return image path list
	return sorted(_image_list)

#Load image and return with resize	
def load_img(pathImageRead, resizeWidthRatio, resizeHeightRatio): 	
	#Load an image
	#cv2.IMREAD_COLOR = Default flag for imread. Loads color image.
	#cv2.IMREAD_GRAYSCALE = Loads image as grayscale.
	#cv2.IMREAD_UNCHANGED = Loads image which have alpha channels.
	#cv2.IMREAD_ANYCOLOR = Loads image in any possible format
	#cv2.IMREAD_ANYDEPTH = Loads image in 16-bit/32-bit otherwise converts it to 8-bit
	_img_input = cv2.imread(pathImageRead,cv2.IMREAD_UNCHANGED)
	#Check if image is not empty
	if _img_input is not None:
	       #Get read images height and width
		_img_height, _img_width = _img_input.shape[:2]
		_img_resized =_img_input
		#if image size is more than resize perform cv2.INTER_AREA interpolation otherwise cv2.INTER_LINEAR for zooming
		if resizeWidthRatio < 1 or resizeHeightRatio < 1:
			interpolation = cv2.INTER_AREA
		else:
			interpolation = cv2.INTER_LINEAR
		
		# perform the actual resizing of the image and show it
		_img_resized = cv2.resize(_img_input, tuple([int(_img_width * resizeWidthRatio), int(_img_height * resizeHeightRatio)]), interpolation)
		
	#return the resized image	
	return _img_resized

def getDirName(fullPath):
    parts = fullPath.split('/')
    slide = parts[-1]
    nameExt = slide.split('.')
    name = nameExt[0]
    Dir = fullPath.replace(slide,'')
    print(Dir)
    return Dir, name


imge_path = sys.argv[1]
rectI = selectinwindow.dragRect
rect_x = rect_y =  rect_w = rect_h = 0
#Load first image
img_one = load_img(imge_path, window_width_downsample, window_height_downsample)
# Define the drag object
rectI.returnflag = False
selectinwindow.init(rectI, img_one, window_name, img_one.shape[1], img_one.shape[0], rect_x, rect_y, rect_w, rect_h)
print(imge_path)
cv2.namedWindow(rectI.wname,cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(rectI.wname, selectinwindow.dragrect, rectI)
cv2.imshow(window_name, rectI.image)
while True:
   # display the image
   key = cv2.waitKey(1) & 0xFF

   # if returnflag is True, break from the loop
   if rectI.returnflag == True:
       break

   if wait_key(slideshow_img_time):
      del img_path_list[:]
      sys.exit(0)

print ("Dragged rectangle coordinates")
print (str(rectI.outRect.x) + ',' + str(rectI.outRect.y) + ',' + \
str(rectI.outRect.w) + ',' + str(rectI.outRect.h))
rect_x = rectI.outRect.x
rect_y = rectI.outRect.y
rect_h = rectI.outRect.h
rect_w = rectI.outRect.w
img_org = cv2.imread(imge_path)

if(rect_h==0 or rect_w ==0):
	_img_height, _img_width = _img_one.shape[:2]
	center_x = _img_height/(2*window_height_downsample)
	center_y = _img_width/(2*window_width_downsample)
	boxSize = 128
	roi = [center_x, center_y, boxSize, boxSize]
else: 
	roi = [int((rect_y+(rect_h/2))/window_height_downsample), int(rect_x+(rect_w/2)/window_width_downsample),
		  int(rect_h/window_height_downsample), int(rect_w/window_width_downsample)]

Dir, name = getDirName(imge_path)
file = open(os.path.join(Dir,'roi_location.txt'),'w') 
file.write("===================================\n")
file.write("COMPUTER GENERATED DO NOT MODIFY\n")
file.write("===================================\n")
file.write("%d %d\n%d %d" % (roi[0], roi[1], roi[2], roi[3]))
file.close()



        

