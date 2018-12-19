import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def calibrate_camera():
    nx = 9 #number of inside corners in x
    ny = 6 #number of inside corners in y
    
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    images = glob.glob('camera_cal/*.jpg')
    for image in images:
        img  = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def color_gradient_transform(img):
    # Convert image to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1 

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def perspective_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, Minv
   
def find_lines(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((binary_warped, binary_warped, binary_warped))*255)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_r = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_r = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, left_fit, right_fit, left_fit_r, right_fit_r

def draw_path(orig_image, warped, left_fit, right_fit, M):
    new_image = np.copy(orig_image)

    blank_image = np.zeros_like(warped).astype(np.uint8)
    rgb_image = np.dstack((blank_image, blank_image, blank_image))

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(rgb_image, np.int_([pts]), (0, 255, 0))

    unwarp = cv2.warpPerspective(rgb_image, M, (warped.shape[1], warped.shape[0]))

    combined = cv2.addWeighted(new_image, 1, unwarp, 0.5, 0)

    return combined

def cal_radius(lfit, rfit):
    y = 720
    ym_per_pix = 30/720
    
    lcur = ((1 + (2*lfit[0]*y*ym_per_pix + lfit[1])**2)**1.5) / np.absolute(2*lfit[0])

    rcur = ((1 + (2*rfit[0]*y*ym_per_pix + rfit[1])**2)**1.5) / np.absolute(2*rfit[0])

    rad = (lcur + rcur) / 2
    return rad


def center_distance(img, left_fit, right_fit):
    h = img.shape[0]
    car_center = img.shape[1]/2


    left_side = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
    right_side = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
    lane_center = (left_side + right_side)/ 2
    xm_per_pix = 3.7/(right_side - left_side)
    
    dist = (car_center - lane_center) * xm_per_pix
    return dist

def draw_info(img, left_fit, right_fit, lfit, rfit):
    info = ''
    dist = center_distance(img, left_fit, right_fit)
    if dist < 0:
        info = 'left'
    if dist > 0:
        info = 'right'

    dist = abs(dist)
    rad = cal_radius(lfit, rfit)
    line1 = 'Radius of Curvature = ' + '{:04.2f}'.format(rad) + 'm'
    line2 = 'Vehicle is ' + '{:04.2f}'.format(dist) + 'm ' + info + ' of center'
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, line1, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    cv2.putText(img, line2, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)


def process_image(image):
    #mtx, dist = calibrate_camera()
    undist =  undistort(image, mtx, dist)
    grad_image = color_gradient_transform(image)
    img_size = (image.shape[1], image.shape[0])
    corners = [[190, 720], [585, 455], [705, 455], [1130, 720]]
    offset = 200
    x = img_size[0]
    y = img_size[1]
    src = np.float32([corners[1], corners[2], corners[3], corners[0]])
    dst = np.float32([[offset, 0], [x-offset, 0], [x-offset, y], [offset, y]])
    warped, M = perspective_transform(grad_image, src, dst)
    _, left_fit, right_fit, left_fit_r, right_fit_r = find_lines(warped)

    comb = draw_path(image, warped, left_fit, right_fit, M)
    draw_info(comb, left_fit, right_fit, left_fit_r, right_fit_r)

    return comb


mtx, dist = calibrate_camera()
chessboard_image = cv2.imread('camera_cal/calibration1.jpg')
undistorted_image = undistort(chessboard_image, mtx, dist)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(chessboard_image)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(undistorted_image)
ax2.set_title('Distortion Corrected Image', fontsize=15)
plt.show(fig1)

image = cv2.imread('test_images/straight_lines1.jpg') 
comb = process_image(image)
undist =  undistort(image, mtx, dist)
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(undist)
ax2.set_title('Distortion Corrected Image', fontsize=15)
plt.show(fig2)
grad_image = color_gradient_transform(image)
fig3 = plt.imshow(grad_image, cmap='gray')
plt.show(fig3)
img_size = (image.shape[1], image.shape[0])
corners = [[190, 720], [585, 455], [705, 455], [1130, 720]]
offset = 200
x = img_size[0]
y = img_size[1]
src = np.float32([corners[1], corners[2], corners[3], corners[0]])
dst = np.float32([[offset, 0], [x-offset, 0], [x-offset, y], [offset, y]])
warped, M = perspective_transform(image, src, dst)
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(undist)
ax1.set_title('Distortion Corrected Image', fontsize=15)
ax2.imshow(warped)
ax2.set_title('Warped Image', fontsize=15)
plt.show(fig4)
warped, M = perspective_transform(grad_image, src, dst)
out_img, left_fit, right_fit, left_fit_r, right_fit_r = find_lines(warped)
ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
fig5 = plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.show(fig5)


comb = draw_path(image, warped, left_fit, right_fit, M)
draw_info(comb, left_fit, right_fit, left_fit_r, right_fit_r)

fig6 = plt.imshow(comb)
plt.show(fig6)

video_output1 = 'project_video_output.mp4'
video_input1 = VideoFileClip('project_video.mp4')#.subclip(22,26)
processed_video = video_input1.fl_image(process_image)
processed_video.write_videofile(video_output1, audio=False)

