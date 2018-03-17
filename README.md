
## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[image1]: ./example_img/camera_undistort.png "Camera Undistorted"
[image2]: ./example_img/undistort.png "Undistorted"
[image3]: ./example_img/color_gradient_threshold.png "Color Gradient Threshold"
[image4]: ./example_img/perspective_transform.png "Perspective Transform"
[image5]: ./example_img/perspective_all.png "Perspective Transform All"
[image6]: ./example_img/find_lane.png "Find Lane"
[image7]: ./example_img/histogram.png "Histogram"
[image8]: ./example_img/find_lane_from_prev_fit.png "Find Lane From Prev Fit"
[image9]: ./example_img/calculate_curvature.png "Curvature"
[image10]: ./example_img/draw_lane.png "Draw Lane"
[video1]: ./output_video/project_video.mp4 "Project Video"
[video2]: ./output_video/challenge_video.mp4 "Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd and 3rd code cell of the IPython notebook adv_lane_finding.ipynb.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0 (depth), such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I used the OpenCV functions `findChessboardCorners` and `drawChessboardCorners` to identify the locations of corners on a chessboard photos in camera_cal folder taken from different angles.

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Camera Undistorted][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Undistorted][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are present in code cell 5, 6 and 7 in `adv_lane_finding.ipynb`. I explored several combinations of sobel gradient thresholds and color channel thresholds in multiple color spaces. These are labeled clearly in the Jupyter notebook. Ultimately, I chose to use S-channel and R-channel. I have also used absolte sobel gradient in x direction along with R-channel with different threshold.

Here's an example of my output for this step. 

![Color Gradient Threshold][image3]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in code cell 9 in `adv_lane_finding.ipynb`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points with following values:



| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 705, 460      | 960, 0      |
| 203, 720     | 320, 720      |
| 1127, 720      | 960,7200        |

It uses the CV2's `getPerspectiveTransform()` and `warpPerspective()` fns. Given below is an example image after three levels of processing:
    1. Distortion correction
    2. Color and Gradient Thresold applied
    3. Perpective Transformed

![Perspective Transform][image4]

Perspective transform applied to all the test images:

![Perspective Transform All][image5]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `sliding_window_polyfit` and `find_lane_from_prev_fit`, which identify lane lines and fit a second order polynomial to both right and left lane lines, are present in the Jupyter notebook cell number 12 and 14 respectively. The first of these computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. These locations were identified from the local maxima of the left and right halves of the histogram. The function then identifies ten windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image. Pixels belonging to each lane line are identified and the Numpy polyfit() method fits a second order polynomial to each set of pixels. The image below demonstrates how this process works:

![Find Lane][image6]

The image below depicts the histogram generated by sliding_window_polyfit; the resulting base points for the left and right lanes - the two peaks nearest the center - are clearly visible:

![Histogram][image7]

The polyfit_using_prev_fit function performs basically the same task, but alleviates much difficulty of the search process by leveraging a previous fit (from a previous video frame, for example) and only searching for lane pixels within a certain range of that fit. The image below demonstrates this - the green shaded area is the range from the previous fit, and the yellow lines and red and blue pixels are from the current image:

![Find Lane From Prev Fit][image8]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in the function `cal_curvature_and_distance_from_center`. The API can calculate curvature in both pixel numbers (in debug mode) and in terms of meters.

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

In this example, left_fit_cr[0] is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and left_fit_cr[1] is the second (y) coefficient. y_eval is the y position within the image upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). ym_per_pix is the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:

        lane_center = (right_x_0 + left_x_0) / 2
        image_center = img.shape[1]/2
        center_dist =  image_center * xm_per_pix - lane_center
        
right_x_0 and left_x_0 are the x-intercepts of the right and left fits, respectively. This requires evaluating the fit at the maximum y value (the bottom of the image) because the minimum y value is actually at the top (otherwise, the constant coefficient of each fit would have sufficed). The car position is the difference between these intercept points and the image midpoint (assuming that the camera is mounted at the center of the vehicle).

Here is the output of the function in debug mode for the following image:

 `-2181.99565858 -4471.3728884`<br>
 `275.018453063 894.012590292 618.994137229`<br>
 `380.62400496 1018.56276983 637.938764873`<br> 
 `719 2.27146583605 6.07851975545 4.17499279575 640.0 -0.355637957043`<br>
 `left_lane_curvature= 634.476595767 m, right_lane_curvature= 1284.85228905 m`<br>
 `Distance from lane center for example: -0.355637957043 m` 

![Curvature][image9]


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the code cells 17 and 18 using function `draw_lane` and `wriite_data`. A polygon is generated based on plots of the left and right fits, warped back to the perspective of the original image using the inverse perspective matrix Minv and overlaid onto the original image. The image below is an example of the results of the draw_lane function:

![draw_lane][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here are the links to my video results:

[Project Video](./output_video/project_video.mp4) <br>
[Challenge Video](./output_video/challenge_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline developed this project did a real good job in detecting the lane lines for the project_video.mp4 video, 
which implied that the code works well for the known ideal conditions having distinct lane lines, and with not much shadows. 
It also performed preety well for challenge_video.mp4 video. It didn't lose the lane even when heavy shadow was there.

But, code failed miserably on the harder_challenge_video.mp4. Here the code got exposed got steep curves, and shadows and 
failed to follow the lanes.

In order to make it more robust, I guess I need to go back and revisit the binary channels selection and see if there is any other combination that can help and work fine in shadows and steep curves. Further reading on this topic may help in making
the code robust for harder challenge video.
