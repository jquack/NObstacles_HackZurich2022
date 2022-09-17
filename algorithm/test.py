import cv2
import numpy as np

def empty(a):
    pass


# path = 'video/20220712_075411.jpg'
# path = 'video/gray_frame_14750.png'
path = 'video/track_test_image_resized.jpg'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("rho","TrackBars",100,200,empty)
cv2.createTrackbar("theta","TrackBars",180,360,empty)
cv2.createTrackbar("threshold","TrackBars",20,50,empty)
cv2.createTrackbar("maxLineGap","TrackBars",10,20,empty)
cv2.createTrackbar("minLineLength","TrackBars",20,300,empty)

cv2.createTrackbar("threshold_canny","TrackBars",35,1000,empty)
cv2.createTrackbar("kernel_size","TrackBars",3,20,empty)

# cv2.createTrackbar("High Threshold","TrackBars",66,1000,empty)
# cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
# cv2.createTrackbar("Val Min","TrackBars",21,255,empty)
# cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.imwrite("output/imgGray.jpg", imgGray)

    rho = cv2.getTrackbarPos("rho","TrackBars")
    theta = cv2.getTrackbarPos("theta","TrackBars")
    threshold = cv2.getTrackbarPos("threshold","TrackBars")
    maxLineGap = cv2.getTrackbarPos("maxLineGap","TrackBars")
    minLineLength = cv2.getTrackbarPos("minLineLength","TrackBars")

    threshold_canny = cv2.getTrackbarPos("threshold_canny","TrackBars")
    kernel_size = cv2.getTrackbarPos("kernel_size","TrackBars")

    imgGray_cropped = imgGray[int(imgGray.shape[0]/3):, :]

    cv2.imwrite("output/imgGray_cropped.jpg", imgGray_cropped)

    h, s, v = imgHSV[:, :, 0], imgHSV[:, :, 1], imgHSV[:, :, 2]

    # cv2.imshow("h", h)
    # cv2.imshow("s", s)
    # cv2.imshow("v", v)
    cv2.imwrite("output/h.jpg", h)
    cv2.imwrite("output/s.jpg", s)
    cv2.imwrite("output/v.jpg", v)

    # h_thres = cv2.getTrackbarPos("High Threshold", "TrackBars")
    # s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    # s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    # v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    # v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    
    cv2.imshow("gray", imgGray)
    cv2.imshow("gray_cropped", imgGray_cropped)

    histeqaul_frame = cv2.equalizeHist(imgGray_cropped)
    img_gaussian = cv2.GaussianBlur(histeqaul_frame,(3,3),0)

    cv2.imshow("hist", histeqaul_frame)
    
    cv2.imwrite("output/hist.jpg", histeqaul_frame)
    cv2.imshow("gaussian", img_gaussian)
    cv2.imwrite("output/gaussian.jpg", img_gaussian)

    #prewitt
    # kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    # kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    # # img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    # img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    edges = cv2.Canny(img_gaussian, threshold_canny, 3*threshold_canny, kernel_size)

    # cv2.imshow("Prewittx", img_prewittx)
    cv2.imshow("canny", edges)
    
    cv2.imwrite("output/canny.jpg", edges)
    # cv2.imshow("Prewitt", img_prewittx + img_prewitty)

    # line_image = np.copy(edges) * 0
    line_image = edges.copy()

    # minLineLength = 20 #img.shape[1] - 300
    lines = cv2.HoughLinesP(image=edges,
                            rho=0.65,
                            theta=np.pi / 113,
                            threshold=17,
                            lines=np.array([]),
                            minLineLength=143,
                            maxLineGap=4)

    # lines = cv2.HoughLinesP(image=edges,
    #                         rho=rho/100,
    #                         theta=np.pi / theta,
    #                         threshold=threshold,
    #                         lines=np.array([]),
    #                         minLineLength=minLineLength,
    #                         maxLineGap=maxLineGap)
    a, b, c = lines.shape

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    # for i in range(a):
    #     cv2.line(line_image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2],  lines[i][0][3]), (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('line_image', line_image)
    
    cv2.imwrite("output/line_image.jpg", line_image)

    # low_threshold = 50
    # high_threshold = 150
    # edges = cv2.Canny(img_gaussian, thres, 3*thres, kernel_size)

    # rho = 5  # distance resolution in pixels of the Hough grid
    # theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 25  # minimum number of votes (intersections in Hough grid cell)
    # min_line_length = 300  # minimum number of pixels making up a line
    # max_line_gap = 20  # maximum gap in pixels between connectable line segments
    # line_image = np.copy(edges) * 0  # creating a blank to draw lines on

    # # Run Hough on edge detected image
    # # Output "lines" is an array containing endpoints of detected line segments
    # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
    #                     min_line_length, max_line_gap)

    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    # lines_edges = cv2.addWeighted(edges, 0.8, line_image, 1, 0)

    # cv2.imshow('edges', edges)

    # print(h_min,h_max,s_min,s_max,v_min,v_max)
    # lower = np.array([h_min,s_min,v_min])
    # upper = np.array([h_max,s_max,v_max])
    # mask = cv2.inRange(imgHSV,lower,upper)
    # imgResult = cv2.bitwise_and(img,img,mask=mask)

    # cv2.imshow('edge', edges)
    # cv2.imshow("Mask Images", mask)
    cv2.imshow("Original Images", img)
    
    cv2.imwrite("output/original.jpg", img)

    cv2.waitKey(1)