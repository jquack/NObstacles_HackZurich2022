import cv2
import numpy as np

# Image Path selection for testing
path = 'video/track_test_image_resized.jpg'
# path = 'video/20220712_075411_resized_cropped.jpg'

# TrackBars for finding the right parameters
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("rho","TrackBars",65,200,empty)
cv2.createTrackbar("theta","TrackBars",113,360,empty)
cv2.createTrackbar("threshold","TrackBars",20,50,empty)
cv2.createTrackbar("maxLineGap","TrackBars",10,20,empty)
cv2.createTrackbar("minLineLength","TrackBars",20,300,empty)

cv2.createTrackbar("threshold_canny","TrackBars",59,1000,empty)
cv2.createTrackbar("kernel_size","TrackBars",3,20,empty)

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

    cv2.imwrite("output/h.jpg", h)
    cv2.imwrite("output/s.jpg", s)
    cv2.imwrite("output/v.jpg", v)

    histeqaul_frame = cv2.equalizeHist(imgGray)

    # We subtract the s channel of the HSV colored image to remove
    # shadows from the image. 
    histeqaul_frame_s = cv2.subtract(histeqaul_frame, s)
    
    cv2.imwrite("output/hist.jpg", histeqaul_frame_s)

    height, width = histeqaul_frame_s.shape

    edges = cv2.Canny(histeqaul_frame_s, threshold_canny, 3*threshold_canny, kernel_size)

    cv2.imshow("canny", edges)
    cv2.imwrite("output/canny.jpg", edges)

    # We apply a mask to limit the influence of the many small edges of
    # the trees and other objects aside the tracks.

    # Triangle Shape V1
    # triangle = np.array([
    #                    [
    #                     (int(width/2), int(height/2 - height*0.1)),
    #                     (int(width/5), height),
    #                     (int(width/5)*4, height)
    #                    ]
    #                    ])
    # Triangle Shape V2
    # triangle = np.array([
    #                    [
    #                     (int(width/2), 0),
    #                     (0, height),
    #                     (width, height)
    #                    ]
    #                    ])

    # mask = np.zeros_like(histeqaul_frame_s)
    # mask = cv2.fillPoly(mask, triangle, 255)

    # cv2.imwrite("output/mask.jpg", mask)
    # masked_edges = cv2.bitwise_and(edges, mask)

    # cv2.imwrite("output/masked_canny.jpg", masked_edges)

    # Draw the detected lines of the rail tracks on top of
    # the original image
    line_image = img.copy()
    lines = cv2.HoughLinesP(image=edges,
                            rho=0.65,
                            theta=np.pi / 113,
                            threshold=17,
                            lines=np.array([]),
                            minLineLength=143,
                            maxLineGap=4)

    # Use the line below to control the parameters by the trackbars.
    # lines = cv2.HoughLinesP(image=masked_edges,
    #                         rho=rho/100,
    #                         theta=np.pi / theta,
    #                         threshold=threshold,
    #                         lines=np.array([]),
    #                         minLineLength=minLineLength,
    #                         maxLineGap=maxLineGap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            dx = x2-x1
            dy = y2-y1

            if y1 < y2:
                final1Y = min(y1-dy, int(height*0.4))
                final1X = x1+int(dx*(final1Y-y1)/dy)
                final2Y = height
                final2X = x2+int(dx*(final2Y-y2)/dy)

                cv2.line(line_image,(final1X,final1Y),(final2X,final2Y),(0,255,255),5)
            else:
                final2Y = max(y2+dy, int(height*0.4))
                final2X = x2+int(dx*(final2Y-y2)/dy)
                final1Y = height
                final1X = x1+int(dx*(final1Y-y1)/dy)

                cv2.line(line_image,(final1X,final1Y),(final2X,final2Y),(0,255,255),5)

    cv2.imshow('line_image', line_image)
    cv2.imwrite("output/line_image.jpg", line_image)
    cv2.imwrite("output/original.jpg", img)

    cv2.waitKey(1)
