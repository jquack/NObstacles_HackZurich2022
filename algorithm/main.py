import cv2
import numpy as np
import argparse
from scipy.special import comb

# args setting
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', "--input", help="input file video")
args = parser.parse_args()


def main():
    # load video class
    cap = VideoCapture(args.input)
    out = cv2.VideoWriter('output/output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (1280,720))

    while True:
        frame = cap.read()

        # Algorithm is stable when reading in the same image consequently.
        # With the produced video file, there are still some hickups.
        # frame = cv2.imread('video/track_test_image_resized.jpg')
        if frame is None:
            break

        valid_frame = frame.copy()

        # gray scale transform
        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)
        hsv_frame = cv2.cvtColor(valid_frame,cv2.COLOR_BGR2HSV)

        # histogram equalization image
        histeqaul_frame = cv2.equalizeHist(gray_frame)

        # We subtract the s channel of the HSV colored image to remove
        # shadows from the image.
        s = hsv_frame[:, :, 1]
        histeqaul_frame_s = cv2.subtract(histeqaul_frame, s)

        # Perform edge detection with the Canny Algorithm
        height, width = histeqaul_frame_s.shape
        edges = cv2.Canny(histeqaul_frame_s, 59, 3*59, 3)

        line_image = valid_frame.copy()
        lines = cv2.HoughLinesP(image=edges,
                                rho=0.65,
                                theta=np.pi / 113,
                                threshold=17,
                                lines=np.array([]),
                                minLineLength=143,
                                maxLineGap=4)

        if lines is not None:
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

        out.write(line_image)

        cv2.imshow('Video', line_image)
        cv2.waitKey(1)

    print('finish')

    out.release()
    cv2.destroyAllWindows()


# class for reading video
class VideoCapture:
    def __init__(self, path):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(path)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def read(self):
        # Grab a single frame of video
        ret, frame = self.video.read()
        return frame


if __name__ == '__main__':
    main()
