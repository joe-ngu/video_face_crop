import cv2
import numpy as np
import argparse
import pathlib

ESCAPE_KEY = 27


def useVideo(video_file):
    # attempt to read video file
    video_name = video_file.split("/")[-1].split(".")[0]

    source = cv2.VideoCapture(video_file)
    if not source.isOpened():
        print("Couldn't open video file")
    return video_name, source


def getRectangleArea(x1, y1, x2, y2):
    return abs(x1 - x2) * abs(y1 - y2)


def caffeFaceDetection(video_name="camera", video=None):
    # default to camera if not using video file
    source = video if video else cv2.VideoCapture(0)
    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)

    # config for Caffe model face detection
    model_file = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_file = "./models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    video_width = 300
    video_height = 300
    mean = (104, 117, 123)
    confidence_threshold = 0.8
    # rect_color = (0, 255, 0)

    roi = None
    while cv2.waitKey(1) != ESCAPE_KEY:
        # read frames from video
        has_frame, frame = source.read()
        if not has_frame:
            break  # exit if unable to read anymore frames
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # detecting faces
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (video_width, video_height), mean, swapRB=False, crop=False
        )
        net.setInput(blob)
        detected = net.forward()

        # drawing rectangles around detected objects that meets confidence threshold
        maxRectArea = 0
        for i in range(detected.shape[2]):
            confidence = detected[0, 0, i, 2]
            if confidence > confidence_threshold:
                rect = detected[0, 0, i, 3:7] * np.array(
                    [frame_width, frame_height, frame_width, frame_height]
                )
                (x_start, y_start, x_end, y_end) = rect.astype("int")
                x_start = max(0, x_start - 50)
                x_end = min(x_end + 50, frame_width)
                y_start = max(0, y_start - 50)
                y_end = min(y_end + 50, frame_height)

                curRectArea = getRectangleArea(x_start, y_start, x_end, y_end)
                # cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), rect_color)
                if curRectArea > maxRectArea:
                    maxRectArea = curRectArea
                    roi = frame[y_start:y_end, x_start:x_end]

        # displaying video
        cv2.imshow(video_name, frame)
        cv2.imshow("roi", roi)

    print("Exiting...")
    source.release()
    cv2.destroyAllWindows()


def cascadeFaceDetection(video_name="camera", video=None):
    # default to camera if not using video file
    source = video if video else cv2.VideoCapture(0)
    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)

    # configure haarcascade model (comes packaged wit open cv)
    cascade_path = (
        pathlib.Path(cv2.__file__).parent.absolute()
        / "data/haarcascade_frontalface_default.xml"
    )
    classifier = cv2.CascadeClassifier(str(cascade_path))

    while cv2.waitKey(1) != ESCAPE_KEY:
        has_frame, frame = source.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = classifier.detectMultiScale(
            gray,
            scaleFactor=1.0,
            minNeighbors=5,  # higher the number the stricter the detection threshold
            minSize=(30, 30),
            flags=cv,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CropFace", description="Crops a face in video"
    )
    parser.add_argument("-v", "--video", type=str)
    parser.add_argument("-c", "--caffe", action="store_true")
    args = parser.parse_args()

    if args.video:
        video_file = f"./videos/{args.video}"
        video_name, video_source = useVideo(video_file)
        if args.caffe:
            caffeFaceDetection(video_name, video_source)  # caffe
        elif args.haar:
            cascadeFaceDetection(video_name, video_source)  # haarcascade
