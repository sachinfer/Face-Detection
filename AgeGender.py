import cv2 as cv
import time
import argparse

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


# Parse command line arguments
parser = argparse.ArgumentParser(description='Age and Gender Recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Leave empty to use webcam.', default='')
parser.add_argument("--device", default="cpu", help="Device to run inference on: 'cpu' or 'gpu'")

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load networks
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Set backend and target for CPU or GPU
if args.device == "cpu":
    ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    faceNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# Initialize camera or video
cap = cv.VideoCapture(args.input if args.input else 0)  # Use webcam if no input provided
if not cap.isOpened():
    print("Error: Could not open camera or video file.")
    exit()

padding = 20
while True:
    # Read frame from webcam or video
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("No frame captured, exiting.")
        break

    # Detect faces in the frame
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face detected, checking next frame.")
        continue

    # Loop through the faces detected
    for bbox in bboxes:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        # Prepare the face for gender prediction
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f"Gender: {gender}, Confidence: {genderPreds[0].max():.3f}")

        # Prepare the face for age prediction
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f"Age: {age}, Confidence: {agePreds[0].max():.3f}")

        # Display label on the frame
        label = f"{gender}, {age}"
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

    # Show the output frame
    cv.imshow("Age and Gender Detection", frameFace)
    print(f"Time per frame: {time.time() - t:.3f} seconds")

    # Exit condition: Press 'q' to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
