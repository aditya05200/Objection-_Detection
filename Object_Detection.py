import cv2

# opencv Dnn
net = cv2.dnn.readNet('dnn_model/yolov4-tiny.weights', 'dnn_model/yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

# load class
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# create video capture object
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: could not open video capture device")
    exit()

# set video capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# create window
cv2.namedWindow("Project")

# mouse click event handler
def click_button(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

# register mouse click event handler
cv2.setMouseCallback("Project", click_button)

# main loop
while True:
    # get frame
    ret, frame = cap.read()
    if not ret:
        print("Error: could not read frame from video capture device")
        break

    # object detection
    (class_ids, score, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, score, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 250), 3)

    cv2.imshow("Project", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release video capture object and close window
cap.release()
cv2.destroyAllWindows()
