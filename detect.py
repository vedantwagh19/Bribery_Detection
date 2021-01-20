import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import copy
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

class data():
    def __init__(self,bbox,box_id):
        self.bbox = bbox
        self.id = box_id


def box_centroid(bbox):
    centroid_x = (bbox[0] + bbox[2])/2
    centroid_y = (bbox[1] + bbox[3])/2
    return centroid_x,centroid_y

def rectangle_coordinates(bbox):
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    return x1,y1,x2,y2

def find_distance(a,b):
    return np.linalg.norm(a-b)

def inside_rectangle(cx,cy,bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    if(cx > x1 and cx < x2 and cy >y1 and cy < y2 ):
        return True
    return False


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    bag_owner=[-1]*1000
    bag_freq=np.zeros(10000)
    bag_flag=np.zeros(1000)
    flag_freq=np.zeros(1000)
    result_path="/content/drive/MyDrive/Detection_Results"

    # while video is running
    print('Video is being Processed......')
    print("")
    while True:
        return_value, frame = vid.read()
        copy_frame=copy.deepcopy(frame)
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Done Processing The Video........')
            print("")
            break
        frame_num +=1
        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # detecting people,bags,envelopes,cash
        allowed_classes = ['person','bagpack','handbag','suitcase','envelopes','cash']


        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        for i in range(0,len(names)):
            if names[i]=='suitcase' or names[i]=='handbag' or names[i]=='bagpack':
                names[i]='bag'

        #deleting redundant detections
        bboxes = np.delete(bboxes, deleted_indx, axis=0)

        # Storing coordinates of person and bags in different array

        bboxes_bag =[]
        no_of_bags = 0
        deleted_indx_1 = []

        for i in range(0,len(names)):
            if(names[i] == 'bag'):
                bboxes_bag.append(bboxes[i])
                no_of_bags = no_of_bags +1
                deleted_indx_1.append(i)
        #print(deleted_indx_1)
        bboxes_bag = np.asarray(bboxes_bag)

        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            #print("Objects being tracked: {}".format(count))

        # delete detections that are not in allowed_classes
        scores = np.delete(scores, deleted_indx, axis=0)
        for i in range(0,len(deleted_indx_1)-1):
            cx,cy = box_centroid(bboxes_bag[i])
            a = np.array((cx,cy))
            for j in range(i+1,len(deleted_indx_1)):
                cx1,cy1 = box_centroid(bboxes_bag[j])
                b = np.array((cx1,cy1))
                c = find_distance(a,b)
                if(c<10):
                    bboxes = bboxes = np.delete(bboxes, deleted_indx_1[j], axis=0)
                    scores = np.delete(scores, deleted_indx_1[j], axis=0)


        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        person_bbox =[]
        bag_bbox =[]
        # total_bbox = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()


            if(class_name=='person'):
                person_bbox.append(data(bbox,track.track_id))
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
            else:
                bag_bbox.append(data(bbox,track.track_id))
                if(bag_flag[track.track_id]==1 and flag_freq[track.track_id]>0):
                    color = (255,0,0)
                else:
                    color = (0,255,0)


            # draw bbox on screen

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
        #if FLAGS.info:
        #print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # Find the bag owner by checking the person bbox in which its centroid lies
        for i in range(0,len(bag_bbox)):
            cx,cy = box_centroid(bag_bbox[i].bbox)
            for j in range(0,len(person_bbox)):

                person_id=int(person_bbox[j].id)
                y = inside_rectangle(cx,cy,person_bbox[j].bbox)

                if y :
                    new_id=int(str(bag_bbox[i].id)+str(person_bbox[j].id))
                    folder_name=str(person_id)

                    if (bag_freq[new_id] == 0):
                        path = os.path.join(result_path, folder_name)
                        if (os.path.isdir(path) != True):
                            os.makedirs(path)

                    path = os.path.join(result_path, folder_name)
                    image_name = "/" + str(folder_name) +"_"+str(int(bag_freq[new_id]))+ ".jpg"
                    full_path = path + image_name
                    #print(full_path)
                    x1,y1,x2,y2 = rectangle_coordinates(person_bbox[j].bbox)
                    new_image=copy_frame[y1:y2, x1:x2]
                    if (os.path.isdir(full_path) != True):         #saving bounding boxes in repective folders
                        cv2.imwrite(full_path, new_image)

                    bag_freq[new_id]+=1
                    threshold=30
                    if(bag_freq[new_id]> threshold ):  #threshold could be changed depending on video length for short videos it should be around 30 for long videos > 1min it should be 120
                        if(bag_owner[bag_bbox[i].id]!=-1 and bag_owner[bag_bbox[i].id]!=person_bbox[j].id):
                            print("!!!!!!!!!!! ALERT !!!!!!!!!!!")
                            print("Exchange Of Bag Detected Between People with ID - "+str(bag_owner[bag_bbox[i].id])+" and "+str(person_bbox[j].id))
                            print("")
                            bag_owner[bag_bbox[i].id] = person_bbox[j].id
                            bag_flag[bag_bbox[i].id]=1
                            flag_freq[bag_bbox[i].id]=30
                        elif(bag_owner[bag_bbox[i].id]==-1):
                            bag_owner[bag_bbox[i].id] = person_bbox[j].id





        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
