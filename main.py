import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from ultralytics import YOLO
import cv2
import time
from paddleocr import PaddleOCR

# Load OCR Model
reader = PaddleOCR(rec_model_dir='./models/ocr', rec_char_dict_path='./models/ocr/custom_dict.txt')
results = {}

# load models
coco_model = YOLO('./models/yolov8n.pt')
np_model = YOLO('./models/license_plate_detector.pt')

# load video
 
cap = cv2.VideoCapture('Mobil Di Indonesia Lebih Bagus Bagus dan Baru Baru dibanding di Malaysia _ Cek Fakta ( Part 2 ).mp4') #using video
# cap = cv2.VideoCapture(1) # using camera/webcam

#Vehicle class in coco128
vehicles = [2, 3, 5, 7]
vehicle_bounding_boxes=[]

def read_license_plate(license_plate_crop):
    detections = reader.ocr(license_plate_crop,cls=False,det=False,rec=True)
    return detections


def write_csv(results, output_path):
    
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(
            'frame_number', 'track_id', 'car_bbox', 'car_bbox_score',
            'license_plate_bbox', 'license_plate_bbox_score', 'license_plate_number',
            'license_text_score'))

        for frame_number in results.keys():
            for track_id in results[frame_number].keys():
                print(results[frame_number][track_id])
                if 'car' in results[frame_number][track_id].keys() and \
                   'license_plate' in results[frame_number][track_id].keys() and \
                   'number' in results[frame_number][track_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        frame_number,
                        track_id,
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['car']['bbox'][0],
                            results[frame_number][track_id]['car']['bbox'][1],
                            results[frame_number][track_id]['car']['bbox'][2],
                            results[frame_number][track_id]['car']['bbox'][3]
                        ),
                        results[frame_number][track_id]['car']['bbox_score'],
                        '[{} {} {} {}]'.format(
                            results[frame_number][track_id]['license_plate']['bbox'][0],
                            results[frame_number][track_id]['license_plate']['bbox'][1],
                            results[frame_number][track_id]['license_plate']['bbox'][2],
                            results[frame_number][track_id]['license_plate']['bbox'][3]
                        ),
                        results[frame_number][track_id]['license_plate']['bbox_score'],
                        results[frame_number][track_id]['license_plate']['number'],
                        results[frame_number][track_id]['license_plate']['text_score'])
                    )
        f.close()

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    start_time = time.time()
    if ret :
        results[frame_nmr] = {}

        # use track() to identify instances and track them frame by frame, take only with conf> 0.5
        detections = coco_model.track(frame, persist=True,classes=vehicles,conf=0.5)[0]
        # save cropped detections
        detections.save_crop('outputs')


        for detection in detections.boxes.data.tolist():
            score=-1
            if len(detection) ==7:
                x1, y1, x2, y2, track_id, score, class_id = detection
            else: # assign track_id = -1 if track id not presence
                x1, y1, x2, y2, score, class_id = detection
                track_id=-1
            vehicle_bounding_boxes=[]
            vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
            for bbox in vehicle_bounding_boxes:
                roi = frame[int(y1):int(y2), int(x1):int(x2)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)

                # license plate detector for region of interest
                license_plates = np_model.predict(roi,conf=0.7)[0]
                # check every bounding box for a license plate
                license_plates.save_crop('outputs_lp')
                for license_plate in license_plates.boxes.data.tolist():
                    plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                    # verify detections

                    print(license_plate, 'track_id: ' + str(bbox[4]))
                    # take only plate with high score
                    plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                     # posterize
                    _, plate_treshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # OCR
                    np_text, np_score = read_license_plate(plate_treshold)[0][0]
                    # if plate could be read write results
                    print(frame_nmr,track_id)
                    if np_text is not None:
                        # object details
                        org = [int(x1), int(y1)]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(frame, np_text, org, font, fontScale, color, thickness)
                        results[frame_nmr][track_id] = {
                            'car': {
                                'bbox': [x1, y1, x2, y2],
                                'bbox_score': score
                            },
                            'license_plate': {
                                'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                'bbox_score': plate_score,
                                'number': np_text,
                                'text_score': np_score
                            }
                        }
                    else:
                        org = [int(x1), int(y1)]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(frame, "None", org, font, fontScale, color, thickness) # if plate number is not recognized
                        
        end_time = time.time()                        
        elapsed_time =  end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds for frame number: {frame_nmr} ")  
    cv2.imshow('Webcam', frame)
    name = f"./output_frame/frame_{frame_nmr}.jpg"
    cv2.imwrite(name, frame)     # uncomment this to save frame
    if cv2.waitKey(1) == ord('q'):
        break
write_csv(results, './results.csv')
cap.release()


