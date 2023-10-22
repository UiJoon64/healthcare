import os
import cv2
import time
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt

def calculate_angle(a,b,c):
    a = np.array(a) # First point (x,y)
    b = np.array(b) # Mid point (x,y)
    c = np.array(c) # End point (x,y)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle,2)


@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="football1.mp4",device='cpu',view_img=False,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):

    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps
    
    device = select_device(opt.device) #select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
   
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
    else :
        cap = cv2.VideoCapture(source)    #pass video to videocapture object
   
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  #get video frame width
        frame_height = int(cap.get(4)) #get video frame height

        
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))

        angles_to_calculate = [(9, 7, 5), (7, 5, 11), (5,11,13),(11,13,15)]
        angle_dict = {angle: [] for angle in range(len(angles_to_calculate))}

        while(cap.isOpened): #loop until cap opened or video not complete
        
            print("Frame {} Processing".format(frame_count+1))

            ret, frame = cap.read()  #get frame and success from video capture
            
            if ret: #if success is true, means frame exist
                orig_image = frame #store frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  #convert image data to device
                image = image.float() #convert image to float precision (cpu)
                start_time = time.time() #start time for fps calculation
            
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)

                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh


                #index_a=0
                #index_b=1
                #index_c=2 for single assumption

                

                for i, pose in enumerate(output_data):  # detections per image
                
                    if len(output_data):  #check if no pose
                        for c in pose[:, 5].unique(): # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))
                        
                        max_conf_index = None
                        max_conf = -1
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            if conf > max_conf:
                                max_conf = conf
                                max_conf_index = det_index

                        if max_conf_index is not None:
                            det_index = max_conf_index 
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]

                            for angle_idx, (index_a,index_b,index_c) in enumerate(angles_to_calculate):
                                try:
                                    a_pt = kpts[index_a*3:index_a*3+2]
                                    b_pt = kpts[index_b*3:index_b*3+2]
                                    c_pt = kpts[index_c*3:index_c*3+2]

                                    angle_value = calculate_angle(a_pt,b_pt,c_pt)

                                    cv2.putText(im0,str(angle_value),(int(b_pt[0]),int(b_pt[1])),cv2.FONT_HERSHEY_SIMPLEX ,  
                                                0.5,(255),2,cv2.LINE_AA)

                                    angle_dict[angle_idx].append(angle_value)

                                except Exception as e:
                                    print(f"Error occurred: {str(e)}")


                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                        line_thickness=opt.line_thickness,kpt_label=True, kpts=kpts,
                                        steps=3,
                                        orig_shape=im0.shape[:2])

                
                end_time = time.time()  #Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps) #append FPS in list
                time_list.append(end_time - start_time) #append time in list
                
                # Stream results
                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)  # 1 millisecond

                out.write(im0)  #writing the video frame

            else:
                break

        cap.release()

        plt.figure()
        joint_names = ['elbow', 'shoulder', 'hip', 'knee']

        for idx, angles_list in angle_dict.items():
            plt.plot(angles_list,label=joint_names[idx])

        plt.title('Angle changes over time')
        plt.xlabel('Frame number')
        plt.ylabel('Angle')
        plt.legend()
        plt.savefig('joint_angle_changes.png')

        # 관절각도 CSV 파일로 저장하기
        df = pd.DataFrame(angle_dict)
        df.columns = joint_names
        df.to_csv(f'{source}_angles.csv', index=True)

        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        
        #plot the comparision graph
        plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    if os.path.isdir(opt.source):  # Check if source is a directory
        videos = [os.path.join(opt.source, v) for v in os.listdir(opt.source) if v.endswith('.mp4')]  # List all .mp4 files
        for video in videos:
            opt.source = video
            run(**vars(opt))
    else:
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
