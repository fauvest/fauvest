import jetson.inference
import jetson.utils
#import speech_dum_test as sp
import pyttsx3
engine = pyttsx3.init()

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("/dev/video0")      # '/dev/video0' for V4L2
#display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
display = jetson.utils.videoOutput('my_video.mp4') # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	#sp.play()
	#class_idx, confidence = net.Detect(img)
	#print('kjaskjdasjdj##################################################################################', class_idx)
	#print("detected {:d} objects in image".format(len(detections)))
	for detection in detections:
	    #print(detection.ClassID)
	    item=net.GetClassDesc(detection.ClassID)
	    print("the item is",item)
	    engine.say(item)
	    engine.runAndWait()
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))





