# Project: Road Scene Understanding for the Visually Impaired

In this project we wanted to build a device which can help visually impaired people while they are on the road. We wanted to make this device portable and light. We use realtime object detection using Nvidia's Jetson Nano developer kit. This can be powered by a powerbank and additional modules like camera, gps, WiFi etc can be attached using various ports available.

**Hardware List**
1. Jetson Nano developer kit 4GB
2. Micro SD card 256GB/512GB/1TB (Maybe SanDisk Ultra microSDHC memory card class 10)
3. Powerbank which supports 5v 3Amp
4. Camera: Logitech C920 (USB webcam)
5. Any external Mouse & Keyboard 
6. Wifi & Bluetooth: Intel 8265 Wireless card with antenna
7. USB to 3.5 mm converter
8. Chestplate to mount all above items together.
9. Straps to hold the chestplate 

![image](https://user-images.githubusercontent.com/87189221/125084065-66a3ef00-e0c9-11eb-902d-f078d325e38c.png)

Output Image: Overlapped image is on the left and mask image is on the right. It is able to segment the sidewalk and the road as well as detect objects like person, car etc using the object detection:
![image](https://github.com/fauvest/fauvest/blob/main/Output/b_output.jpg)

