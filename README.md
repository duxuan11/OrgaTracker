# OrgaTracker
Involving in Organoids Growth and Recreation – Development of an Intelligent OrgaTracker System for Multi-Dimensional Organoid Analysis 
The system is divided into three parts:   
## 1.Organoid_tracking
The first part realizes organoid tracking and segmentation and organoid fusion capture.
![1231](https://user-images.githubusercontent.com/64136561/213462384-be285cfe-f595-4c46-b4cd-31a620fa2615.gif)
![res](https://user-images.githubusercontent.com/64136561/213459901-a060eb68-d5ac-4957-89f6-9c2eb132ce11.gif)
## 2.Intestinal_organoids_analysis
The second part shows the quantitative budding and the structural analysis of mouse intestinal organoids from day 0 to day 4.
<img width="1182" alt="2023-01-19_202642" src="https://user-images.githubusercontent.com/64136561/213442802-953141e1-93ee-42fb-acb2-5d02c98cf0c3.png">
<img width="1188" alt="2023-01-19_202917" src="https://user-images.githubusercontent.com/64136561/213443312-7e5575f1-5255-4b16-a341-487718222a64.png">
## 3.Organoid_sketches_to_images
The third part shows the generation of organoids from sketches.
      <img width="743" alt="20230119203553" src="https://user-images.githubusercontent.com/64136561/213444517-50330bf4-d584-42d2-b897-6bde95002b03.png">
      
      

Video: https://user-images.githubusercontent.com/64136561/213444202-5425bb0e-31d6-4b16-9b70-5cf8ee18474d.mp4


# Installation
---
Clone this project:
```
git clone git@github.com:duxuan11/OrgaTracker.git
```
## For Pip install
```
pip install -r requirements.txt
```
## Verify Installation
Run the demo:
### Organoid_tracking
```
python test.py
```
### Intestinal_organoid_analysis
```
python test.py
```
### Organoid_sketches_to_images
```
cd Organoid_sketches_to_images\pix2pix
python organ_test.py --dataroot ./datasets/edges2organoids --name edges2organoids_pix2pix --model pix2pix --direction AtoB
```
