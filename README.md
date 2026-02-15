## Camera Calibration using Zhang’s Method  
**RBE 549 – Computer Vision | HW1**
### Overview
This project implements camera calibration from scratch using Zhang’s planar calibration method.  
The goal is to estimate the intrinsic camera matrix \(K\), radial distortion coefficients \((k_1, k_2)\), and the extrinsic parameters \((R, t)\) from multiple images of a checkerboard calibration target.

The calibration target used is a 9 × 6 inner-corner checkerboard printed on A4 paper with each square having a side length of 21.5 mm.

### Directory Structure
```
nchandramouli_hw1/      
│── Results/                
│── Wrapper.py
│── calib.py              
│── Report.pdf              
|__ README.md
```
### Dependencies
Install the required Python packages:

```bash
pip install numpy opencv-python scipy
```

### How to Run
Place all calibration images inside:

```
Calibration_Imgs/
```

Then execute:

```bash
python Wrapper.py
```

The reprojection images will be stored in the folder:

```bash
Results/
```
