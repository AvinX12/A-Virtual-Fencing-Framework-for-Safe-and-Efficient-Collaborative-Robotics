# A-Virtual-Fencing-Framework-for-Safe-and-Efficient-Collaborative-Robotics

Paper submitted to the IEEE International Conference on Automation Science and Engineering (CASE) 2025 - [More Info Here](https://2025.ieeecase.org). <br />
Please go through the [project paper](https://github.com/AvinX12/A-Virtual-Fencing-Framework-for-Safe-and-Efficient-Collaborative-Robotics/blob/main/documents/IEEE_Paper_Submitted_To_CASE2025.pdf) on understanding on our work.

## Abstract

Collaborative robots (cobots) increasingly operate alongside humans, demanding robust real-time safeguarding. Current safety standards (e.g., ISO 10218, ANSI/RIA 15.06, ISO/TS 15066) require risk assessments but offer limited guidance for real-time responses. We propose a virtual fencing approach that detects and predicts human motion, ensuring safe cobot operation. Safety and performance tradeoffs are modeled as an optimization problem and solved via sequential quadratic programming. Experimental validation shows that our method minimizes operational pauses while maintaining safety, providing a modular solution for human-robot collaboration.

## Experimental Setup (Hardware)
1. Universal Robots UR16e - [official technical specification](https://www.universal-robots.com/media/1819035/ur16e-tech-spec_tcn_202105.pdf)
2. Nvidia Jetson Orin Nano - [official user guide](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/index.html)
3. Arducam IMX447 Camera - [official site](https://www.arducam.com/product/b0242-arducam-imx477-hq-camera/)
4. Arducam IMX477 UVC Camera Adapter Board - [official site](https://www.arducam.com/product/arducam-uvc-camera-adapter-board-for-12mp-imx477-raspberry-pi-hq-camera/)

<img src="https://github.com/AvinX12/A-Virtual-Fencing-Framework-for-Safe-and-Efficient-Collaborative-Robotics/blob/main/documents/media/real-setup.png" width="400" height="300">

## Software Setup Installation Instructions
1. Nvidia JetPack SDK 6.0 (Ubuntu 22.04 LTS) - [instructions](https://developer.nvidia.com/embedded/jetpack-sdk-60)
2. Real-Time Kernel Using OTA Update - [instructions](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html#real-time-kernel-using-ota-update)
3. ROS 2 Humble - [instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
4. UR Robot Drivers (Build From Source) - [instructions](https://docs.ros.org/en/humble/p/ur_robot_driver/doc/installation/installation.html#build-from-source)
5. OpenCV for Jetson Devices - [instructions](https://qengineering.eu/install-opencv-on-orin-nano.html)
6. Install Other Dependencies (as listed below)

#### Other dependencies
```bash
# Before installing dependencies, activate your Python virtual environment.
source <your_env>/bin/activate  # For Linux/macOS
<your_env>\Scripts\activate  # For Windows

# Install Dependencies
pip3 install ultralytics onnxruntime numpy
pip3 install jetson-inference jetson-utils
sudo apt install -y v4l-utils

# Check if Camera is Active
v4l2-ctl --list-devices
# It should return /dev/video0 or /dev/video1

# Install NVIDIA TensorRT
sudo apt install -y nvidia-tensorrt
# Verify TensorRT Installation
dpkg -l | grep nvidia-tensorrt

# Install Additional Requirements
# This requirements.txt file is available at ./installation/
pip install -r requirements.txt
```
![Status: Success](https://img.shields.io/badge/status-Success-brightgreen?style=flat-square) <br /> <br />

For more info on installation process, please refer to step-by step instructions from [installation guide](). <br />
![Work in Progress](https://img.shields.io/badge/status-Work%20in%20Progress-orange?style=flat-square)

## Experimentation
The robot operates at normal speed when no person is detected (top left in the figure below). However, when a person is detected (highlighted in yellow) in the increased attention zone (top right), it slows down. When a person is detected (highlighted in red) in the critical zone (bottom right), the robot stops. Once the person moves back to the increased attention zone (bottom left), the robot resumes slow movement. Once the area is clear, it returns to normal speed.

<img src="https://github.com/AvinX12/A-Virtual-Fencing-Framework-for-Safe-and-Efficient-Collaborative-Robotics/blob/main/documents/media/zone-scenarios.png" width="400" height="300">

## Results
In the absence of human detection, the robot completes six cycles in 60 seconds, which is considered as the operational efficiency of 100%. The latency of the system is the time taken by the ambient sensor to detect humans in designated zones, process the feed by the detection & SQP algorithms, issue the corresponding safeguarding command to the robot and execute that command. The collision avoidance rate quantifies the effectiveness of the system in safeguarding collisions between humans and robots.

The table below compares **Operational Efficiency (OE)**, **System Latency (SL)**, and **Collision Avoidance Rate (CAR)** for different safety methods.
| Method            | OE    | SL     | CAR  |
|------------------|------|-------|-----|
| **Immediate Stop Approach** | 61.8% | 31.4 ms | 98%  |
| **Zone-based**    | 66.7% | 32.7 ms | 98%  |
| **Zone-based + SQP-optimization** | 64.5% | 33.2 ms | 98%  |

The figure below shows effectiveness of velocity smoothening with and without SQP-based optimization. Zone-based detection is employed in both cases.

<img src="https://github.com/AvinX12/A-Virtual-Fencing-Framework-for-Safe-and-Efficient-Collaborative-Robotics/blob/main/documents/media/exp-2-results.png" width="400" height="250">

The video demonstration of our proposed zone-based SQP-optimization-enabled framework is available here: [YouTube]()
![Work in Progress](https://img.shields.io/badge/status-Work%20in%20Progress-orange?style=flat-square)

## Conclusion
In conclusion, this paper presented a modular safeguarding mechanism designed to enhance collaborative robotic operations in dynamic industrial environments. The proposed approach integrates a real-time human detection and prediction module with a UR16e workstation, allowing a flexible fencing framework. By formulating the safety–performance tradeoff as an optimization problem and solving it with SQP, a zone-based switching control strategy is achieved. Experimental results confirm that the virtual fencing method is cost-effective, exhibits low latency, and can be readily adapted to diverse industrial applications. Furthermore, minimized halts and smooth speed reductions were demonstrated, underscoring their critical role in maintaining operational efficiency.

## Acknowledgment
The authors sincerely thank Prof. Katsuo Kurabayashi and Ray Li for their invaluable guidance and encouragement throughout this project. The authors also express their gratitude to the NYU Tandon School of Engineering for providing the resources and facilities. This work was supported by Mechanical and Aerospace Engineering Department at Tandon School of Engineering in New York University.

## Authors
**Vineela Reddy Pippera Badguna** (vp2504@nyu.edu) - Mechanical and Aerospace Engineering Department, New York University, 5 MetroTech Center, Brooklyn, NY 11201, USA <br /> <br />
**Aliasghar Arab** - Mechanical and Aerospace Engineering Department, New York University, 5 MetroTech Center, Brooklyn, NY 11201, USA <br /> <br />
**Durga Avinash Kodavalla** (durga.avinash.k@nyu.edu) - Electrical and Computer Engineering Department, New York University, 5 MetroTech Center, Brooklyn, NY 11201, USA
