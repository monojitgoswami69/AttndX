# AI Smart Attendance System

## Overview
The AI Smart Attendance System is a computer-vision-based attendance application that uses face detection and face recognition to register students and track attendance automatically. The project combines webcam-based capture, face embeddings, session monitoring, and reporting into a single interactive Streamlit dashboard.

## Work Completed
- Built a complete Streamlit-based UI for registration, attendance, student management, and reports
- Implemented face registration workflow with multiple captured face images
- Added attendance session monitoring with scheduled checks and live session status
- Integrated YOLO-based face detection and ArcFace-style embedding support
- Added persistence for registered students and attendance records
- Implemented camera service, light monitoring, and session snapshot handling
- Added supporting modules for twin handling, liveness checks, and reporting

## What the System Does
- Registers new students by capturing multiple face images from a webcam
- Stores student identity data and face embeddings for later recognition
- Starts an attendance session and performs repeated checks during the session
- Detects faces in live camera frames and matches them with registered students
- Tracks attendance records and stores session results for review
- Provides a dashboard for viewing registered students and attendance reports

## Technology Stack
- Python
- Streamlit
- OpenCV
- NumPy
- Ultralytics YOLOv8
- ONNX Runtime
- Pillow
- FastAPI (integration support)
- Pickle-based local persistence

## Key Features
- Webcam-based student registration
- Face detection and recognition
- Attendance session workflow
- Real-time status monitoring
- Student gallery and management
- Report generation for attendance sessions
- Session snapshots and stored records
- Demo mode and normal mode configuration

## Project Status
The core application flow is implemented and runnable, including UI, registration, attendance session handling, and reporting. The project is ready for further enhancement such as improved accuracy, deployment, and cloud integration.
