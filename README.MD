1. Project Objective & Scope
The project is a real-time, gesture-based desktop control system designed to mimic an "Iron Man-style" interface. The primary objective is to build a robust portfolio piece that bridges the gap into Cloud Computing, AI/ML, and MLOps roles by demonstrating end-to-end machine learning infrastructure, multi-model orchestration, and cloud deployment, rather than just simple API integration.

2. Infrastructure & MLOps Stack
The backend and deployment pipeline will be engineered to simulate a production-grade cloud environment.

Version Control & CI/CD: GitHub Actions will automate the testing and building of the environment upon code commits.

Containerization: Docker will be used to encapsulate the ML models, dependencies, and OS integration scripts into reproducible, lightweight images.

Container Orchestration: A Kubernetes cluster (deployed via Amazon EKS) will manage the Docker containers, demonstrating the ability to scale compute resources and maintain uptime.

Cloud Computing: AWS will serve as the backbone for hosting the cluster, routing inference requests, and storing the processed datasets.

3. Data & Neural Architecture
The system will rely on a Two-Stream Pipeline, utilizing two distinct PyTorch models running concurrently to handle fundamentally different data types.

Stream 1 (Static Spatial Data):

Dataset: HaGRID (HAnd Gesture Recognition Image Dataset).

Function: A 2D Convolutional Neural Network (CNN) evaluates individual frames to detect static functional poses (e.g., a closed fist for "grab", an open palm for "stop").

Stream 2 (Dynamic Spatiotemporal Data):

Dataset: 20bn-Jester Dataset.

Function: A 3D CNN or sequence model evaluates a rolling buffer of frames to detect moving gestures over time (e.g., a left-to-right hand sweep for "swipe left").

Orchestration: The pipeline bridges these two models in real-time, evaluating the webcam feed and dispatching the recognized classification to the application layer.

4. Execution & OS Integration Layer
The translation of ML classifications into tangible desktop actions will be handled locally.

Execution Library: pyautogui will be utilized to inject discrete OS commands and hotkeys.

Trigger Mechanism: Instead of continuous cursor tracking, gestures will strictly fire atomic, discrete actions (e.g., Alt + Tab, taking screenshots, minimizing windows).

State Management: A stateful dispatcher with built-in debounce logic (cooldown timers) will sit between the PyTorch output and pyautogui to prevent a single gesture from triggering multiple duplicate hotkey executions across consecutive video frames.