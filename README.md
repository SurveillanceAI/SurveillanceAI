# Abstract Workshop Project Proposal: Surveillance AI

## 1. What is the problem you are trying to solve?
The problem being addressed is the detection of potential theft activities in retail shops by leveraging state-of-the-art computer vision techniques in addition to LLM/AI Agents to monitor security camera feeds. This aims to enhance shop security, reduce theft-related losses, and provide peace of mind to business owners.

---

## 2. Describe briefly, in high-level, your presumed solution.
The solution involves using real-time video analysis with computer vision models and AI agents to detect suspicious behavior. The system will analyze security camera feeds using object detection and abnormal human detection techniques and notify shop owners promptly if potential theft activities are identified.

---

## 3. Are there other approaches?
Yes, alternative approaches include:

- **Employing human monitoring** for live security footage.
- **Using traditional motion or activity detection systems.**
- **Implementing AI-powered generic surveillance solutions** without specific theft detection optimization.

Existing solutions are either manual or automated based on specific rules or a single AI model. Our solution aims to improve results by researching the effects of multi-model solutions (e.g., integrating both sound and vision) and exploring new approaches with agentic LLMs.

---

## 4. Who are the expected users of the application?
The primary users are small to medium-sized shop owners who already have security cameras installed in their shops.

---

## 5. What will be the main features and flows of the (different) user(s)?

### Shop Owner Features:
- **Real-Time Alerts**: Immediate notifications about detected suspicious activity.
- **Event Review**: Ability to view flagged incidents with snapshots or video segments.
- **Customizable Sensitivity**: Configure detection thresholds and notification preferences.
- **Dashboard**: Visualize live feeds, flagged incidents, and system status. The main goal is to make it as easy as possible for clients to view their data through an intuitive app.

### Additional Features (If Time Allows):
- **Predictive Order Optimization System**: Helps manage inventory by analyzing purchasing patterns.
- **Intelligent Shelf Restocking Tool**: Provides restocking recommendations based on customer behaviors.

### Flow:
1. System monitors camera feeds in real time.
2. Detects suspicious activities using trained AI models.
3. Logs events to the client’s dashboard for later review.
4. Notifies the client via their preferred notification method.
5. (Optional) Collects customer feedback.

---

## 6. Are there any external dependencies?
- **Hardware**: Security cameras and GPUs for processing.
- **Software Frameworks**:
  - Models like YOLO, MMAction2, and MMPose.
  - AI agents and LLM-based bots like VisionAgent for fast text-based results.
- **Data Sources**: Video data from shops, open datasets (e.g., Kaggle), and custom recordings.
- **Infrastructure**: GPU clusters from the college for computation.
- **Compliance**: Ensuring data privacy and ethics through agreements with shops for video usage.
