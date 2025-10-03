#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <vector>
#include <memory>
#include <chrono>
#include "yolov11.h"

struct CameraFrame {
    cv::Mat frame;
    int cameraId;
    std::chrono::steady_clock::time_point timestamp;
};

class Pipeline {
public:
    Pipeline(const std::string& enginePath, nvinfer1::ILogger& logger);
    ~Pipeline();

    // Single image/video inference
    bool inferImage(const std::string& imagePath);
    bool inferVideo(const std::string& videoPath);
    
    // Multi-camera inference
    bool inferMultiCamera(const std::vector<int>& cameraIndices, int fps = 30);
    
    void stop();

private:
    // Frame grabber thread function
    void frameGrabberThread(int cameraId, int targetFps);
    
    // Inference thread function
    void inferenceThread();
    
    // Display thread function
    void displayThread();
    
    // Helper functions
    void processFrame(const CameraFrame& cameraFrame);
    std::string getWindowName(int cameraId);

    std::unique_ptr<YOLOv11> yolov11_;
    
    // Threading components
    std::vector<std::thread> grabberThreads_;
    std::thread inferenceThread_;
    std::thread displayThread_;
    
    // Synchronization
    std::mutex frameQueueMutex_;
    std::mutex resultQueueMutex_;
    std::queue<CameraFrame> frameQueue_;
    std::queue<std::pair<cv::Mat, int>> resultQueue_;
    
    std::atomic<bool> running_;
    
    // Camera management
    std::vector<cv::VideoCapture> cameras_;
    
    // Color constants
    const std::string RED_COLOR = "\033[31m";
    const std::string GREEN_COLOR = "\033[32m";
    const std::string RESET_COLOR = "\033[0m";
};
