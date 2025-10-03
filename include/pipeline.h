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
#include "pipeline_config.h"

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
    
    // Inference thread function - now supports multiple threads
    void inferenceThread(int threadId);
    
    // Display thread function
    void displayThread();
    
    // Helper functions
    void processFrame(const CameraFrame& cameraFrame);
    std::string getWindowName(int cameraId);
    
    // Performance optimization functions
    bool tryPopFrame(CameraFrame& frame);
    void pushResult(cv::Mat&& frame, int cameraId);
    void optimizeCamera(cv::VideoCapture& cap);

    std::unique_ptr<YOLOv11> yolov11_;
    
    // Threading components - multiple inference threads for parallel processing
    std::vector<std::thread> grabberThreads_;
    std::vector<std::thread> inferenceThreads_;
    std::thread displayThread_;
    
    // High-performance lock-free queues
    std::mutex frameQueueMutex_;
    std::mutex resultQueueMutex_;
    std::queue<CameraFrame> frameQueue_;
    std::queue<std::pair<cv::Mat, int>> resultQueue_;
    
    // Thread pool configuration from config file
    static const int NUM_INFERENCE_THREADS = PipelineConfig::NUM_INFERENCE_THREADS;
    static const int MAX_QUEUE_SIZE = PipelineConfig::MAX_FRAME_QUEUE_SIZE;
    
    std::atomic<bool> running_;
    
    // Camera management
    std::vector<cv::VideoCapture> cameras_;
    
    // Color constants
    const std::string RED_COLOR = "\033[31m";
    const std::string GREEN_COLOR = "\033[32m";
    const std::string RESET_COLOR = "\033[0m";
};
