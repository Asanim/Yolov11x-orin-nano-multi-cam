#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <vector>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <set>
#include "yolov11.h"
#include "pipeline_config.h"

// Forward declarations for better performance
struct Detection;

struct CameraFrame {
    cv::Mat frame;
    int cameraId;
    std::chrono::steady_clock::time_point timestamp;
    
    // Default constructor
    CameraFrame() = default;
    
    // Move constructor for better performance
    CameraFrame(CameraFrame&& other) noexcept 
        : frame(std::move(other.frame)), cameraId(other.cameraId), timestamp(other.timestamp) {}
    
    // Move assignment operator
    CameraFrame& operator=(CameraFrame&& other) noexcept {
        if (this != &other) {
            frame = std::move(other.frame);
            cameraId = other.cameraId;
            timestamp = other.timestamp;
        }
        return *this;
    }
};

// GPU processing result after preprocessing and inference
struct GPUProcessingResult {
    YOLOv11* yolo_instance;  // Raw pointer to thread-specific instance
    int cameraId;
    std::chrono::steady_clock::time_point timestamp;
    cv::Mat original_frame;  // Keep original for drawing
};

// Final processing result ready for display
struct ProcessedFrame {
    cv::Mat frame;
    int cameraId;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<Detection> detections;
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
    // === Thread Functions for Pipeline Architecture ===
    
    // Stage 1: Frame grabber threads (one per camera)
    void frameGrabberThread(int cameraId, int targetFps);
    
    // Stage 2: GPU processing threads (preprocessing + inference)
    void gpuProcessingThread(int threadId);
    
    // Stage 3: CPU postprocessing threads (NMS + detection extraction)
    void cpuPostprocessingThread(int threadId);
    
    // Stage 4: Drawing threads (CPU-intensive bbox drawing)
    void drawingThread(int threadId);
    
    // Stage 5: Display thread
    void displayThread();
    
    // === Helper Functions ===
    void processFrameGPU(const CameraFrame& cameraFrame, int threadId);
    void processFrameCPU(const GPUProcessingResult& gpuResult, int threadId);
    void drawDetections(ProcessedFrame& result, int threadId);
    std::string getWindowName(int cameraId);
    
    // === Optimized Queue Operations ===
    bool tryPopFrame(CameraFrame& frame);
    bool tryPopGPUResult(GPUProcessingResult& result);
    bool tryPopCPUResult(ProcessedFrame& result);
    bool tryPopDrawingResult(ProcessedFrame& result);
    
    void pushGPUResult(GPUProcessingResult&& result);
    void pushCPUResult(ProcessedFrame&& result);
    void pushDrawingResult(ProcessedFrame&& result);
    void pushFinalResult(cv::Mat&& frame, int cameraId);
    
    void optimizeCamera(cv::VideoCapture& cap);

    // === YOLO Instances ===
    std::unique_ptr<YOLOv11> yolov11_main_;
    std::vector<std::unique_ptr<YOLOv11>> yolov11_gpu_instances_;  // For GPU processing
    
    // === Thread Pools ===
    std::vector<std::thread> frameGrabberThreads_;
    std::vector<std::thread> gpuProcessingThreads_;
    std::vector<std::thread> cpuPostprocessingThreads_;
    std::vector<std::thread> drawingThreads_;
    std::thread displayThread_;
    
    // === Pipeline Queues with Mutexes ===
    // Stage 1 -> Stage 2: Raw frames
    std::mutex frameQueueMutex_;
    std::queue<CameraFrame> frameQueue_;
    std::condition_variable frameAvailable_;
    
    // Stage 2 -> Stage 3: GPU processed frames
    std::mutex gpuResultQueueMutex_;
    std::queue<GPUProcessingResult> gpuResultQueue_;
    std::condition_variable gpuResultAvailable_;
    
    // Stage 3 -> Stage 4: CPU processed frames (with detections)
    std::mutex cpuResultQueueMutex_;
    std::queue<ProcessedFrame> cpuResultQueue_;
    std::condition_variable cpuResultAvailable_;
    
    // Stage 4 -> Stage 5: Frames with drawn detections
    std::mutex drawingResultQueueMutex_;
    std::queue<ProcessedFrame> drawingResultQueue_;
    std::condition_variable drawingResultAvailable_;
    
    // Final display queue
    std::mutex displayQueueMutex_;
    std::queue<std::pair<cv::Mat, int>> displayQueue_;
    
    // === Configuration ===
    static const int NUM_GPU_THREADS = PipelineConfig::NUM_GPU_THREADS;
    static const int NUM_CPU_THREADS = PipelineConfig::NUM_CPU_THREADS;
    static const int NUM_DRAWING_THREADS = PipelineConfig::NUM_DRAWING_THREADS;
    
    std::atomic<bool> running_;
    
    // Camera management
    std::vector<cv::VideoCapture> cameras_;
    
    // === Performance Monitoring ===
    std::atomic<uint64_t> totalFramesProcessed_{0};
    std::chrono::steady_clock::time_point pipelineStartTime_;
    
    // Color constants
    const std::string RED_COLOR = "\033[31m";
    const std::string GREEN_COLOR = "\033[32m";
    const std::string RESET_COLOR = "\033[0m";
};
