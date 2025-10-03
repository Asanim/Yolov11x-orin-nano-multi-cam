#include "pipeline.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <set>
#include <pthread.h>  // For thread priorities
#include <sched.h>    // For CPU affinity

Pipeline::Pipeline(const std::string& enginePath, nvinfer1::ILogger& logger)
    : running_(false) {
    
    // Create main YOLO instance
    yolov11_main_ = std::make_unique<YOLOv11>(enginePath, logger);
    
    // Create GPU processing instances (one per GPU thread)
    yolov11_gpu_instances_.reserve(NUM_GPU_THREADS);
    for (int i = 0; i < NUM_GPU_THREADS; ++i) {
        yolov11_gpu_instances_.push_back(yolov11_main_->createThreadInstance());
    }
    
    // Initialize pipeline start time for performance monitoring
    pipelineStartTime_ = std::chrono::steady_clock::now();
    
    std::cout << GREEN_COLOR << "Pipeline initialized with " << NUM_GPU_THREADS 
              << " GPU threads, " << NUM_CPU_THREADS << " CPU threads, and " 
              << NUM_DRAWING_THREADS << " drawing threads" << RESET_COLOR << std::endl;
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::inferImage(const std::string& imagePath) {
    try {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Read the image
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << RED_COLOR << "Failed to read image: " << imagePath << RESET_COLOR << std::endl;
            return false;
        }

        auto preprocessStart = std::chrono::high_resolution_clock::now();
        // Preprocess the image
        yolov11_main_->preprocess(image);
        auto preprocessEnd = std::chrono::high_resolution_clock::now();

        auto inferenceStart = std::chrono::high_resolution_clock::now();
        // Perform inference
        yolov11_main_->infer();
        auto inferenceEnd = std::chrono::high_resolution_clock::now();

        auto postprocessStart = std::chrono::high_resolution_clock::now();
        // Postprocess to get detections
        std::vector<Detection> detections;
        yolov11_main_->postprocess(detections);
        auto postprocessEnd = std::chrono::high_resolution_clock::now();

        // Calculate timing
        auto preprocessTime = std::chrono::duration_cast<std::chrono::microseconds>(preprocessEnd - preprocessStart).count() / 1000.0;
        auto inferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(inferenceEnd - inferenceStart).count() / 1000.0;
        auto postprocessTime = std::chrono::duration_cast<std::chrono::microseconds>(postprocessEnd - postprocessStart).count() / 1000.0;
        auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(postprocessEnd - startTime).count() / 1000.0;

        std::cout << GREEN_COLOR << "Timing Info - Preprocess: " << std::fixed << std::setprecision(2) 
                  << preprocessTime << "ms, Inference: " << inferenceTime 
                  << "ms, Postprocess: " << postprocessTime << "ms, Total: " << totalTime << "ms" << RESET_COLOR << std::endl;

        // Draw detections on the image
        yolov11_main_->draw(image, detections);

        // Display the image
        cv::imshow("Inference", image);
        cv::waitKey(0);

        // Save the output image
        std::string outputImagePath = "output_image.jpg";
        cv::imwrite(outputImagePath, image);
        std::cout << GREEN_COLOR << "Image inference completed. Output saved to " 
                  << outputImagePath << RESET_COLOR << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << RED_COLOR << "Error during image inference: " << e.what() << RESET_COLOR << std::endl;
        return false;
    }
}

bool Pipeline::inferVideo(const std::string& videoPath) {
    try {
        // Open the video file
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << RED_COLOR << "Failed to open video file: " << videoPath << RESET_COLOR << std::endl;
            return false;
        }

        // Prepare video writer to save the output
        std::string outputVideoPath = "output_video.avi";
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        cv::VideoWriter video(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30,
                             cv::Size(frame_width, frame_height));

        cv::Mat frame;
        int frameCount = 0;
        auto lastFpsTime = std::chrono::high_resolution_clock::now();
        
        while (cap.read(frame)) {
            auto frameStart = std::chrono::high_resolution_clock::now();
            
            auto preprocessStart = std::chrono::high_resolution_clock::now();
            // Preprocess the frame
            yolov11_main_->preprocess(frame);
            auto preprocessEnd = std::chrono::high_resolution_clock::now();

            auto inferenceStart = std::chrono::high_resolution_clock::now();
            // Perform inference
            yolov11_main_->infer();
            auto inferenceEnd = std::chrono::high_resolution_clock::now();

            auto postprocessStart = std::chrono::high_resolution_clock::now();
            // Postprocess to get detections
            std::vector<Detection> detections;
            yolov11_main_->postprocess(detections);
            auto postprocessEnd = std::chrono::high_resolution_clock::now();

            // Draw detections on the frame
            yolov11_main_->draw(frame, detections);

            auto frameEnd = std::chrono::high_resolution_clock::now();
            
            // Calculate timing for this frame
            auto preprocessTime = std::chrono::duration_cast<std::chrono::microseconds>(preprocessEnd - preprocessStart).count() / 1000.0;
            auto inferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(inferenceEnd - inferenceStart).count() / 1000.0;
            auto postprocessTime = std::chrono::duration_cast<std::chrono::microseconds>(postprocessEnd - postprocessStart).count() / 1000.0;
            auto totalFrameTime = std::chrono::duration_cast<std::chrono::microseconds>(frameEnd - frameStart).count() / 1000.0;
            
            frameCount++;
            
            // Calculate FPS every 30 frames
            if (frameCount % 30 == 0) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastFpsTime).count();
                double fps = 30000.0 / timeDiff;
                
                std::cout << GREEN_COLOR << "Frame " << frameCount << " - FPS: " << std::fixed << std::setprecision(1) << fps
                          << ", Inference: " << std::setprecision(2) << inferenceTime << "ms"
                          << ", Total: " << totalFrameTime << "ms" << RESET_COLOR << std::endl;
                
                lastFpsTime = currentTime;
            }

            // Display the frame
            cv::imshow("Inference", frame);
            if (cv::waitKey(1) == 27) { // Exit on 'ESC' key
                break;
            }

            // Write the frame to the output video
            video.write(frame);
        }

        cap.release();
        video.release();
        cv::destroyAllWindows();
        std::cout << GREEN_COLOR << "Video inference completed. Output saved to " 
                  << outputVideoPath << RESET_COLOR << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << RED_COLOR << "Error during video inference: " << e.what() << RESET_COLOR << std::endl;
        return false;
    }
}

bool Pipeline::inferMultiCamera(const std::vector<int>& cameraIndices, int fps) {
    try {
        running_ = true;
        
        // Initialize cameras
        cameras_.clear();
        cameras_.reserve(cameraIndices.size());
        
        for (int cameraId : cameraIndices) {
            cv::VideoCapture cap(cameraId);
            if (!cap.isOpened()) {
                std::cerr << RED_COLOR << "Failed to open camera " << cameraId << RESET_COLOR << std::endl;
                continue;
            }
            cameras_.push_back(std::move(cap));
        }
        
        if (cameras_.empty()) {
            std::cerr << RED_COLOR << "No cameras could be opened" << RESET_COLOR << std::endl;
            return false;
        }
        
        std::cout << GREEN_COLOR << "Starting multi-camera inference with " 
                  << cameras_.size() << " cameras at " << fps << " FPS" << RESET_COLOR << std::endl;
        
        // Optimize camera settings for low latency
        for (auto& camera : cameras_) {
            optimizeCamera(camera);
        }
        
        // Start frame grabber threads (one per camera)
        frameGrabberThreads_.reserve(cameraIndices.size());
        for (size_t i = 0; i < cameraIndices.size(); ++i) {
            if (i < cameras_.size()) {
                frameGrabberThreads_.emplace_back(&Pipeline::frameGrabberThread, this, cameraIndices[i], fps);
            }
        }
        
        // Start GPU processing threads
        gpuProcessingThreads_.reserve(NUM_GPU_THREADS);
        for (int i = 0; i < NUM_GPU_THREADS; ++i) {
            gpuProcessingThreads_.emplace_back(&Pipeline::gpuProcessingThread, this, i);
        }
        
        // Start CPU postprocessing threads
        cpuPostprocessingThreads_.reserve(NUM_CPU_THREADS);
        for (int i = 0; i < NUM_CPU_THREADS; ++i) {
            cpuPostprocessingThreads_.emplace_back(&Pipeline::cpuPostprocessingThread, this, i);
        }
        
        // Start drawing threads
        drawingThreads_.reserve(NUM_DRAWING_THREADS);
        for (int i = 0; i < NUM_DRAWING_THREADS; ++i) {
            drawingThreads_.emplace_back(&Pipeline::drawingThread, this, i);
        }
        
        // Start display thread
        displayThread_ = std::thread(&Pipeline::displayThread, this);
        
        // Wait for user to press ESC to stop
        std::cout << GREEN_COLOR << "Press ESC in any window to stop..." << RESET_COLOR << std::endl;
        while (running_) {
            if (cv::waitKey(30) == 27) { // ESC key
                break;
            }
        }
        
        stop();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << RED_COLOR << "Error during multi-camera inference: " << e.what() << RESET_COLOR << std::endl;
        return false;
    }
}

void Pipeline::stop() {
    running_ = false;
    
    // Notify all condition variables to wake up waiting threads
    frameAvailable_.notify_all();
    gpuResultAvailable_.notify_all();
    cpuResultAvailable_.notify_all();
    drawingResultAvailable_.notify_all();
    
    // Join all thread pools
    for (auto& thread : frameGrabberThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    for (auto& thread : gpuProcessingThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    for (auto& thread : cpuPostprocessingThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    for (auto& thread : drawingThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    if (displayThread_.joinable()) {
        displayThread_.join();
    }
    
    // Release cameras
    for (auto& camera : cameras_) {
        if (camera.isOpened()) {
            camera.release();
        }
    }
    cameras_.clear();
    
    // Clear all thread vectors
    frameGrabberThreads_.clear();
    gpuProcessingThreads_.clear();
    cpuPostprocessingThreads_.clear();
    drawingThreads_.clear();
    
    cv::destroyAllWindows();
}

void Pipeline::frameGrabberThread(int cameraId, int targetFps) {
    auto frameDuration = std::chrono::milliseconds(1000 / targetFps);
    auto nextFrameTime = std::chrono::steady_clock::now();
    
    // Find the camera index in our cameras vector
    size_t cameraIndex = 0;
    for (size_t i = 0; i < cameras_.size(); ++i) {
        // This is a simplified approach - in practice you'd want to track camera IDs better
        if (i < cameras_.size()) {
            cameraIndex = i;
            break;
        }
    }
    
    while (running_) {
        if (cameraIndex < cameras_.size() && cameras_[cameraIndex].isOpened()) {
            cv::Mat frame;
            if (cameras_[cameraIndex].read(frame) && !frame.empty()) {
                CameraFrame cameraFrame;
                cameraFrame.frame = std::move(frame); // Use move semantics to avoid copying
                cameraFrame.cameraId = cameraId;
                cameraFrame.timestamp = std::chrono::steady_clock::now();
                
                {
                    std::lock_guard<std::mutex> lock(frameQueueMutex_);
                    // Drop old frames if queue is full for low latency
                    while (frameQueue_.size() >= PipelineConfig::MAX_FRAME_QUEUE_SIZE) {
                        frameQueue_.pop();
                    }
                    frameQueue_.push(std::move(cameraFrame));
                }
                frameAvailable_.notify_one(); // Notify GPU threads
            }
        }
        
        // Maintain target FPS
        nextFrameTime += frameDuration;
        std::this_thread::sleep_until(nextFrameTime);
    }
}

void Pipeline::gpuProcessingThread(int threadId) {
    while (running_) {
        CameraFrame cameraFrame;
        
        if (tryPopFrame(cameraFrame)) {
            processFrameGPU(cameraFrame, threadId);
        } else {
            // Use shorter sleep for better responsiveness
            std::this_thread::sleep_for(std::chrono::microseconds(PipelineConfig::GPU_THREAD_SLEEP_US));
        }
    }
}

void Pipeline::cpuPostprocessingThread(int threadId) {
    while (running_) {
        GPUProcessingResult gpuResult;
        
        if (tryPopGPUResult(gpuResult)) {
            processFrameCPU(gpuResult, threadId);
        } else {
            // Use shorter sleep for better responsiveness
            std::this_thread::sleep_for(std::chrono::microseconds(PipelineConfig::CPU_THREAD_SLEEP_US));
        }
    }
}

void Pipeline::drawingThread(int threadId) {
    while (running_) {
        ProcessedFrame processedFrame;
        
        if (tryPopCPUResult(processedFrame)) {
            drawDetections(processedFrame, threadId);
            pushDrawingResult(std::move(processedFrame));
        } else {
            // Use shorter sleep for better responsiveness
            std::this_thread::sleep_for(std::chrono::microseconds(PipelineConfig::DRAWING_THREAD_SLEEP_US));
        }
    }
}

void Pipeline::displayThread() {
    // Create windows once at startup for better performance
    std::set<int> createdWindows;
    
    while (running_) {
        ProcessedFrame result;
        bool hasResult = false;
        
        {
            std::lock_guard<std::mutex> lock(drawingResultQueueMutex_);
            if (!drawingResultQueue_.empty()) {
                result = std::move(drawingResultQueue_.front());
                drawingResultQueue_.pop();
                hasResult = true;
            }
        }
        
        if (hasResult) {
            std::string windowName = getWindowName(result.cameraId);
            
            // Create window only once per camera
            if (createdWindows.find(result.cameraId) == createdWindows.end()) {
                cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                createdWindows.insert(result.cameraId);
            }
            
            cv::imshow(windowName, result.frame);
            
            // Update performance counters
            totalFramesProcessed_.fetch_add(1);
        } else {
            // Shorter sleep for better responsiveness
            std::this_thread::sleep_for(std::chrono::microseconds(PipelineConfig::DISPLAY_THREAD_SLEEP_US));
        }
        
        // Process window events more frequently
        cv::waitKey(1);
    }
}



std::string Pipeline::getWindowName(int cameraId) {
    return "Camera " + std::to_string(cameraId) + " - YOLOv11 Inference";
}

// Performance optimization functions
bool Pipeline::tryPopFrame(CameraFrame& frame) {
    std::lock_guard<std::mutex> lock(frameQueueMutex_);
    if (!frameQueue_.empty()) {
        frame = std::move(frameQueue_.front());
        frameQueue_.pop();
        return true;
    }
    return false;
}

bool Pipeline::tryPopGPUResult(GPUProcessingResult& result) {
    std::lock_guard<std::mutex> lock(gpuResultQueueMutex_);
    if (!gpuResultQueue_.empty()) {
        result = std::move(gpuResultQueue_.front());
        gpuResultQueue_.pop();
        return true;
    }
    return false;
}

bool Pipeline::tryPopCPUResult(ProcessedFrame& result) {
    std::lock_guard<std::mutex> lock(cpuResultQueueMutex_);
    if (!cpuResultQueue_.empty()) {
        result = std::move(cpuResultQueue_.front());
        cpuResultQueue_.pop();
        return true;
    }
    return false;
}

bool Pipeline::tryPopDrawingResult(ProcessedFrame& result) {
    std::lock_guard<std::mutex> lock(drawingResultQueueMutex_);
    if (!drawingResultQueue_.empty()) {
        result = std::move(drawingResultQueue_.front());
        drawingResultQueue_.pop();
        return true;
    }
    return false;
}



void Pipeline::pushGPUResult(GPUProcessingResult&& result) {
    {
        std::lock_guard<std::mutex> lock(gpuResultQueueMutex_);
        // Drop old results if queue is full for low latency
        while (gpuResultQueue_.size() >= PipelineConfig::MAX_GPU_RESULT_QUEUE_SIZE) {
            gpuResultQueue_.pop();
        }
        gpuResultQueue_.push(std::move(result));
    }
    gpuResultAvailable_.notify_one();
}

void Pipeline::pushCPUResult(ProcessedFrame&& result) {
    {
        std::lock_guard<std::mutex> lock(cpuResultQueueMutex_);
        // Drop old results if queue is full for low latency
        while (cpuResultQueue_.size() >= PipelineConfig::MAX_CPU_RESULT_QUEUE_SIZE) {
            cpuResultQueue_.pop();
        }
        cpuResultQueue_.push(std::move(result));
    }
    cpuResultAvailable_.notify_one();
}

void Pipeline::pushDrawingResult(ProcessedFrame&& result) {
    {
        std::lock_guard<std::mutex> lock(drawingResultQueueMutex_);
        // Drop old results if queue is full for low latency
        while (drawingResultQueue_.size() >= PipelineConfig::MAX_DRAWING_QUEUE_SIZE) {
            drawingResultQueue_.pop();
        }
        drawingResultQueue_.push(std::move(result));
    }
    drawingResultAvailable_.notify_one();
}

void Pipeline::pushFinalResult(cv::Mat&& frame, int cameraId) {
    std::lock_guard<std::mutex> lock(displayQueueMutex_);
    // Drop old results if queue is full for low latency
    while (displayQueue_.size() >= PipelineConfig::MAX_DISPLAY_QUEUE_SIZE) {
        displayQueue_.pop();
    }
    displayQueue_.push({std::move(frame), cameraId});
}

void Pipeline::optimizeCamera(cv::VideoCapture& cap) {
    // Set camera properties for optimal performance and low latency
    cap.set(cv::CAP_PROP_BUFFERSIZE, PipelineConfig::CAMERA_BUFFER_SIZE);
    cap.set(cv::CAP_PROP_FPS, PipelineConfig::DEFAULT_CAMERA_FPS);
    
    // Try to enable hardware acceleration if available
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    
    // Set optimal resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, PipelineConfig::CAMERA_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, PipelineConfig::CAMERA_HEIGHT);
    
    // Disable auto-exposure and auto-focus for consistent performance
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // Manual exposure
    cap.set(cv::CAP_PROP_AUTOFOCUS, 0);        // Disable autofocus
}

void Pipeline::processFrameGPU(const CameraFrame& cameraFrame, int threadId) {
    try {
        auto processStart = std::chrono::steady_clock::now();
        
        // Get thread-specific YOLO instance
        auto& yolo_instance = yolov11_gpu_instances_[threadId];
        
        // Use reference to avoid copying
        cv::Mat& frame = const_cast<cv::Mat&>(cameraFrame.frame);
        
        auto preprocessStart = std::chrono::steady_clock::now();
        // Asynchronous preprocessing with dedicated stream
        yolo_instance->preprocessAsync(frame);
        auto preprocessEnd = std::chrono::steady_clock::now();
        
        auto inferenceStart = std::chrono::steady_clock::now();
        // Asynchronous inference
        yolo_instance->inferAsync();
        auto inferenceEnd = std::chrono::steady_clock::now();
        
        // Create GPU result and pass to CPU processing
        GPUProcessingResult gpuResult;
        gpuResult.yolo_instance = yolo_instance.get(); // Use raw pointer instead of moving
        gpuResult.cameraId = cameraFrame.cameraId;
        gpuResult.timestamp = cameraFrame.timestamp;
        gpuResult.original_frame = frame; // Copy the frame for CPU processing
        
        // Push to CPU processing queue
        pushGPUResult(std::move(gpuResult));
        
        // Performance monitoring
        if (PipelineConfig::ENABLE_STAGE_TIMING) {
            auto preprocessTime = std::chrono::duration_cast<std::chrono::microseconds>(preprocessEnd - preprocessStart).count() / 1000.0;
            auto inferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(inferenceEnd - inferenceStart).count() / 1000.0;
            
            static std::atomic<int> gpuFrameCounter{0};
            if (gpuFrameCounter.fetch_add(1) % 30 == 0) {
                std::cout << GREEN_COLOR << "[GPU-T" << threadId << "] Cam " << cameraFrame.cameraId 
                          << " - Preprocess: " << std::fixed << std::setprecision(2) << preprocessTime << "ms"
                          << ", Inference: " << inferenceTime << "ms" << RESET_COLOR << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << RED_COLOR << "[GPU-T" << threadId << "] Error processing frame from camera " 
                  << cameraFrame.cameraId << ": " << e.what() << RESET_COLOR << std::endl;
    }
}

void Pipeline::processFrameCPU(const GPUProcessingResult& gpuResult, int threadId) {
    try {
        auto processStart = std::chrono::steady_clock::now();
        
        // Wait for GPU processing to complete if needed
        while (!gpuResult.yolo_instance->isReady() && running_) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        auto postprocessStart = std::chrono::steady_clock::now();
        
        // Pre-allocate detection vector for better performance
        std::vector<Detection> detections;
        detections.reserve(PipelineConfig::DETECTION_VECTOR_RESERVE);
        
        // Asynchronous postprocessing (NMS and detection extraction)
        gpuResult.yolo_instance->postprocessAsync(detections);
        
        auto postprocessEnd = std::chrono::steady_clock::now();
        
        // Create processed frame result
        ProcessedFrame processedFrame;
        processedFrame.frame = gpuResult.original_frame.clone(); // Clone for drawing
        processedFrame.cameraId = gpuResult.cameraId;
        processedFrame.timestamp = gpuResult.timestamp;
        processedFrame.detections = std::move(detections);
        
        // Push to drawing queue
        pushCPUResult(std::move(processedFrame));
        
        // Performance monitoring
        if (PipelineConfig::ENABLE_STAGE_TIMING) {
            auto postprocessTime = std::chrono::duration_cast<std::chrono::microseconds>(postprocessEnd - postprocessStart).count() / 1000.0;
            
            static std::atomic<int> cpuFrameCounter{0};
            if (cpuFrameCounter.fetch_add(1) % 30 == 0) {
                std::cout << GREEN_COLOR << "[CPU-T" << threadId << "] Cam " << gpuResult.cameraId 
                          << " - Postprocess: " << std::fixed << std::setprecision(2) << postprocessTime 
                          << "ms, Detections: " << processedFrame.detections.size() << RESET_COLOR << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << RED_COLOR << "[CPU-T" << threadId << "] Error processing frame from camera " 
                  << gpuResult.cameraId << ": " << e.what() << RESET_COLOR << std::endl;
    }
}

void Pipeline::drawDetections(ProcessedFrame& result, int threadId) {
    try {
        auto drawStart = std::chrono::steady_clock::now();
        
        // Use the main YOLO instance for drawing (thread-safe for read-only operations)
        yolov11_main_->draw(result.frame, result.detections);
        
        auto drawEnd = std::chrono::steady_clock::now();
        
        // Performance monitoring
        if (PipelineConfig::ENABLE_STAGE_TIMING) {
            auto drawTime = std::chrono::duration_cast<std::chrono::microseconds>(drawEnd - drawStart).count() / 1000.0;
            
            static std::atomic<int> drawFrameCounter{0};
            if (drawFrameCounter.fetch_add(1) % 30 == 0) {
                std::cout << GREEN_COLOR << "[DRAW-T" << threadId << "] Cam " << result.cameraId 
                          << " - Draw: " << std::fixed << std::setprecision(2) << drawTime << "ms" << RESET_COLOR << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << RED_COLOR << "[DRAW-T" << threadId << "] Error drawing frame from camera " 
                  << result.cameraId << ": " << e.what() << RESET_COLOR << std::endl;
    }
}
