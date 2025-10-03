#include "pipeline.h"
#include <iostream>
#include <chrono>
#include <iomanip>

Pipeline::Pipeline(const std::string& enginePath, nvinfer1::ILogger& logger)
    : running_(false) {
    yolov11_ = std::make_unique<YOLOv11>(enginePath, logger);
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
        yolov11_->preprocess(image);
        auto preprocessEnd = std::chrono::high_resolution_clock::now();

        auto inferenceStart = std::chrono::high_resolution_clock::now();
        // Perform inference
        yolov11_->infer();
        auto inferenceEnd = std::chrono::high_resolution_clock::now();

        auto postprocessStart = std::chrono::high_resolution_clock::now();
        // Postprocess to get detections
        std::vector<Detection> detections;
        yolov11_->postprocess(detections);
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
        yolov11_->draw(image, detections);

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
            yolov11_->preprocess(frame);
            auto preprocessEnd = std::chrono::high_resolution_clock::now();

            auto inferenceStart = std::chrono::high_resolution_clock::now();
            // Perform inference
            yolov11_->infer();
            auto inferenceEnd = std::chrono::high_resolution_clock::now();

            auto postprocessStart = std::chrono::high_resolution_clock::now();
            // Postprocess to get detections
            std::vector<Detection> detections;
            yolov11_->postprocess(detections);
            auto postprocessEnd = std::chrono::high_resolution_clock::now();

            // Draw detections on the frame
            yolov11_->draw(frame, detections);

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
        
        // Start frame grabber threads
        for (size_t i = 0; i < cameraIndices.size(); ++i) {
            if (i < cameras_.size()) {
                grabberThreads_.emplace_back(&Pipeline::frameGrabberThread, this, cameraIndices[i], fps);
            }
        }
        
        // Start inference thread
        inferenceThread_ = std::thread(&Pipeline::inferenceThread, this);
        
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
    
    // Join all threads
    for (auto& thread : grabberThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    if (inferenceThread_.joinable()) {
        inferenceThread_.join();
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
    grabberThreads_.clear();
    
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
                cameraFrame.frame = frame.clone();
                cameraFrame.cameraId = cameraId;
                cameraFrame.timestamp = std::chrono::steady_clock::now();
                
                {
                    std::lock_guard<std::mutex> lock(frameQueueMutex_);
                    frameQueue_.push(cameraFrame);
                    
                    // Limit queue size to prevent memory issues
                    while (frameQueue_.size() > 10) {
                        frameQueue_.pop();
                    }
                }
            }
        }
        
        // Maintain target FPS
        nextFrameTime += frameDuration;
        std::this_thread::sleep_until(nextFrameTime);
    }
}

void Pipeline::inferenceThread() {
    while (running_) {
        CameraFrame cameraFrame;
        bool hasFrame = false;
        
        {
            std::lock_guard<std::mutex> lock(frameQueueMutex_);
            if (!frameQueue_.empty()) {
                cameraFrame = frameQueue_.front();
                frameQueue_.pop();
                hasFrame = true;
            }
        }
        
        if (hasFrame) {
            processFrame(cameraFrame);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void Pipeline::displayThread() {
    while (running_) {
        std::pair<cv::Mat, int> result;
        bool hasResult = false;
        
        {
            std::lock_guard<std::mutex> lock(resultQueueMutex_);
            if (!resultQueue_.empty()) {
                result = resultQueue_.front();
                resultQueue_.pop();
                hasResult = true;
            }
        }
        
        if (hasResult) {
            std::string windowName = getWindowName(result.second);
            cv::imshow(windowName, result.first);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void Pipeline::processFrame(const CameraFrame& cameraFrame) {
    try {
        auto processStart = std::chrono::steady_clock::now();
        cv::Mat frame = cameraFrame.frame.clone();
        
        auto preprocessStart = std::chrono::steady_clock::now();
        // Preprocess the frame
        yolov11_->preprocess(frame);
        auto preprocessEnd = std::chrono::steady_clock::now();
        
        auto inferenceStart = std::chrono::steady_clock::now();
        // Perform inference
        yolov11_->infer();
        auto inferenceEnd = std::chrono::steady_clock::now();
        
        auto postprocessStart = std::chrono::steady_clock::now();
        // Postprocess to get detections
        std::vector<Detection> detections;
        yolov11_->postprocess(detections);
        auto postprocessEnd = std::chrono::steady_clock::now();
        
        // Draw detections on the frame
        yolov11_->draw(frame, detections);
        
        auto processEnd = std::chrono::steady_clock::now();
        
        // Calculate timing
        auto captureDelay = std::chrono::duration_cast<std::chrono::milliseconds>(processStart - cameraFrame.timestamp).count();
        auto preprocessTime = std::chrono::duration_cast<std::chrono::microseconds>(preprocessEnd - preprocessStart).count() / 1000.0;
        auto inferenceTime = std::chrono::duration_cast<std::chrono::microseconds>(inferenceEnd - inferenceStart).count() / 1000.0;
        auto postprocessTime = std::chrono::duration_cast<std::chrono::microseconds>(postprocessEnd - postprocessStart).count() / 1000.0;
        auto totalProcessTime = std::chrono::duration_cast<std::chrono::microseconds>(processEnd - processStart).count() / 1000.0;
        
        // Print timing info for each camera
        std::cout << GREEN_COLOR << "Cam " << cameraFrame.cameraId 
                  << " - Delay: " << captureDelay << "ms"
                  << ", Inference: " << std::fixed << std::setprecision(2) << inferenceTime << "ms"
                  << ", Total: " << totalProcessTime << "ms" 
                  << ", inference FPS: " << (1000.0 / inferenceTime) << "Hz"
                  << ", overall FPS: " << (1000.0 / totalProcessTime) << "Hz"
                  << RESET_COLOR << std::endl;

        // Add frame to result queue
        {
            std::lock_guard<std::mutex> lock(resultQueueMutex_);
            resultQueue_.push({frame, cameraFrame.cameraId});
            
            // Limit queue size
            while (resultQueue_.size() > 5) {
                resultQueue_.pop();
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << RED_COLOR << "Error processing frame from camera " 
                  << cameraFrame.cameraId << ": " << e.what() << RESET_COLOR << std::endl;
    }
}

std::string Pipeline::getWindowName(int cameraId) {
    return "Camera " + std::to_string(cameraId) + " - YOLOv11 Inference";
}
