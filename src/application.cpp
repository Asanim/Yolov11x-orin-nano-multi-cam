#include "application.h"
#include <iostream>
#include <opencv2/opencv.hpp>

Application::Application() {}

Application::~Application() {}

bool Application::parseArguments(int argc, char* argv[], Config& config) {
    // Check for valid number of arguments
    if (argc < 4) {
        printUsage(argv[0]);
        return false;
    }

    // Parse command-line arguments
    config.mode = argv[1];
    config.inputPath = argv[2];
    config.enginePath = argv[3];

    // Handle different modes
    if (config.mode == "convert") {
        if (argc != 5) {
            printColoredText("Usage for conversion: ", YELLOW_COLOR);
            std::cout << argv[0] << " convert <onnx_path> <engine_path>" << std::endl;
            return false;
        }
        config.onnxPath = config.inputPath;  // In 'convert' mode, inputPath is actually onnx_path
    }
    else if (config.mode == "infer_video" || config.mode == "infer_image") {
        if (argc < 4 || argc > 5) {
            printColoredText("Usage for " + config.mode + ": ", YELLOW_COLOR);
            std::cout << argv[0] << " " << config.mode << " <input_path> <engine_path> [fps]" << std::endl;
            return false;
        }
        if (argc == 5) {
            config.fps = std::stoi(argv[4]);
        }
    }
    else if (config.mode == "infer_usb_cameras") {
        if (argc < 4 || argc > 5) {
            printColoredText("Usage for USB cameras: ", YELLOW_COLOR);
            std::cout << argv[0] << " infer_usb_cameras <engine_path> [fps]" << std::endl;
            return false;
        }
        config.useUsbCameras = true;
        config.enginePath = argv[2];  // Adjust argument positions
        if (argc == 4) {
            config.fps = std::stoi(argv[3]);
        }
        
        // Enumerate USB cameras
        config.cameraIndices = enumerateUsbCameras();
        if (config.cameraIndices.empty()) {
            printColoredText("No USB cameras found!", RED_COLOR);
            return false;
        }
    }
    else {
        printColoredText("Invalid mode. Use 'convert', 'infer_video', 'infer_image', or 'infer_usb_cameras'.", RED_COLOR);
        return false;
    }

    return true;
}

void Application::printUsage(const char* programName) {
    printColoredText("Usage: ", RED_COLOR);
    std::cout << programName << " <mode> <input_path> <engine_path> [options]" << std::endl;
    
    printColoredText("Modes:", YELLOW_COLOR);
    std::cout << "  convert                - Convert ONNX to TensorRT engine" << std::endl;
    std::cout << "  infer_video           - Inference on video file" << std::endl;
    std::cout << "  infer_image           - Inference on image file" << std::endl;
    std::cout << "  infer_usb_cameras     - Inference on USB cameras" << std::endl;
    
    printColoredText("Arguments:", YELLOW_COLOR);
    std::cout << "  <input_path>          - Path to input video/image or ONNX model" << std::endl;
    std::cout << "  <engine_path>         - Path to TensorRT engine file" << std::endl;
    std::cout << "  [fps]                 - Target FPS for camera capture (default: 30)" << std::endl;
}

std::vector<int> Application::enumerateUsbCameras() {
    std::vector<int> availableCameras;
    
    printColoredText("Enumerating USB cameras...", GREEN_COLOR);
    
    // Test camera indices from 0 to 9
    for (int i = 0; i < 24; ++i) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            // Try to read a frame to confirm the camera is working
            cv::Mat testFrame;
            if (cap.read(testFrame) && !testFrame.empty()) {
                availableCameras.push_back(i);
                std::cout << "Found camera at index: " << i << std::endl;
            }
            cap.release();
        }
    }
    
    if (!availableCameras.empty()) {
        printColoredText("Found " + std::to_string(availableCameras.size()) + " USB camera(s)", GREEN_COLOR);
    }
    
    return availableCameras;
}

void Application::printColoredText(const std::string& text, const std::string& color) {
    std::cout << color << text << RESET_COLOR;
}
