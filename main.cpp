#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "yolov11.h"
#include "application.h"
#include "pipeline.h"


/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;


int main(int argc, char* argv[]) {
    // Initialize the Logger
    Logger logger;
    
    // Initialize application and parse arguments
    Application app;
    Application::Config config;
    
    if (!app.parseArguments(argc, argv, config)) {
        return 1;
    }

    // Handle different modes
    if (config.mode == "convert") {
        try {
            // Initialize YOLOv11 with the ONNX model path
            YOLOv11 yolov11(config.onnxPath, logger);
            std::cout << "\033[32mModel conversion successful. Engine saved.\033[0m" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "\033[31mError during model conversion: " << e.what() << "\033[0m" << std::endl;
            return 1;
        }
    }
    else {
        try {
            // Initialize pipeline with the TensorRT engine path
            Pipeline pipeline(config.enginePath, logger);
            
            if (config.mode == "infer_video") {
                if (!pipeline.inferVideo(config.inputPath)) {
                    return 1;
                }
            }
            else if (config.mode == "infer_image") {
                if (!pipeline.inferImage(config.inputPath)) {
                    return 1;
                }
            }
            else if (config.mode == "infer_usb_cameras") {
                if (!pipeline.inferMultiCamera(config.cameraIndices, config.fps)) {
                    return 1;
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "\033[31mError during inference: " << e.what() << "\033[0m" << std::endl;
            return 1;
        }
    }

    return 0;
}