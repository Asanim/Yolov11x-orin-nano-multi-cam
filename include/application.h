#pragma once

#include <string>
#include <vector>

class Application {
public:
    struct Config {
        std::string mode;
        std::string inputPath;
        std::string enginePath;
        std::string onnxPath;
        int fps = 30;
        bool useUsbCameras = false;
        std::vector<int> cameraIndices;
    };

    Application();
    ~Application();

    bool parseArguments(int argc, char* argv[], Config& config);
    void printUsage(const char* programName);
    std::vector<int> enumerateUsbCameras();

private:
    void printColoredText(const std::string& text, const std::string& color);
    
    // Color constants
    const std::string RED_COLOR = "\033[31m";
    const std::string GREEN_COLOR = "\033[32m";
    const std::string YELLOW_COLOR = "\033[33m";
    const std::string RESET_COLOR = "\033[0m";
};
