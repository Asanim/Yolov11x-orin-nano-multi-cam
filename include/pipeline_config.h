#pragma once

/**
 * @file pipeline_config.h
 * @brief Configuration constants for pipeline optimization
 */

namespace PipelineConfig {
    // Threading configuration
    constexpr int NUM_INFERENCE_THREADS = 2;  // Adjust based on GPU capacity
    constexpr int MAX_FRAME_QUEUE_SIZE = 3;   // Lower for reduced latency
    constexpr int MAX_RESULT_QUEUE_SIZE = 3;  // Lower for reduced latency
    
    // Timing configuration (microseconds)
    constexpr int INFERENCE_THREAD_SLEEP_US = 100;
    constexpr int DISPLAY_THREAD_SLEEP_US = 500;
    
    // Camera optimization
    constexpr int CAMERA_BUFFER_SIZE = 1;     // Minimal buffer for low latency
    constexpr int DEFAULT_CAMERA_FPS = 30;
    constexpr int CAMERA_WIDTH = 640;         // Adjust based on requirements
    constexpr int CAMERA_HEIGHT = 480;
    
    // Memory optimization
    constexpr int DETECTION_VECTOR_RESERVE = 100; // Pre-allocate detection vector
    
    // Performance monitoring
    constexpr bool ENABLE_DETAILED_TIMING = true;
    constexpr int FPS_CALCULATION_INTERVAL = 30; // frames
}
