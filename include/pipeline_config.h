#pragma once

/**
 * @file pipeline_config.h
 * @brief Configuration constants for pipeline optimization
 */

namespace PipelineConfig {
    // === Thread Pool Configuration ===
    constexpr int NUM_GPU_THREADS = 2;        // GPU preprocessing + inference threads
    constexpr int NUM_CPU_THREADS = 3;        // CPU postprocessing threads (NMS, detection extraction)
    constexpr int NUM_DRAWING_THREADS = 2;    // CPU-intensive drawing threads
    
    // === Queue Size Configuration (smaller for lower latency) ===
    constexpr int MAX_FRAME_QUEUE_SIZE = 2;         // Raw camera frames
    constexpr int MAX_GPU_RESULT_QUEUE_SIZE = 3;    // GPU processed results
    constexpr int MAX_CPU_RESULT_QUEUE_SIZE = 3;    // CPU processed results
    constexpr int MAX_DRAWING_QUEUE_SIZE = 2;       // Drawing queue
    constexpr int MAX_DISPLAY_QUEUE_SIZE = 2;       // Final display queue
    
    // === Thread Sleep Configuration (microseconds) ===
    constexpr int GPU_THREAD_SLEEP_US = 50;         // Aggressive GPU polling
    constexpr int CPU_THREAD_SLEEP_US = 100;        // CPU thread sleep
    constexpr int DRAWING_THREAD_SLEEP_US = 200;    // Drawing thread sleep
    constexpr int DISPLAY_THREAD_SLEEP_US = 500;    // Display thread sleep
    
    // === Timeout Configuration (milliseconds) ===
    constexpr int QUEUE_WAIT_TIMEOUT_MS = 5;        // Queue condition variable timeout
    
    // === Camera Optimization ===
    constexpr int CAMERA_BUFFER_SIZE = 1;           // Minimal buffer for low latency
    constexpr int DEFAULT_CAMERA_FPS = 30;
    constexpr int CAMERA_WIDTH = 640;               // Adjust based on requirements
    constexpr int CAMERA_HEIGHT = 480;
    
    // === Memory Optimization ===
    constexpr int DETECTION_VECTOR_RESERVE = 100;   // Pre-allocate detection vector
    constexpr int FRAME_MEMORY_POOL_SIZE = 10;      // Pre-allocated frame buffers
    
    // === Performance Monitoring ===
    constexpr bool ENABLE_DETAILED_TIMING = true;
    constexpr int FPS_CALCULATION_INTERVAL = 30;    // frames
    constexpr bool ENABLE_STAGE_TIMING = false;     // Detailed per-stage timing
    
    // === GPU Stream Management ===
    constexpr bool USE_MULTIPLE_STREAMS = true;     // Use separate CUDA streams per thread
    constexpr int STREAM_PRIORITY_HIGH = -1;        // High priority streams
    constexpr int STREAM_PRIORITY_LOW = 0;          // Low priority streams
}
