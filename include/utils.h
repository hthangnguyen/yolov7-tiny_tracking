#pragma once

constexpr int kNumColors = 32;

constexpr int kMaxCoastCycles = 5; // original = 1

constexpr int kMinHits = 2; // original = 3

// Set threshold to 0 to accept all detections
constexpr float kMinConfidence = 0; // original 0.6