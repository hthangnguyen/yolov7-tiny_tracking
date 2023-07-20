
#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "tracker.h"
#include "utils.h"

std::vector<std::string> classNames = {"car", "truck", "bus", "human", "motorbike", "bicycle"};
const cv::Scalar color = cv::Scalar(255, 255, 0);
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.3;
const float CONFIDENCE_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.4;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect_<float> box;
};

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::dnn::Net network, cv::Mat &img, std::vector<Detection>& output, bool draw = false)
{
    auto input_image = format_yolov5(img);
    cv::Mat blob;
    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    network.setInput(blob);

    std::vector<cv::Mat> outputs;
    network.forward(outputs, network.getUnconnectedOutLayersNames());

    float* data = (float*)outputs[0].data;

    const int dimensions = 11;
    const int rows = 25200;

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float* classes_scores = data + 5;
            cv::Mat scores(1, classNames.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

        data += dimensions;

    }
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    if (draw)
        for (int i = 0; i < output.size(); ++i)
        {
            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            cv::rectangle(img, box, color, 3);
            cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(img, classNames[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
}

int main()
{
    std::string video = "tracking_test.mp4";

    cv::VideoCapture cap(video);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file\n";
        return -1;
    }
    std::string model = "for_trackingv5n.onnx";
    auto results = cv::dnn::readNet(model);
    std::vector<Detection> output;
    results.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    results.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::dnn::Net network = results;
    auto start = std::chrono::high_resolution_clock::now();
    // cv::VideoWriter outcap("tracking_output.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 10, cv::Size(960, 960));

    Tracker tracker;
    int frame_index = 0;
    while (1) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        std::vector<Detection> output;
        detect(network, frame, output, true);
        std::vector<cv::Rect> dets;
        for (unsigned int i = 0; i < output.size(); i++) {
            dets.push_back(output[i].box);
        }
        tracker.Run(dets);
        const auto tracks = tracker.GetTracks();

        for (auto& trk : tracks) {
            // only draw tracks which meet certain criteria
            if (trk.second.coast_cycles_ < kMaxCoastCycles &&
                (trk.second.hit_streak_ >= kMinHits || frame_index < kMinHits)) 
            {  
                const auto& bbox = trk.second.GetStateAsBbox();
                cv::putText(frame, std::to_string(trk.first), cv::Point(bbox.tl().x, bbox.tl().y - 10),
                    cv::FONT_HERSHEY_DUPLEX, 2, cv::Scalar(255, 255, 255), 2);
                cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 3);
            }
        }

        std::string rst = "det num = " + std::to_string(output.size()) + ", trk num = " + std::to_string(tracks.size());
        cv::putText(frame, rst, cv::Point(0, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::imshow("result", frame);
        char c = (char)cv::waitKey(0);
        if (c == 27) // ESC to stop 
            break;
        frame_index++;
    }
    cv::destroyAllWindows();
    cap.release();
}