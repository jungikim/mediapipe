// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// April, 2020
// SYSTRAN Software, Inc.
// MediaPipe Graph to extract Hand Keypoints from Multihand Tracking model and storing them in a JSON file
//

#include <cstdlib>
#include <fstream>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

constexpr char kInputStream[] = "input_video";
constexpr char kLandmarksStream[] = "multi_hand_landmarks";

DEFINE_string(calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path,             "", "Full path of video to load. If not provided, attempt to use a webcam.");
DEFINE_string(output_json_path,             "", "Full path of where to save result (.json). If not provided, show result in stdout.");


::mediapipe::Status InitMPPGragph(mediapipe::CalculatorGraph &graph) {
	std::string calculator_graph_config_contents;
	MP_RETURN_IF_ERROR(mediapipe::file::GetContents(FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
	LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;
	mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

	LOG(INFO) << "Initialize the calculator graph.";
	MP_RETURN_IF_ERROR(graph.Initialize(config));

	return ::mediapipe::OkStatus();
}

::mediapipe::Status RunMPPGraph(mediapipe::CalculatorGraph &graph,
								const std::string &input_video_path,
								const std::string &output_json_path) {
  LOG(INFO) << "Load the video.";
  cv::VideoCapture capture;
  if (!input_video_path.empty()) {
    capture.open(input_video_path);
  }
  RET_CHECK(capture.isOpened());

  std::ofstream jsonOutF(output_json_path.c_str());
  jsonOutF << "{ \"frames\": [" << std::endl;

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark,
		  	  	   graph.AddOutputStreamPoller(kLandmarksStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  int frameIdx = 0;
  while (true) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    // End of video.
    if (camera_frame_raw.empty())
    	break;
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
								mediapipe::ImageFormat::SRGB,
								camera_frame.cols,
								camera_frame.rows,
								mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
								kInputStream,
								mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet landmark_packet;
    if (!poller_landmark.Next(&landmark_packet)) break;

    auto& output_landmarks = landmark_packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();

    if (frameIdx > 0){
    	jsonOutF << "," << std::endl;
    }
    jsonOutF << "\t{" << std::endl;
    for (int handIdx = 0 ; handIdx < output_landmarks.size(); handIdx++){
    	const ::mediapipe::NormalizedLandmarkList& landmarks = output_landmarks.at(handIdx);
        jsonOutF << "\t\"" << handIdx << "\": [";
		for (int keypoint = 0; keypoint < landmarks.landmark_size(); keypoint++) {
			const ::mediapipe::NormalizedLandmark& landmark = landmarks.landmark(keypoint);
			if(keypoint!=0) jsonOutF << "\t";
			jsonOutF << "\t" << landmark.x() << ", "
							 << landmark.y() << ", "
							 << landmark.z();
			if (keypoint+1 == landmarks.landmark_size())
				jsonOutF << " ]";
			else
				jsonOutF << "," << std::endl;
		}
        if (handIdx+1 != output_landmarks.size())
        	jsonOutF << ",";
        jsonOutF << std::endl;
    }
    jsonOutF << "\t}";
    frameIdx++;
  }

  jsonOutF << std::endl
		   << "], " << std::endl
		   << "  \"num_frames\": " << frameIdx << std::endl << "}" << std::endl;

  jsonOutF.close();

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status CloseMPPGraph(mediapipe::CalculatorGraph &graph) {
	LOG(INFO) << "Shutting down.";
	return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ::mediapipe::Status run_status;
  mediapipe::CalculatorGraph graph;

  run_status = InitMPPGragph(graph);
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to initialize the graph: " << run_status.message();
    return EXIT_FAILURE;
  }

  run_status = RunMPPGraph(graph, FLAGS_input_video_path, FLAGS_output_json_path);
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  }

  run_status = CloseMPPGraph(graph);
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to close the graph: " << run_status.message();
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Success!";
  return EXIT_SUCCESS;
}
