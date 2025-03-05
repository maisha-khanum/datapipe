#include "rclcpp/rclcpp.hpp"
#include <cstdio>
#include <memory>


// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>
// #include <Eigen/SVD>

#include <time.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <map> //dictionary equivalent

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_ros/buffer.h>
#include "tf2/exceptions.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> //necessary for doTransform


#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include "sensor_msgs/image_encodings.hpp"

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
// #include <visualization_msgs/msg/marker.hpp>
// #include <visualization_msgs/msg/marker_array.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//for pausing so variables can be filled for service
#include <chrono>
#include <thread>


//include the service for this package
// #include <me326_locobot_example/PixtoPoint.h>


//Setup the class:
class Matching_Pix_to_Ptcld : public rclcpp::Node
{
public:
	Matching_Pix_to_Ptcld(); //constructor declaration

	// Make callback functions for subscribers
	void info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
	void color_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
	void masked_img_gen(const cv::Mat& color_img, const cv::Mat& mask);
	// bool run_gsam2(const std::string& image_path, const std::string& mask_path);
	bool run_gsam2(const cv::Mat& image_arr);
	std::vector<cv::Point> extract_mask_pixels(const cv::Mat& mask);

private:
	rclcpp::QoS qos_; //message reliability

	// Publisher declarations
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_color_filt_pub_;

	//Subscriber declaration
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_image_sub_;

	std::vector<geometry_msgs::msg::Point> mask_img_pixels_;

	std::string color_image_topic_;
	std::string depth_img_camera_info_;

	image_geometry::PinholeCameraModel camera_model_;
	bool depth_cam_info_ready_;

	// TF Listener
	std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
	std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

Matching_Pix_to_Ptcld::Matching_Pix_to_Ptcld() 
    : Node("Matching_Pix_to_Ptcld"), qos_(2)
{
	tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
	tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

	this->declare_parameter<std::string>("pt_srv_color_img_topic", "/d455/color/image_raw");
	this->declare_parameter<std::string>("pt_srv_depth_img_cam_info_topic", "/d455/depth/camera_info");

	//   std::string color_image_topic_;
	this->get_parameter("pt_srv_color_img_topic", color_image_topic_);

	//   std::string depth_img_camera_info_;
	this->get_parameter("pt_srv_depth_img_cam_info_topic", depth_img_camera_info_);


	qos_.reliability(rclcpp::ReliabilityPolicy::BestEffort);

	image_color_filt_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/processed_d455/image_gsam2_mask", 1);

	cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
		depth_img_camera_info_, qos_, std::bind(&Matching_Pix_to_Ptcld::info_callback, this, std::placeholders::_1));
	
	rgb_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
		color_image_topic_, qos_, std::bind(&Matching_Pix_to_Ptcld::color_image_callback, this, std::placeholders::_1));

	depth_cam_info_ready_ = false;
}

void Matching_Pix_to_Ptcld::masked_img_gen(const cv::Mat& color_img, const cv::Mat& mask) {
	if (color_img.empty() || mask.empty()) {
		RCLCPP_ERROR(rclcpp::get_logger("MK_matching_ptcld"), "Masked RGB generation failed: empty input.");
		return;
	}

	// Ensure mask is binary
	cv::Mat binary_mask;
	cv::threshold(mask, binary_mask, 1, 255, cv::THRESH_BINARY);

	// Apply mask to the color image
	cv::Mat masked_img;
	color_img.copyTo(masked_img, binary_mask);

	// Convert to ROS Image message
	cv_bridge::CvImage cv_bridge_masked_image;
	cv_bridge_masked_image.header.stamp = rclcpp::Clock().now();
	cv_bridge_masked_image.header.frame_id = "camera_frame";
	cv_bridge_masked_image.encoding = sensor_msgs::image_encodings::RGB8;
	cv_bridge_masked_image.image = masked_img;

	// Publish masked image
	sensor_msgs::msg::Image ros_masked_image;
	cv_bridge_masked_image.toImageMsg(ros_masked_image);
	image_color_filt_pub_->publish(ros_masked_image);
	RCLCPP_INFO(this->get_logger(), "Published ros_masked_img.");
}

// bool Matching_Pix_to_Ptcld::run_gsam2(const std::string& image_path, const std::string& mask_path) {
bool Matching_Pix_to_Ptcld::run_gsam2(const cv::Mat& image_arr) {
    std::string python_path = "/home/mkhanum/miniconda3/envs/GSAM2Env/bin/python"; //Replace with your python path
    std::string script_path = "/home/mkhanum/datapipe/ros2_ws/src/smartbelt_segmentation/scripts/run_gsam2.py";
    // std::string command = python_path + " " + script_path + " " + image_path + " " + mask_path;
	std::string command = python_path + " " + script_path + " " + image_arr;
    int ret_code = std::system(command.c_str());
    // return (ret_code == 0);

	if (ret_code == 0) {
        RCLCPP_INFO(this->get_logger(), "GSAM2 segmentation finished successfully.");
        return true;
    } else {
        RCLCPP_ERROR(this->get_logger(), "GSAM2 segmentation failed with return code: %d", ret_code);
        return false;
    }
}

std::vector<cv::Point> Matching_Pix_to_Ptcld::extract_mask_pixels(const cv::Mat& mask) {
	std::vector<cv::Point> mask_pixels;
	for (int y = 0; y < mask.rows; ++y) {
		for (int x = 0; x < mask.cols; ++x) {
			if (mask.at<uchar>(y, x) > 0) {
				mask_pixels.emplace_back(x, y);
			}
		}
	}
	return mask_pixels;
}

void Matching_Pix_to_Ptcld::color_image_callback(const sensor_msgs::msg::Image::SharedPtr msg){
	cv_bridge::CvImagePtr color_img_ptr;
	try{
		color_img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);}
	catch (cv_bridge::Exception& e){
		RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
		return;}

	cv::Mat color_img = color_img_ptr->image;
	RCLCPP_INFO(this->get_logger(), "Color image shape: rows=%d, cols=%d, channels=%d", 
            color_img.rows, color_img.cols, color_img.channels());

	std::string image_path = "/tmp/frame.png";
	std::string mask_path = "/tmp/mask.png";

	run_gsam2(color_img);

	// if (!cv::imwrite(image_path, color_img)){
	// 	RCLCPP_ERROR(this->get_logger(), "Failed to save image as PNG.");
	// 	return;}

	// if (!run_gsam2(image_path, mask_path)){
	// 	RCLCPP_ERROR(this->get_logger(), "GSAM2 segmentation failed.");
	// 	return;}

	// cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);

	// if (mask.empty()){
	// 	RCLCPP_ERROR(this->get_logger(), "Failed to load segmentation mask.");
	// 	return;}

	// std::vector<cv::Point> mask_img_pixels_ = extract_mask_pixels(mask);
	RCLCPP_INFO(this->get_logger(), "Extracted %lu mask pixels.", mask_img_pixels_.size());

	// masked_img_gen(color_img, mask);
}



void Matching_Pix_to_Ptcld::info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg){
	//create a camera model from the camera info
	camera_model_.fromCameraInfo(msg);
	depth_cam_info_ready_ = true;	
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Matching_Pix_to_Ptcld>());
  rclcpp::shutdown();
  return 0;
}