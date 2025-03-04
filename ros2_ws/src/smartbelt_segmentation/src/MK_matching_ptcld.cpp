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
	void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
	void color_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
	void masked_img_gen(const cv::Mat& color_img, const cv::Mat& mask);
	void masked_pc_gen(const std::vector<geometry_msgs::msg::PointStamped> masked_points_);
	bool run_gsam2(const std::string& image_path, const std::string& mask_path);
	std::vector<cv::Point> extract_mask_pixels(const cv::Mat& mask);

private:
	rclcpp::QoS qos_; //message reliability

	// Publisher declarations
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_color_filt_pub_;
	rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_filt_pub_;

	//Subscriber declaration
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
	rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_image_sub_;

	std::vector<geometry_msgs::msg::Point> mask_img_pixels_;

	std::string color_image_topic_;
	std::string depth_img_camera_info_;
	std::string pc_topic_;

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
	this->declare_parameter<std::string>("pt_srv_reg_pt_cld_topic", "/d455/depth/color/points");

	//   std::string color_image_topic_;
	this->get_parameter("pt_srv_color_img_topic", color_image_topic_);

	//   std::string depth_img_camera_info_;
	this->get_parameter("pt_srv_depth_img_cam_info_topic", depth_img_camera_info_);

	//   std::string pc_topic_;
	this->get_parameter("pt_srv_reg_pt_cld_topic", pc_topic_);

	qos_.reliability(rclcpp::ReliabilityPolicy::BestEffort);

	image_color_filt_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/processed_d455/image_gsam2_mask", 1);
	pc_filt_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/processed_d455/points_gsam2_mask", 1);

	cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
		depth_img_camera_info_, qos_, std::bind(&Matching_Pix_to_Ptcld::info_callback, this, std::placeholders::_1));

	pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
		pc_topic_, qos_, std::bind(&Matching_Pix_to_Ptcld::pointcloud_callback, this, std::placeholders::_1));
	
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
}

bool Matching_Pix_to_Ptcld::run_gsam2(const std::string& image_path, const std::string& mask_path) {
    std::string python_path = "/home/mkhanum/miniconda3/envs/GSAM2Env/bin/python"; //Replace with your python path
    std::string script_path = "/home/mkhanum/datapipe/ros2_ws/src/smartbelt_segmentation/scripts/run_gsam2.py";
    std::string command = python_path + " " + script_path + " " + image_path + " " + mask_path;
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
	std::string image_path = "/tmp/frame.png";
	std::string mask_path = "/tmp/mask.png";

	if (!cv::imwrite(image_path, color_img)){
		RCLCPP_ERROR(this->get_logger(), "Failed to save image as PNG.");
		return;}

	if (!run_gsam2(image_path, mask_path)){
		RCLCPP_ERROR(this->get_logger(), "GSAM2 segmentation failed.");
		return;}

	cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
	if (mask.empty()){
		RCLCPP_ERROR(this->get_logger(), "Failed to load segmentation mask.");
		return;}

	std::vector<cv::Point> mask_pixels = extract_mask_pixels(mask);
	RCLCPP_INFO(this->get_logger(), "Extracted %lu mask pixels.", mask_pixels.size());

	masked_img_gen(color_img, mask);
}


void Matching_Pix_to_Ptcld::masked_pc_gen(const std::vector<geometry_msgs::msg::PointStamped> masked_points_) {
	if (masked_points_.empty()) {
		RCLCPP_WARN(this->get_logger(), "No valid masked points to publish");
		return;
	}

	// Initialize point cloud message
	sensor_msgs::msg::PointCloud2 pc;
	pc.header.stamp = rclcpp::Clock().now();
	pc.header.frame_id = camera_model_.tfFrame();
	pc.height = 1;
	pc.width = masked_points_.size();
	pc.is_dense = false;

	// Define point cloud fields
	sensor_msgs::PointCloud2Modifier modifier(pc);
	modifier.setPointCloud2FieldsByString(1, "xyz");
	sensor_msgs::PointCloud2Iterator<float> iter_x(pc, "x");
	sensor_msgs::PointCloud2Iterator<float> iter_y(pc, "y");
	sensor_msgs::PointCloud2Iterator<float> iter_z(pc, "z");

	// Transform each point and populate point cloud
	for (const auto& point : masked_points_) {
		geometry_msgs::msg::PointStamped transformed_point;
		try {
			tf_buffer_->transform(point, transformed_point, camera_model_.tfFrame());
		} catch (tf2::TransformException &ex) {
			RCLCPP_ERROR(this->get_logger(), "TF Transform Exception: %s", ex.what());
			continue;
		}

		*iter_x = transformed_point.point.x;
		*iter_y = transformed_point.point.y;
		*iter_z = transformed_point.point.z;
		++iter_x; ++iter_y; ++iter_z;
	}

	// Publish the masked point cloud
	pc_filt_pub_->publish(pc);
}

void Matching_Pix_to_Ptcld::info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg){
	//create a camera model from the camera info
	camera_model_.fromCameraInfo(msg);
	depth_cam_info_ready_ = true;	
}

void Matching_Pix_to_Ptcld::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromROSMsg(*msg, *cloud);

	std::vector<geometry_msgs::msg::PointStamped> points_in_camera_frame;

	for (const auto& pixel : mask_img_pixels_){
		int u = pixel.x;
		int v = pixel.y;

		int index = v * msg->width + u;
		if (index >= cloud->points.size()){
			RCLCPP_WARN(this->get_logger(), "Skipping pixel (%d, %d) - out of bounds", u, v);
			continue;
		}

		pcl::PointXYZ point = cloud->points[index];

		if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)){
			RCLCPP_WARN(this->get_logger(), "Skipping invalid point at pixel (%d, %d)", u, v);
			continue;
		}

		geometry_msgs::msg::PointStamped point_msg;
		point_msg.header = msg->header;
		point_msg.point.x = point.x;
		point_msg.point.y = point.y;
		point_msg.point.z = point.z;

		points_in_camera_frame.push_back(point_msg);
	}

	masked_pc_gen(points_in_camera_frame);
}


// bool Matching_Pix_to_Ptcld::service_callback(me326_locobot_example::PixtoPoint::Request &req, me326_locobot_example::PixtoPoint::Response &res){
// 	// the topic for the rgb_img should be set as a rosparam when the file is launched (this can be done in the launch file, it is not done here since the subscriber is started with the class object instantiation)
// 	res.ptCld_point = point_3d_cloud_; //send the point back as a response	
// 	return true;
// }


int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Matching_Pix_to_Ptcld>());
  rclcpp::shutdown();
  return 0;
}