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
// #include <image_geometry/pinhole_camera_model.h>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "sensor_msgs/image_encodings.hpp"

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
// #include <visualization_msgs/msg/marker.hpp>
// #include <visualization_msgs/msg/marker_array.hpp>

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
  // void info_callback(const sensor_msgs::msg::CameraInfo & msg);
  void info_callback(const std::shared_ptr<sensor_msgs::msg::CameraInfo> msg);
//   void depth_callback(const sensor_msgs::msg::Image & msg);
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr & msg);
  void color_image_callback(const sensor_msgs::msg::Image & msg);
  void service_callback(const std::shared_ptr<la_msgs::srv::Ptps::Request> req, std::shared_ptr<la_msgs::srv::Ptps::Response> res);
  void camera_cube_locator_marker_gen();
  bool blocks_of_specific_color_present(const std::shared_ptr<cv::Mat>);

  std::vector<geometry_msgs::msg::Point> blob_locator(std::shared_ptr<cv::Mat> & color_image_canvas_ptr,  
                                                      std::shared_ptr<cv::Mat> & mask_ptr);

  std::vector<geometry_msgs::msg::PointStamped> register_rgb_pix_to_depth_pts(const cv_bridge::CvImageConstPtr cv_ptr,
                                                                          std_msgs::msg::Header msg_header, 
                                                                          const std::shared_ptr<std::vector<geometry_msgs::msg::Point>> &uv_pix_list_ptr);


  private:
  
  rclcpp::QoS qos_; //message reliability

  // Publisher declarations
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_color_filt_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr camera_cube_locator_marker_;

  //Subscriber declaration
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_image_sub_;

  //Rosservice
  rclcpp::Service<la_msgs::srv::Ptps>::SharedPtr service_;



  //Variables
  geometry_msgs::msg::PointStamped point_3d_cloud_; //Point in pointcloud corresponding to desired pixel
  geometry_msgs::msg::Point uv_pix_; //pixel index

  std::vector<geometry_msgs::msg::PointStamped> red_3d_cloud_; //red block points list
  std::vector<geometry_msgs::msg::PointStamped> blue_3d_cloud_; //blue block points list
  std::vector<geometry_msgs::msg::PointStamped> yellow_3d_cloud_; //yellow block points list
  std::vector<geometry_msgs::msg::PointStamped> green_3d_cloud_; //green block points list


  std::string color_image_topic_; // topic for the color image
  std::string depth_image_topic_; // topic for the depth image
  std::string depth_img_camera_info_; // topic for the camera info
  std::string registered_pt_cld_topic_; // topic for the point cloud
  std::string desired_block_frame_; // what desired frame the blocks poses are expressed in
  
  image_geometry::PinholeCameraModel camera_model_; //Camera model, will help us with projecting the ray through the depth image
  bool depth_cam_info_ready_; //This will help us ensure we don't ask for a variable before its ready
  
  // TF Listener
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;


};

class Matching_Pix_to_Ptcld
{
public:
	Matching_Pix_to_Ptcld();

	// Make callback functions for subscribers
	void info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg);
	// void depth_callback(const sensor_msgs::Image::ConstPtr& msg);
	// void depth_callback(const sensor_msgs::Image::ConstPtr& msg); // TODO: POSSIBLE ISSUE, MAY NEED TO BE PC
	void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr& msg);
	void color_image_callback(const sensor_msgs::Image::ConstPtr& msg);
	// bool service_callback(me326_locobot_example::PixtoPoint::Request &req, me326_locobot_example::PixtoPoint::Response &res);
	void camera_cube_locator_marker_gen();
	void masked_pc_gen();
	void masked_img_gen();


  private:
	rclcpp::NodeHandle nh;

  // Publisher declarations
	rclcpp::Publisher image_color_filt_pub_;
	rclcpp::Publisher camera_cube_locator_marker_;
	// Subscriber declarations
	// rclcpp::Subscriber cam_info_sub_;
	rclcpp::Subscriber depth_sub_;
	rclcpp::Subscriber rgb_image_sub_;
	// Rosservice
	rclcpp::ServiceServer service_;
	//Variables
	geometry_msgs::PointStamped point_3d_cloud_; //Point in pointcloud corresponding to desired pixels
	// geometry_msgs::Point uv_pix_; //pixel index
	std::vector<geometry_msgs::Point> mask_pixels_;
	std::string color_image_topic_; // this string is over-written by the service request
	std::string depth_image_topic_; // this string is over-written by the service request
	std::string depth_img_camera_info_; // this string is over-written by the service request
	std::string registered_pt_cld_topic_; // this string is over-written by the service request
	image_geometry::PinholeCameraModel camera_model_; //Camera model, will help us with projecting the ray through the depth image
	bool depth_cam_info_ready_; //This will help us ensure we don't ask for a variable before its ready
	// TF Listener
	tf2_rclcpp::Buffer tf_buffer_;
	std::unique_ptr<tf2_rclcpp::TransformListener> tf_listener_;
};

Matching_Pix_to_Ptcld::Matching_Pix_to_Ptcld() 
{
	//Class constructor
	nh = rclcpp::NodeHandle(); //This argument makes all topics internal to this node namespace. //takes the node name (global node handle), If you use ~ then its private (under the node handle name) /armcontroller/ *param*
	//this is how to setup the TF buffer in a class:
	tf_listener_.reset(new tf2_rclcpp::TransformListener(tf_buffer_));
	//ROSparam set variables
	nh.param<std::string>("pt_srv_color_img_topic", color_image_topic_, "/d455/color/image_raw");
	nh.param<std::string>("pt_srv_reg_pt_cld_topic", registered_pt_cld_topic_, "/d455/depth/color/points");

	// nh.param<std::string>("pt_srv_depth_img_topic", depth_image_topic_, "/locobot/camera/aligned_depth_to_color/image_raw");
	// nh.param<std::string>("pt_srv_depth_img_cam_info_topic", depth_img_camera_info_, "/locobot/camera/aligned_depth_to_color/camera_info");

  // Publisher declarations
	masked_img_pub_ = nh.advertise<sensor_msgs::Image>("/processed_d455/image_gsam2_mask",1);
	masked_pc_filt_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/processed_d455/points_gsam2_mask",1);
	// camera_cube_locator_marker_ = nh.advertise<visualization_msgs::Marker>("/locobot/camera_cube_locator",1);
	// Subscriber declarations
	// cam_info_sub_ = nh.subscribe(depth_img_camera_info_,1,&Matching_Pix_to_Ptcld::info_callback,this);
	// depth_sub_ = nh.subscribe(depth_image_topic_,1,&Matching_Pix_to_Ptcld::depth_callback,this);
	pc_sub_ = nh.subscribe(depth_image_topic_,1,&Matching_Pix_to_Ptcld::depth_callback,this);
	rgb_image_sub_ = nh.subscribe(color_image_topic_,1,&Matching_Pix_to_Ptcld::color_image_callback,this);
	depth_cam_info_ready_ = false; //set this to false so that depth doesn't ask for camera_model_ until its been set
	//Service
	service_ = nh.advertiseService("pix_to_point_cpp", &Matching_Pix_to_Ptcld::service_callback, this);
}

void Matching_Pix_to_Ptcld::masked_img_gen(const cv::Mat& color_img, const cv::Mat& mask) {
	if (color_img.empty() || mask.empty()) {
		ROS_ERROR("Masked RGB generation failed: empty input.");
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
	cv_bridge_masked_image.header.stamp = rclcpp::Time::now();
	cv_bridge_masked_image.header.frame_id = "camera_frame";
	cv_bridge_masked_image.encoding = sensor_msgs::image_encodings::RGB8;
	cv_bridge_masked_image.image = masked_img;

	// Publish masked image
	sensor_msgs::Image ros_masked_image;
	cv_bridge_masked_image.toImageMsg(ros_masked_image);
	masked_img_pub_.publish(ros_masked_image);
}

void Matching_Pix_to_Ptcld::masked_pc_gen() {
	if (masked_points_.empty()) {
		ROS_WARN("No valid masked points to publish");
		return;
	}

	// Initialize point cloud message
	sensor_msgs::PointCloud2 pc;
	pc.header.stamp = rclcpp::Time::now();
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
		geometry_msgs::PointStamped transformed_point;
		try {
			tf_buffer_.transform(point, transformed_point, camera_model_.tfFrame());
		} catch (tf2::TransformException &ex) {
			ROS_ERROR("TF Transform Exception: %s", ex.what());
			continue;
		}

		*iter_x = transformed_point.point.x;
		*iter_y = transformed_point.point.y;
		*iter_z = transformed_point.point.z;
		++iter_x; ++iter_y; ++iter_z;
	}

	// Publish the masked point cloud
	masked_pc_filt_pub_.publish(pc);
}

void Matching_Pix_to_Ptcld::info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg){
	//create a camera model from the camera info
	camera_model_.fromCameraInfo(msg);
	depth_cam_info_ready_ = true;	
}

void Matching_Pix_to_Ptcld::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);  // Convert from ROS2 message to PCL format

    std::vector<geometry_msgs::msg::PointStamped> points_in_camera_frame;

    for (const auto& pixel : mask_pixels_) {  // Assume mask_pixels_ stores pixel coordinates (u, v)
        int u = pixel.x;  // Column index
        int v = pixel.y;  // Row index
        
        // Compute the index in the PointCloud2 data array
        int index = v * msg->width + u;
        if (index >= cloud->points.size()) {
            RCLCPP_WARN(this->get_logger(), "Skipping pixel (%d, %d) - out of bounds", u, v);
            continue;
        }

        pcl::PointXYZ point = cloud->points[index];

        // Ignore invalid points
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            RCLCPP_WARN(this->get_logger(), "Skipping invalid point at pixel (%d, %d)", u, v);
            continue;
        }

        // Convert to geometry_msgs::msg::PointStamped
        geometry_msgs::msg::PointStamped point_msg;
        point_msg.header = msg->header;
        point_msg.point.x = point.x;
        point_msg.point.y = point.y;
        point_msg.point.z = point.z;

        points_in_camera_frame.push_back(point_msg);
    }

    // Store processed points and generate the masked point cloud
    masked_points_ = points_in_camera_frame;
    masked_pc_gen();  // Generate and publish masked point cloud
}


// void Matching_Pix_to_Ptcld::depth_callback(const sensor_msgs::Image::ConstPtr& msg){
// 	//Take the depth message, using the 32FC1 encoding and define the depth pointer
// 	cv_bridge::CvImageConstPtr cv_ptr;
// 	try
// 	{
// 		cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
// 	}
// 	catch (cv_bridge::Exception& e)
// 	{
// 		ROS_ERROR("cv_bridge exception: %s", e.what());
// 		return;
// 	}

// 	//Access the pixel of interest
// 	cv::Mat depth_image = cv_ptr->image;
// 	std::vector<geometry_msgs::PointStamped> points_in_camera_frame;

// 	// Loop over all desired pixels in the mask
// 	for (const auto& pixel : mask_pixels_) {  // Assume mask_pixels_ is a list of pixels to process
//   		float depth_value = depth_image.at<float>(pixel.x,pixel.y);  // access the depth value of the desired pixel
// 		//If the pixel that was chosen has non-zero depth, then find the point projected along the ray at that depth value
// 		if (depth_value == 0)
// 		{
// 			ROS_WARN("Skipping cause pixel had no depth");
// 			return;
// 		}else{
// 			if (depth_cam_info_ready_)
// 			{
// 				//Pixel has depth, now we need to find the corresponding point in the pointcloud
// 				//Use the camera model to get the 3D ray for the current pixel
// 				cv::Point2d pixel_cv(pixel.y, pixel.x);
// 				cv::Point3d ray = camera_model_.projectPixelTo3dRay(pixel_cv);
// 				//Calculate the 3D point on the ray using the depth value
// 				cv::Point3d point_3d = ray*depth_value;	
				
				
// 				geometry_msgs::PointStamped point_3d_geom_msg; 
// 				point_3d_geom_msg.header = msg->header;
// 				point_3d_geom_msg.point.x = point_3d.x;
// 				point_3d_geom_msg.point.y = point_3d.y;
// 				point_3d_geom_msg.point.z = point_3d.z;

// 				points_in_camera_frame.push_back(point_msg);
// 			}
// 		}
// 		// Store the processed points and generate the masked point cloud
// 		masked_points_ = points_in_camera_frame; // Store points for external use
// 		masked_pc_gen();  // Generate and publish the masked point cloud
// 	}
// }

void Matching_Pix_to_Ptcld::color_image_callback(const sensor_msgs::Image::ConstPtr& msg){
	//convert sensor_msgs image to opencv image : http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages
	cv_bridge::CvImagePtr color_img_ptr;
	try
	{
	  color_img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8); //accesses image through color_img_ptr->image
	}
	catch (cv_bridge::Exception& e)
	{
	  ROS_ERROR("cv_bridge exception: %s", e.what());
	  return;
	}

	cv::Mat color_img = color_img_ptr->image;
	std::string image_path = "/tmp/frame.png";
	std::string mask_path = "/tmp/mask.png";

	// Save frame as PNG
	if (!cv::imwrite(image_path, color_img)) {
		ROS_ERROR("Failed to save image as PNG.");
		return;
	}

	// Run GSAM2 segmentation
	if (!run_gsam2(image_path, mask_path)) {
		ROS_ERROR("GSAM2 segmentation failed.");
		return;
	}

	// Load mask
	cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
	if (mask.empty()) {
		ROS_ERROR("Failed to load segmentation mask.");
		return;
	}

	// Process mask: Convert to list of pixel coordinates
	std::vector<cv::Point> mask_pixels = extract_mask_pixels(mask);

	// Print number of segmented pixels
	ROS_INFO("Extracted %lu mask pixels.", mask_pixels.size());
    

	//Now show the cube location spherical marker: 
	Matching_Pix_to_Ptcld::masked_img_gen(color_img, mask);
	
}


bool Matching_Pix_to_Ptcld::run_gsam2(const std::string& image_path, const std::string& mask_path) {
	std::string command = "python3 /home/mkhanum/Grounded-SAM-2/run_gsam2.py " + image_path + " " + mask_path;
	int ret_code = std::system(command.c_str());
	return (ret_code == 0);
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



bool Matching_Pix_to_Ptcld::service_callback(me326_locobot_example::PixtoPoint::Request &req, me326_locobot_example::PixtoPoint::Response &res){
	// the topic for the rgb_img should be set as a rosparam when the file is launched (this can be done in the launch file, it is not done here since the subscriber is started with the class object instantiation)
	res.ptCld_point = point_3d_cloud_; //send the point back as a response	
	return true;
}


int main(int argc, char **argv)
{
  rclcpp::init(argc,argv,"matching_ptcld_serv");
  rclcpp::NodeHandle nh("~");
  Matching_Pix_to_Ptcld ctd_obj;
  rclcpp::spin();
  return 0;
}