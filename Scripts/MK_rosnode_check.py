#include <ros/ros.h>
#include <ros/package.h>

/*
//This block of Eigen functions aren't required in this script, 
but I personally include this on most applications so I have easy access 
to matrix functionatliy when needed (similar to python numpy). 
*/
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <eigen_conversions/eigen_msg.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <map> //dictionary equivalent
#include<std_msgs/Header.h>

#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>

//include the service for this package
#include <me326_locobot_example/PixtoPoint.h>


//Setup the class:
class Matching_Pix_to_Ptcld
{
public:
	Matching_Pix_to_Ptcld();

	// Make callback functions for subscribers
	void info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg);
	void depth_callback(const sensor_msgs::Image::ConstPtr& msg);
	void color_image_callback(const sensor_msgs::Image::ConstPtr& msg);
	bool service_callback(me326_locobot_example::PixtoPoint::Request &req, me326_locobot_example::PixtoPoint::Response &res);
	void camera_cube_locator_marker_gen();


  private:
	ros::NodeHandle nh;

  // Publisher declarations
	ros::Publisher image_color_filt_pub_;
	ros::Publisher camera_cube_locator_marker_;
	// Subscriber declarations
	ros::Subscriber cam_info_sub_;
	ros::Subscriber depth_sub_;
	ros::Subscriber rgb_image_sub_;
	// Rosservice
	ros::ServiceServer service_;
	//Variables
	geometry_msgs::PointStamped point_3d_cloud_; //Point in pointcloud corresponding to desired pixel
	geometry_msgs::Point uv_pix_; //pixel index
	std::string color_image_topic_; // this string is over-written by the service request
	std::string depth_image_topic_; // this string is over-written by the service request
	std::string depth_img_camera_info_; // this string is over-written by the service request
	std::string registered_pt_cld_topic_; // this string is over-written by the service request
	image_geometry::PinholeCameraModel camera_model_; //Camera model, will help us with projecting the ray through the depth image
	bool depth_cam_info_ready_; //This will help us ensure we don't ask for a variable before its ready
	// TF Listener
	tf2_ros::Buffer tf_buffer_;
	std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
};

Matching_Pix_to_Ptcld::Matching_Pix_to_Ptcld() 
{
	//Class constructor
	nh = ros::NodeHandle(); //This argument makes all topics internal to this node namespace. //takes the node name (global node handle), If you use ~ then its private (under the node handle name) /armcontroller/ *param*
	//this is how to setup the TF buffer in a class:
	tf_listener_.reset(new tf2_ros::TransformListener(tf_buffer_));
	//ROSparam set variables
	nh.param<std::string>("pt_srv_color_img_topic", color_image_topic_, "/camera/camera/color/image_raw");
	nh.param<std::string>("pt_srv_depth_img_topic", depth_image_topic_, "/locobot/camera/aligned_depth_to_color/image_raw");
	nh.param<std::string>("pt_srv_depth_img_cam_info_topic", depth_img_camera_info_, "/locobot/camera/aligned_depth_to_color/camera_info");
	nh.param<std::string>("pt_srv_reg_pt_cld_topic", registered_pt_cld_topic_, "/locobot/camera/depth_registered/points");

    // Publisher declarations //to change
	image_color_filt_pub_ = nh.advertise<sensor_msgs::Image>("/locobot/camera/block_color_filt_img",1);
	camera_cube_locator_marker_ = nh.advertise<visualization_msgs::Marker>("/locobot/camera_cube_locator",1);

	// Subscriber declarations
	cam_info_sub_ = nh.subscribe(depth_img_camera_info_,1,&Matching_Pix_to_Ptcld::info_callback,this);
	depth_sub_ = nh.subscribe(depth_image_topic_,1,&Matching_Pix_to_Ptcld::depth_callback,this);
	# rgb_image_sub_ = nh.subscribe(color_image_topic_,1,&Matching_Pix_to_Ptcld::color_image_callback,this);
	depth_cam_info_ready_ = false; //set this to false so that depth doesn't ask for camera_model_ until its been set
	//Service
	service_ = nh.advertiseService("pix_to_point_cpp", &Matching_Pix_to_Ptcld::service_callback, this);
}

void Matching_Pix_to_Ptcld::camera_cube_locator_marker_gen(){
	visualization_msgs::Marker marker;
	marker.header.frame_id = point_3d_cloud_.header.frame_id;
	marker.header.stamp = ros::Time::now();
	marker.id = 0;
	marker.type = visualization_msgs::Marker::SPHERE;
	// Set the marker scale
	marker.scale.x = 0.05;  //radius of the sphere
	marker.scale.y = 0.05;
	marker.scale.z = 0.05;
	// Set the marker pose
	marker.pose.position.x = point_3d_cloud_.point.x;
	marker.pose.position.y = point_3d_cloud_.point.y;
	marker.pose.position.z = point_3d_cloud_.point.z;
	// Set the marker color
	marker.color.a = 1.0; //transparency
	marker.color.r = 1.0; //red
	marker.color.g = 0.0;
	marker.color.b = 0.0;
	// Publish the marker
	camera_cube_locator_marker_.publish(marker);
}

void Matching_Pix_to_Ptcld::info_callback(const sensor_msgs::CameraInfo::ConstPtr& msg){
	//create a camera model from the camera info
	camera_model_.fromCameraInfo(msg);
	depth_cam_info_ready_ = true;	
}

void Matching_Pix_to_Ptcld::depth_callback(const sensor_msgs::Image::ConstPtr& msg){
	//Take the depth message, using teh 32FC1 encoding and define the depth pointer
	 cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  //Access the pixel of interest
  cv::Mat depth_image = cv_ptr->image;
  float depth_value = depth_image.at<float>(uv_pix_.x,uv_pix_.y);  // access the depth value of the desired pixel
  //If the pixel that was chosen has non-zero depth, then find the point projected along the ray at that depth value
	if (depth_value == 0)
	{
		ROS_WARN("Skipping cause pixel had no depth");
		return;
	}else{
		if (depth_cam_info_ready_)
		{
			//Pixel has depth, now we need to find the corresponding point in the pointcloud
			//Use the camera model to get the 3D ray for the current pixel
			cv::Point2d pixel(uv_pix_.y, uv_pix_.x);
			cv::Point3d ray = camera_model_.projectPixelTo3dRay(pixel);
			//Calculate the 3D point on the ray using the depth value
			cv::Point3d point_3d = ray*depth_value;		
			geometry_msgs::PointStamped point_3d_geom_msg; 
			point_3d_geom_msg.header = msg->header;
			point_3d_geom_msg.point.x = point_3d.x;
			point_3d_geom_msg.point.y = point_3d.y;
			point_3d_geom_msg.point.z = point_3d.z;
			//Transform the point to the pointcloud frame using tf
			std::string point_cloud_frame = camera_model_.tfFrame();
			// Get the camera pose in the desired reference frame
			geometry_msgs::TransformStamped transform;
			try {
			    transform = tf_buffer_.lookupTransform(point_cloud_frame, msg->header.frame_id, ros::Time(0));
			} catch (tf2::TransformException &ex) {
			    ROS_ERROR("%s", ex.what());
			}
			// Transform a point cloud point
			tf2::doTransform(point_3d_geom_msg, point_3d_cloud_, transform); // syntax: (points_in, points_out, transform)
		}
	}
}

int main(int argc, char **argv)
{
  ros::init(argc,argv,"matching_ptcld_serv");
  ros::NodeHandle nh("~");
  Matching_Pix_to_Ptcld ctd_obj;
  ros::spin();
  return 0;
}