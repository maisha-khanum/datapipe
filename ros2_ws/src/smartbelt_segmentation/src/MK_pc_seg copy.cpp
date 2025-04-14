#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <tf2_msgs/msg/tf_message.hpp>
#include <image_geometry/pinhole_camera_model.h>

// #include <tf2_msgs/msg/tf_static_message.hpp>

class BagProcessor : public rclcpp::Node {
public:
    BagProcessor()
        : Node("bag_processor"), writer_(std::make_shared<rosbag2_cpp::Writer>()) {
        process_bags();
    }

private:
    std::shared_ptr<rosbag2_cpp::Writer> writer_;
    image_geometry::PinholeCameraModel camera_model_;

    void process_bags() {
        rosbag2_cpp::Reader reader_mask, reader_cloud;
        // reader_mask.open("/home/mkhanum/datapipe/Bags/stair2_masked");
        // reader_cloud.open("/mnt/c/Users/mkhan/Downloads/realsense_ros2A");
        // writer_->open("/home/mkhanum/datapipe/Bags/stair2_seg_pc");

        reader_cloud.open("/home/mkhanum/datapipe/Bags/stair1");
        reader_mask.open("/home/mkhanum/datapipe/Bags/stair1_masked");
        writer_->open("/home/mkhanum/datapipe/Bags/stair1_seg_pc_v2");
    
        while (reader_mask.has_next() && reader_cloud.has_next()) {
            auto mask_bag_message = reader_mask.read_next();
            auto cloud_bag_message = reader_cloud.read_next();
    
            if (mask_bag_message->topic_name == "segmented_mask" && cloud_bag_message->topic_name == "/d455/depth/color/points") {
                auto mask_msg = std::make_shared<sensor_msgs::msg::Image>();
                auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
                auto camera_info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
    
                rclcpp::SerializedMessage mask_serialized_msg(*mask_bag_message->serialized_data);
                rclcpp::SerializedMessage cloud_serialized_msg(*cloud_bag_message->serialized_data);
    
                rclcpp::Serialization<sensor_msgs::msg::Image> mask_serializer;
                rclcpp::Serialization<sensor_msgs::msg::PointCloud2> cloud_serializer;
    
                mask_serializer.deserialize_message(&mask_serialized_msg, mask_msg.get());
                cloud_serializer.deserialize_message(&cloud_serialized_msg, cloud_msg.get());
    
                //get camera info message.
                rosbag2_cpp::Reader camera_info_reader;
                // camera_info_reader.open("/mnt/c/Users/mkhan/Downloads/realsense_ros2A");
                camera_info_reader.open("/home/mkhanum/datapipe/Bags/stair1");
                while(camera_info_reader.has_next()){
                    auto camera_info_bag_message = camera_info_reader.read_next();
                    if(camera_info_bag_message->topic_name == "/d455/depth/camera_info" && camera_info_bag_message->time_stamp == cloud_bag_message->time_stamp){
                        rclcpp::SerializedMessage camera_info_serialized_msg(*camera_info_bag_message->serialized_data);
                        rclcpp::Serialization<sensor_msgs::msg::CameraInfo> camera_info_serializer;
                        camera_info_serializer.deserialize_message(&camera_info_serialized_msg, camera_info_msg.get());
                        break;
                    }
                }
                camera_info_reader.close();
    
                camera_model_.fromCameraInfo(camera_info_msg);
                auto segmented_cloud = apply_mask(cloud_msg, mask_msg, camera_model_);
                writer_->write(*segmented_cloud, "/segmented_pointcloud", cloud_msg->header.stamp);
                RCLCPP_INFO(this->get_logger(), "Processed a pair of messages.");
            }
            else if(cloud_bag_message->topic_name == "/tf_static"){
                auto tf_msg = std::make_shared<tf2_msgs::msg::TFMessage>();
                rclcpp::SerializedMessage serialized_msg(*cloud_bag_message->serialized_data);
                rclcpp::Serialization<tf2_msgs::msg::TFMessage> serializer;
                serializer.deserialize_message(&serialized_msg, tf_msg.get());
                writer_->write(*tf_msg, "/tf_static", rclcpp::Time(cloud_bag_message->time_stamp));
                RCLCPP_INFO(this->get_logger(), "Wrote TF_STATIC data.");
            }
        }
        RCLCPP_INFO(this->get_logger(), "Finished processing all pairs.");
    }

    sensor_msgs::msg::PointCloud2::SharedPtr apply_mask(
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
        const sensor_msgs::msg::Image::SharedPtr& mask,
        const image_geometry::PinholeCameraModel& camera_model) { // Pass the camera model
    
        // Convert mask to OpenCV format
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(mask, "mono8");
        cv::Mat mask_mat = cv_ptr->image;
    
        auto filtered_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>(*cloud);
        filtered_cloud->data.clear();
        filtered_cloud->width = 0;
    
        // Get camera intrinsic parameters from the camera model
        cv::Mat camera_matrix = cv::Mat(camera_model.intrinsicMatrix());
        cv::Mat dist_coeffs = cv::Mat(camera_model.distortionCoeffs());
    
        for (size_t i = 0; i < cloud->width; ++i) {
            const uint8_t* point_data = &cloud->data[i * cloud->point_step];
    
            // Extract x, y, z from point_data (adjust based on your point cloud format)
            float x, y, z;
            memcpy(&x, point_data + 0 * sizeof(float), sizeof(float)); // Adjust offsets as needed
            memcpy(&y, point_data + 1 * sizeof(float), sizeof(float));
            memcpy(&z, point_data + 2 * sizeof(float), sizeof(float));
    
            cv::Point3f point3d(x, y, z);
    
            std::vector<cv::Point2f> projected_points;
            cv::projectPoints(std::vector<cv::Point3f>{point3d}, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F), camera_matrix, dist_coeffs, projected_points);
    
            if (!projected_points.empty()) {
                cv::Point2f projected_point = projected_points[0];
                int u = static_cast<int>(projected_point.x);
                int v = static_cast<int>(projected_point.y);
    
                if (u >= 0 && u < mask_mat.cols && v >= 0 && v < mask_mat.rows) {
                    if (mask_mat.at<uchar>(v, u) > 0) {
                        filtered_cloud->data.insert(filtered_cloud->data.end(), point_data, point_data + cloud->point_step);
                        filtered_cloud->width++;
                    }
                }
            }
        }
    
        filtered_cloud->row_step = filtered_cloud->width * cloud->point_step;
    
        return filtered_cloud;
    }
    
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BagProcessor>());
    rclcpp::shutdown();
    return 0;
}