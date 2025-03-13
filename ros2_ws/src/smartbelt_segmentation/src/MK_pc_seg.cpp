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
        reader_mask.open("/home/mkhanum/datapipe/Bags/stair1_full");
        reader_cloud.open("/home/mkhanum/datapipe/Bags/stair1");
        writer_->open("/home/mkhanum/datapipe/Bags/stair1_seg_pc");

        std::vector<sensor_msgs::msg::Image::SharedPtr> mask_msgs;
        std::vector<sensor_msgs::msg::PointCloud2::SharedPtr> cloud_msgs;
        std::vector<sensor_msgs::msg::CameraInfo::SharedPtr> camera_info_msgs;

        // Read all mask messages
        while (reader_mask.has_next()) {
            auto bag_message = reader_mask.read_next();
            if (bag_message->topic_name == "segmented_mask") {
                auto img_msg = std::make_shared<sensor_msgs::msg::Image>();
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
                serializer.deserialize_message(&serialized_msg, img_msg.get());
                mask_msgs.push_back(img_msg);
            }
        }

        // Read all cloud messages
        while (reader_cloud.has_next()) {
            auto bag_message = reader_cloud.read_next();
            if (bag_message->topic_name == "/d455/depth/color/points") {
                auto cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serializer;
                serializer.deserialize_message(&serialized_msg, cloud_msg.get());
                cloud_msgs.push_back(cloud_msg);
            } 
            else if (bag_message->topic_name == "/tf_static") {
                auto tf_msg = std::make_shared<tf2_msgs::msg::TFMessage>();
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                rclcpp::Serialization<tf2_msgs::msg::TFMessage> serializer;
                serializer.deserialize_message(&serialized_msg, tf_msg.get());
                // rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepAll())
                            //   .transient_local()  // Ensure transforms persist
                            //   .reliable();

                writer_->write(*tf_msg, "/tf_static", rclcpp::Time(bag_message->time_stamp));

                RCLCPP_INFO(this->get_logger(), "Wrote TF_STATIC data.");
            } 
            else if (bag_message->topic_name == "/d455/depth/camera_info") {
                auto camera_info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                rclcpp::Serialization<sensor_msgs::msg::CameraInfo> serializer;
                serializer.deserialize_message(&serialized_msg, camera_info_msg.get());
                // camera_model_.fromCameraInfo(camera_info_msg);
                camera_info_msgs.push_back(camera_info_msg);
            }
        }

        // Use cloud_msg instead of cloud
        RCLCPP_INFO(this->get_logger(),
        "Header: frame_id=%s, stamp=%d.%d", 
        cloud_msgs[0]->header.frame_id.c_str(),
        cloud_msgs[0]->header.stamp.sec, cloud_msgs[0]->header.stamp.nanosec);
        
        RCLCPP_INFO(this->get_logger(),
            "Width: %d, Height: %d, is_dense: %d", 
            cloud_msgs[0]->width, cloud_msgs[0]->height, cloud_msgs[0]->is_dense);
        
        RCLCPP_INFO(this->get_logger(),
            "Point Step: %d, Row Step: %d, Data Size: %zu", 
            cloud_msgs[0]->point_step, cloud_msgs[0]->row_step, cloud_msgs[0]->data.size());
        
        RCLCPP_INFO(this->get_logger(),
            "Fields (%zu):", cloud_msgs[0]->fields.size());
        
        RCLCPP_INFO(this->get_logger(),
        "MASK Width: %d, Height: %d", 
        mask_msgs[0]->width, mask_msgs[0]->height);
        

            
        // Print vector sizes
        RCLCPP_INFO(this->get_logger(), "Mask messages vector size: %zu", mask_msgs.size());
        RCLCPP_INFO(this->get_logger(), "Cloud messages vector size: %zu", cloud_msgs.size());

        // Match messages in pairs and print progress
        size_t num_pairs = std::min(mask_msgs.size(), cloud_msgs.size());
        RCLCPP_INFO(this->get_logger(), "Processing %zu pairs of messages.", num_pairs);

        for (size_t i = 0; i < num_pairs; ++i) {
            camera_model_.fromCameraInfo(camera_info_msgs[i]);
            auto segmented_cloud = apply_mask(cloud_msgs[i], mask_msgs[i], camera_model_);
            writer_->write(*segmented_cloud, "/segmented_pointcloud", cloud_msgs[i]->header.stamp);
            RCLCPP_INFO(this->get_logger(), "Processed pair %zu of %zu.", i + 1, num_pairs);
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