#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <tf2_msgs/msg/tf_message.hpp>
// #include <tf2_msgs/msg/tf_static_message.hpp>

class BagProcessor : public rclcpp::Node {
public:
    BagProcessor()
        : Node("bag_processor"), writer_(std::make_shared<rosbag2_cpp::Writer>()) {
        process_bags();
    }

private:
    std::shared_ptr<rosbag2_cpp::Writer> writer_;

    void process_bags() {
        rosbag2_cpp::Reader reader_mask, reader_cloud;
        reader_mask.open("/home/mkhanum/datapipe/Bags/stair1_full");
        reader_cloud.open("/home/mkhanum/datapipe/Bags/stair1");
        writer_->open("/home/mkhanum/datapipe/Bags/stair1_seg_pc");

        std::vector<sensor_msgs::msg::Image::SharedPtr> mask_msgs;
        std::vector<sensor_msgs::msg::PointCloud2::SharedPtr> cloud_msgs;

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
            // else if (bag_message->topic_name == "/tf") {
            //     writer_->write(*bag_message->serialized_data, bag_message->topic_name, rclcpp::Time(bag_message->time_stamp));
            //     RCLCPP_INFO(this->get_logger(), "Wrote TF data.");
            // }
        }

        // Print vector sizes
        RCLCPP_INFO(this->get_logger(), "Mask messages vector size: %zu", mask_msgs.size());
        RCLCPP_INFO(this->get_logger(), "Cloud messages vector size: %zu", cloud_msgs.size());

        // Match messages in pairs and print progress
        size_t num_pairs = std::min(mask_msgs.size(), cloud_msgs.size());
        RCLCPP_INFO(this->get_logger(), "Processing %zu pairs of messages.", num_pairs);

        for (size_t i = 0; i < num_pairs; ++i) {
            auto segmented_cloud = apply_mask(cloud_msgs[i], mask_msgs[i]);
            writer_->write(*segmented_cloud, "/segmented_pointcloud", cloud_msgs[i]->header.stamp);
            RCLCPP_INFO(this->get_logger(), "Processed pair %zu of %zu.", i + 1, num_pairs);
        }

        RCLCPP_INFO(this->get_logger(), "Finished processing all pairs.");
    }

    // sensor_msgs::msg::PointCloud2::SharedPtr apply_mask(
    //     const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
    //     const sensor_msgs::msg::Image::SharedPtr& mask) {
    //     // Convert mask to OpenCV image
    //     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(mask, "mono8");
    //     cv::Mat mask_mat = cv_ptr->image;

    //     std::vector<cv::Point> mask_pixels;
    //     for (int y = 0; y < mask_mat.rows; ++y) {
    //         for (int x = 0; x < mask_mat.cols; ++x) {
    //             if (mask_mat.at<uchar>(y, x) > 0) {
    //                 mask_pixels.emplace_back(x, y);
    //             }
    //         }
    //     }

        
    //     // Process point cloud using the mask (dummy implementation)
    //     // auto filtered_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>(*cloud);
    //     // Apply filtering logic based on mask_mat


    //     std::vector<geometry_msgs::msg::PointStamped> points_in_camera_frame; // TODO

    //     for (const auto& pixel : mask_pixels){ // TODO 
    //         int u = pixel.x;
    //         int v = pixel.y;
    
    //         int index = v * msg->width + u;
    //         if (index >= cloud->points.size()){
    //             RCLCPP_WARN(this->get_logger(), "Skipping pixel (%d, %d) - out of bounds", u, v);
    //             continue;
    //         }
    
    //         pcl::PointXYZ point = cloud->points[index];
    
    //         if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)){
    //             RCLCPP_WARN(this->get_logger(), "Skipping invalid point at pixel (%d, %d)", u, v);
    //             continue;
    //         }
    
    //         geometry_msgs::msg::PointStamped point_msg;
    //         point_msg.header = msg->header;
    //         point_msg.point.x = point.x;
    //         point_msg.point.y = point.y;
    //         point_msg.point.z = point.z;
    
    //         points_in_camera_frame.push_back(point_msg);
    //     }
        
    //     return filtered_cloud;
    // }


    sensor_msgs::msg::PointCloud2::SharedPtr apply_mask(
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud,
        const sensor_msgs::msg::Image::SharedPtr& mask) {
    
        // Convert mask to OpenCV format
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(mask, "mono8");
        cv::Mat mask_mat = cv_ptr->image;
    
        // Prepare a new filtered point cloud
        auto filtered_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
        filtered_cloud->header = cloud->header;
        filtered_cloud->height = 1; // Unorganized point cloud
        filtered_cloud->is_dense = false;
    
        // Copy point cloud metadata
        filtered_cloud->fields = cloud->fields;
        filtered_cloud->point_step = cloud->point_step;
        filtered_cloud->row_step = cloud->point_step; // Since height = 1
        filtered_cloud->is_bigendian = cloud->is_bigendian;
    
        // Use a vector to store valid points
        std::vector<uint8_t> filtered_data;
    
        // Iterate through the mask and extract valid points
        for (int v = 0; v < mask_mat.rows; ++v) {
            for (int u = 0; u < mask_mat.cols; ++u) {
                if (mask_mat.at<uchar>(v, u) > 0) { // If mask pixel is valid
                    int index = v * cloud->width + u;
                    if (index * cloud->point_step >= cloud->data.size()) {
                        // RCLCPP_WARN(rclcpp::get_logger("apply_mask"), 
                        //     "Skipping pixel (%d, %d) - out of bounds", u, v);
                        continue;
                    }
                    // Copy the corresponding point data
                    const uint8_t* src_ptr = &cloud->data[index * cloud->point_step];
                    filtered_data.insert(filtered_data.end(), src_ptr, src_ptr + cloud->point_step);
                }
            }
        }
    
        // Assign filtered data to the new cloud
        filtered_cloud->data = std::move(filtered_data);
        filtered_cloud->width = filtered_cloud->data.size() / filtered_cloud->point_step;
        filtered_cloud->row_step = filtered_cloud->width * filtered_cloud->point_step;
    
        return filtered_cloud;
    }
    
};


// void Matching_Pix_to_Ptcld::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
// 	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
// 	pcl::fromROSMsg(*msg, *cloud);

// 	std::vector<geometry_msgs::msg::PointStamped> points_in_camera_frame;

// 	for (const auto& pixel : mask_img_pixels_){ // TODO 
// 		int u = pixel.x;
// 		int v = pixel.y;

// 		int index = v * msg->width + u;
// 		if (index >= cloud->points.size()){
// 			RCLCPP_WARN(this->get_logger(), "Skipping pixel (%d, %d) - out of bounds", u, v);
// 			continue;
// 		}

// 		pcl::PointXYZ point = cloud->points[index];

// 		if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)){
// 			RCLCPP_WARN(this->get_logger(), "Skipping invalid point at pixel (%d, %d)", u, v);
// 			continue;
// 		}

// 		geometry_msgs::msg::PointStamped point_msg;
// 		point_msg.header = msg->header;
// 		point_msg.point.x = point.x;
// 		point_msg.point.y = point.y;
// 		point_msg.point.z = point.z;

// 		points_in_camera_frame.push_back(point_msg);
// 	}

// 	masked_pc_gen(points_in_camera_frame);
// }

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BagProcessor>());
    rclcpp::shutdown();
    return 0;
}