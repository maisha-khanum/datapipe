#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <fstream>
#include <chrono>

class ROSBagProcessor : public rclcpp::Node
{
public:
    ROSBagProcessor(const std::string &input_bag, const std::string &output_bag)
        : Node("rosbag_processor"), input_bag_path_(input_bag), output_bag_path_(output_bag), message_counter_(0)
    {
        reader_.open(rosbag2_storage::StorageOptions{input_bag, "sqlite3"}, rosbag2_cpp::ConverterOptions{});
        writer_.open(rosbag2_storage::StorageOptions{output_bag, "sqlite3"}, rosbag2_cpp::ConverterOptions{});
        writer_.create_topic({"segmented_mask", "sensor_msgs/msg/Image", "cdr"});
    }

    void process()
    {
        while (reader_.has_next())
        {
            auto bag_message = reader_.read_next();
            std::string topic_name = bag_message->topic_name;

            if (topic_name == "/d455/color/image_raw")
            {
                auto img_msg = std::make_shared<sensor_msgs::msg::Image>();
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
                serializer.deserialize_message(&serialized_msg, img_msg.get());

                // Convert ROS image to OpenCV
                cv_bridge::CvImagePtr cv_ptr;
                try
                {
                    cv_ptr = cv_bridge::toCvCopy(*img_msg, "bgr8");
                }
                catch (cv_bridge::Exception &e)
                {
                    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                    continue;
                }

                // Process Image using GSAM2
                cv::Mat mask;
                if (!process_with_gsam2(cv_ptr->image, mask))
                {
                    RCLCPP_ERROR(this->get_logger(), "Skipping this image due to GSAM2 failure.");
                    continue;
                }

                // Convert mask to ROS Image
                cv_bridge::CvImage out_cv_img;
                out_cv_img.header = img_msg->header;
                out_cv_img.encoding = "mono8";
                out_cv_img.image = mask;
                auto mask_msg = out_cv_img.toImageMsg();


                // Serialize and write to output bag
                rclcpp::SerializedMessage serialized_output;
                rclcpp::Serialization<sensor_msgs::msg::Image> output_serializer;
                output_serializer.serialize_message(mask_msg.get(), &serialized_output);

                // Create a rosbag2_storage::SerializedBagMessage
                auto output_bag_message = std::make_shared<rosbag2_storage::SerializedBagMessage>();
                output_bag_message->topic_name = "segmented_mask";
                output_bag_message->time_stamp = bag_message->time_stamp;
                output_bag_message->serialized_data = std::make_shared<rcutils_uint8_array_t>();
                output_bag_message->serialized_data->buffer = serialized_output.get_rcl_serialized_message().buffer;
                output_bag_message->serialized_data->buffer_length = serialized_output.get_rcl_serialized_message().buffer_length;
                output_bag_message->serialized_data->buffer_capacity = serialized_output.get_rcl_serialized_message().buffer_capacity;

                // Write to output bag
                writer_.write(output_bag_message);
            }
        }
        RCLCPP_INFO(this->get_logger(), "Processing complete. Output saved in: %s", output_bag_path_.c_str());
    }

    void save_data_to_file(const std::string& filename) {
        std::ofstream outfile(filename);
        if (outfile.is_open()) {
            outfile << "Message Counter,Duration (ms),Empty Mask\n";
            for (const auto& point : data) {
                outfile << point.message_counter << "," << point.duration << "," << point.is_empty_mask << "\n";
            }
            outfile.close();
            RCLCPP_INFO(this->get_logger(), "Data saved to %s", filename.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file for writing: %s", filename.c_str());
        }
    }

private:
    std::string input_bag_path_;
    std::string output_bag_path_;
    rosbag2_cpp::Reader reader_;
    rosbag2_cpp::Writer writer_;
    int message_counter_;

    struct DataPoint {
        int message_counter;
        long long duration;
        bool is_empty_mask;
    };
    std::vector<DataPoint> data;

    bool process_with_gsam2(const cv::Mat &color_img, cv::Mat &mask)
    {
        std::string image_path = "/tmp/frame.png";
        std::string mask_path = "/tmp/mask.png";

        if (!cv::imwrite(image_path, color_img))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to save image as PNG.");
            return false;
        }

        // Log the message number being processed
        message_counter_++;
        RCLCPP_INFO(this->get_logger(), "Processing message number: %d", message_counter_);

        
        // Time measurement start
        auto start_time = std::chrono::high_resolution_clock::now();
        if (!run_gsam2(image_path, mask_path))
        {
            RCLCPP_ERROR(this->get_logger(), "GSAM2 segmentation failed.");
            return false;
        }
        // Time measurement end
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load segmentation mask.");
            return false;
        }

        bool is_empty_mask = true;
        if (!mask.empty()) {
          is_empty_mask = cv::countNonZero(mask) == 0;
        }
    
        // Store data for plotting
        data.push_back({message_counter_, duration.count(), is_empty_mask});
    
        return true;
    }

    bool run_gsam2(const std::string& image_path, const std::string& mask_path) {
        CURL *curl;
        CURLcode res;
    
        // JSON payload
        nlohmann::json json_data;
        json_data["image_path"] = image_path;
        json_data["mask_path"] = mask_path;
    
        std::string json_str = json_data.dump();
    
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
    
        curl = curl_easy_init();
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:5000/run_gsam2");
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_str.size());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            
            res = curl_easy_perform(curl);
            
            // Clean up
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
    
            return res == CURLE_OK;
        }
        return false;
    }
    
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    if (argc < 3)
    {
        std::cerr << "Usage: ros2 run <package> rosbag_processor <input_bag> <output_bag>" << std::endl;
        return 1;
    }

    std::string input_bag = argv[1];
    std::string output_bag = argv[2];

    auto processor = std::make_shared<ROSBagProcessor>(input_bag, output_bag);
    processor->process();
    processor->save_data_to_file("gsam2_timing_data.csv");

    rclcpp::shutdown();
    return 0;
}
