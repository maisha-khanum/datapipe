#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

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

private:
    std::string input_bag_path_;
    std::string output_bag_path_;
    rosbag2_cpp::Reader reader_;
    rosbag2_cpp::Writer writer_;
    int message_counter_;

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

        if (!run_gsam2(image_path, mask_path))
        {
            RCLCPP_ERROR(this->get_logger(), "GSAM2 segmentation failed.");
            return false;
        }

        mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load segmentation mask.");
            return false;
        }
        return true;
    }

    bool run_gsam2(const std::string &image_path, const std::string &mask_path)
    {
        std::string python_path = "/home/mkhanum/miniconda3/envs/GSAM2Env/bin/python";
        std::string script_path = "/home/mkhanum/datapipe/ros2_ws/src/smartbelt_segmentation/scripts/run_gsam2.py";
        std::string command = python_path + " " + script_path + " " + image_path + " " + mask_path;
        int ret_code = std::system(command.c_str());

        if (ret_code == 0)
        {
            RCLCPP_INFO(this->get_logger(), "GSAM2 segmentation finished successfully.");
            return true;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "GSAM2 segmentation failed with return code: %d", ret_code);
            return false;
        }
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

    rclcpp::shutdown();
    return 0;
}
