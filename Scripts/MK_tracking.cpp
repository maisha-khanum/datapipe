#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

const double fx = 431.0087890625;
const double fy = 431.0087890625;
const double ppx = 429.328704833984;
const double ppy = 242.162155151367;

// Function to load a depth image from a CSV file
cv::Mat loadDepthFromCSV(const std::string& filepath, int rows, int cols) {
    cv::Mat depth_image(rows, cols, CV_32F, cv::Scalar(0));
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open depth CSV: " << filepath << std::endl;
        return depth_image;
    }

    std::string line;
    int r = 0;
    while (std::getline(file, line) && r < rows) {
        std::stringstream ss(line);
        std::string value;
        int c = 0;
        while (std::getline(ss, value, ',') && c < cols) {
            depth_image.at<float>(r, c) = std::stof(value);
            c++;
        }
        r++;
    }
    file.close();
    return depth_image;
}

void depthToFilteredPointCloud(const cv::Mat& depth_image, const cv::Mat& mask, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    cloud->clear();

    // Compute scaling factors
    float scale_x = static_cast<float>(depth_image.cols) / mask.cols;
    float scale_y = static_cast<float>(depth_image.rows) / mask.rows;

    for (int v = 0; v < mask.rows; ++v) {
        for (int u = 0; u < mask.cols; ++u) {
            if (mask.at<uchar>(v, u) > 0) { // Only use pixels where mask is nonzero
                // Map segmentation mask coordinates to depth image coordinates
                int u_depth = static_cast<int>(u * scale_x);
                int v_depth = static_cast<int>(v * scale_y);

                // Ensure indices are within bounds
                if (u_depth >= 0 && u_depth < depth_image.cols && v_depth >= 0 && v_depth < depth_image.rows) {
                    float depth = depth_image.at<float>(v_depth, u_depth) * 10.0f; // Scale factor
                    if (depth > 0) {
                        float x = (u_depth - ppx) * depth / fx;
                        float y = (v_depth - ppy) * depth / fy;
                        float z = depth;
                        cloud->points.emplace_back(x, y, z);
                    }
                }
            }
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
}


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <depth_csv_folder> <mask_folder> <output_folder>" << std::endl;
        return -1;
    }

    std::string depth_folder = argv[1];
    std::string mask_folder = argv[2];
    std::string output_folder = argv[3];

    std::vector<std::string> depth_paths, mask_paths;
    
	// Load and sort depth and mask file paths
	for (const auto& entry : fs::directory_iterator(depth_folder)) {
		std::string filename = entry.path().filename().string();
		if (entry.path().extension() == ".csv" && filename.find("depth_") == 0) {
			// std::cout << "opening depth!" << std::endl;
			int frame_number = std::stoi(filename.substr(6, 3)); // Extract "000" from "depth_000.csv"
			if (frame_number >= 0) {
				depth_paths.push_back(entry.path().string());
			}
		}
	}

	for (const auto& entry : fs::directory_iterator(mask_folder)) {
		std::string filename = entry.path().filename().string();
		if (entry.path().extension() == ".png" && filename.find("seg_frame_color_") == 0) {
			size_t channel_pos = filename.find("_channel_");
			if (channel_pos != std::string::npos) {
				// std::cout << "getting 000 from seg!" << std::endl;
				int frame_number = std::stoi(filename.substr(16, 3)); // Extract "000" from "seg_frame_color_000_channel_0.png"
				// std::cout << "getting channel from seg!" << std::endl;
				std::string extracted_channel = filename.substr(channel_pos + 9);
				// std::cout << "Extracted channel substring: '" << extracted_channel << "'" << std::endl;
				int channel_number = std::stoi(extracted_channel);
				// int channel_number = std::stoi(filename.substr(channel_pos + 8, 1)); // Extract "0" from "seg_frame_color_000_channel_0.png"
				if (frame_number >= 0 && channel_number == 0) {
					mask_paths.push_back(entry.path().string());
				}
			}
		}
	}

	std::sort(depth_paths.begin(), depth_paths.end());
	std::sort(mask_paths.begin(), mask_paths.end());

	if (depth_paths.size() != mask_paths.size()) {
		std::cerr << "Mismatch between depth images and masks! "
				  << depth_paths.size() << " vs " << mask_paths.size() 
				  << std::endl;
		return -1;
	}
	

    for (size_t i = 0; i < depth_paths.size(); ++i) {
        std::cout << "Processing: " << depth_paths[i] << " with " << mask_paths[i] << std::endl;

        cv::Mat depth_image = loadDepthFromCSV(depth_paths[i], 480, 640); // Adjust size as needed
        cv::Mat mask_image = cv::imread(mask_paths[i], cv::IMREAD_GRAYSCALE);

        if (depth_image.empty() || mask_image.empty()) {
            std::cerr << "Skipping due to loading error: " << depth_paths[i] << " or " << mask_paths[i] << std::endl;
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        depthToFilteredPointCloud(depth_image, mask_image, cloud);

        char filename[50];
        sprintf(filename, "%s/filtered_%03d.ply", output_folder.c_str(), static_cast<int>(i));
        std::string output_ply(filename);

		if (cloud->empty()) {
			std::cerr << "Warning: Point cloud for " << depth_paths[i] << " is empty. Saving an empty PLY file." << std::endl;
		}
		
		// Attempt to save, even if the point cloud is empty
		if (pcl::io::savePLYFileBinary(output_ply, *cloud) == -1) {
			std::cerr << "Error: Could not save PLY file " << output_ply << std::endl;
		} else {
			std::cout << "Saved filtered point cloud to " << output_ply << " with " << cloud->points.size() << " points." << std::endl;
		}

        // std::cout << "Saved filtered point cloud to " << output_ply << " with " << cloud->points.size() << " points." << std::endl;
    }

    return 0;
}
