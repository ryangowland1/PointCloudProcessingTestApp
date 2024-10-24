#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

// Global configuration
const std::string INPUT_FILE = "Data/point_cloud_PCD_23310350_1080_23-10-2024-16-19-14.pcd";  // Change this to your input file path
const float GREEN_THRESHOLD = 1.2f;          // Green should be 20% higher than red and blue

// Custom point struct to store XYZ and RGB values
struct Point {
    float x, y, z;
    uint8_t r, g, b;
};

void visualizeCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& original_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& filtered_cloud) {
    // Create a PCLVisualizer object
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Create two viewports
    int v1(0);
    int v2(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);

    // Add the point clouds to their respective viewports
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_original(original_cloud);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_filtered(filtered_cloud);

    viewer->addPointCloud<pcl::PointXYZRGB>(original_cloud, rgb_original, "original_cloud", v1);
    viewer->addPointCloud<pcl::PointXYZRGB>(filtered_cloud, rgb_filtered, "filtered_cloud", v2);

    // Add viewport labels
    viewer->addText("Original Cloud", 10, 10, "original_label", v1);
    viewer->addText("Filtered Cloud (Green Points)", 10, 10, "filtered_label", v2);

    // Set point size
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "filtered_cloud");

    // Camera settings for a nicer initial view
    viewer->initCameraParameters();
    viewer->resetCamera();

    std::cout << "Visualizing point clouds. Press 'q' to exit..." << std::endl;

    // Main visualization loop
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main() {
    // Load the PCD file
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(INPUT_FILE, *cloud) == -1) {
        std::cerr << "Couldn't read file " << INPUT_FILE << std::endl;
        return -1;
    }

    // Convert PCL cloud to vector of custom points
    std::vector<Point> points;
    for (const auto& pcl_point : cloud->points) {
        uint32_t rgb = *reinterpret_cast<const int*>(&pcl_point.rgb);
        Point p;
        p.x = pcl_point.x;
        p.y = pcl_point.y;
        p.z = pcl_point.z;
        p.r = (rgb >> 16) & 0xFF;
        p.g = (rgb >> 8) & 0xFF;
        p.b = rgb & 0xFF;
        points.push_back(p);
    }

    // Filter points based on green color dominance
    auto it = std::remove_if(points.begin(), points.end(),
        [](const Point& p) {
            return !(p.g > GREEN_THRESHOLD * p.r && p.g > GREEN_THRESHOLD * p.b);
        });
    points.erase(it, points.end());

    // Convert filtered points back to PCL cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    filtered_cloud->width = points.size();
    filtered_cloud->height = 1;
    filtered_cloud->points.resize(filtered_cloud->width * filtered_cloud->height);

    for (size_t i = 0; i < points.size(); ++i) {
        filtered_cloud->points[i].x = points[i].x;
        filtered_cloud->points[i].y = points[i].y;
        filtered_cloud->points[i].z = points[i].z;
        uint32_t rgb = ((uint32_t)points[i].r << 16 | (uint32_t)points[i].g << 8 | (uint32_t)points[i].b);
        filtered_cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);
    }

    // Save the filtered cloud
    pcl::io::savePCDFile("filtered_green.pcd", *filtered_cloud);

    std::cout << "Original points: " << cloud->size() << std::endl;
    std::cout << "Filtered points: " << points.size() << std::endl;
    std::cout << "Saved filtered cloud to 'filtered_green.pcd'" << std::endl;

    // Visualize the point clouds
    visualizeCloud(cloud, filtered_cloud);

    return 0;
}