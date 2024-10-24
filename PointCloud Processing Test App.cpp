#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

// Global configuration
const std::string INPUT_FILE = "Data/point_cloud_PCD_23310350_1080_23-10-2024-16-19-14.pcd";
const float GREEN_THRESHOLD = 1.2f;
const float PLANE_DISTANCE_THRESHOLD = 0.02f; // Maximum distance from point to plane for inliers
const int STATISTICAL_NEIGHBORS = 50;         // Number of neighbors to analyze for statistical filtering
const float STANDARD_DEV_MULTIPLIER = 2.0f;   // Standard deviation threshold multiplier

struct Point {
    float x, y, z;
    uint8_t r, g, b;
};

pcl::PointCloud<pcl::PointXYZRGB>::Ptr fitPlaneAndFilterOutliers(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {

    // First, fit a plane using RANSAC
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(PLANE_DISTANCE_THRESHOLD);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
        return cloud;
    }

    // Create a new cloud for the inliers
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& idx : inliers->indices) {
        plane_cloud->points.push_back(cloud->points[idx]);
    }
    plane_cloud->width = plane_cloud->points.size();
    plane_cloud->height = 1;

    // Apply statistical outlier removal
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    sor.setInputCloud(plane_cloud);
    sor.setMeanK(STATISTICAL_NEIGHBORS);
    sor.setStddevMulThresh(STANDARD_DEV_MULTIPLIER);
    sor.filter(*filtered_cloud);

    std::cout << "Plane coefficients: " << coefficients->values[0] << " "
        << coefficients->values[1] << " "
        << coefficients->values[2] << " "
        << coefficients->values[3] << std::endl;
    std::cout << "Original points: " << cloud->size() << std::endl;
    std::cout << "Points after plane fitting: " << plane_cloud->size() << std::endl;
    std::cout << "Points after statistical filtering: " << filtered_cloud->size() << std::endl;

    return filtered_cloud;
}

void visualizeCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& original_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& green_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& plane_cloud) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Create three viewports
    int v1(0), v2(1), v3(2);
    viewer->createViewPort(0.0, 0.0, 0.33, 1.0, v1);
    viewer->createViewPort(0.33, 0.0, 0.66, 1.0, v2);
    viewer->createViewPort(0.66, 0.0, 1.0, 1.0, v3);

    // Add the point clouds to their respective viewports
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_original(original_cloud);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_green(green_cloud);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_plane(plane_cloud);

    viewer->addPointCloud<pcl::PointXYZRGB>(original_cloud, rgb_original, "original_cloud", v1);
    viewer->addPointCloud<pcl::PointXYZRGB>(green_cloud, rgb_green, "green_cloud", v2);
    viewer->addPointCloud<pcl::PointXYZRGB>(plane_cloud, rgb_plane, "plane_cloud", v3);

    // Add viewport labels
    viewer->addText("Original Cloud", 10, 10, "original_label", v1);
    viewer->addText("Green Points", 10, 10, "green_label", v2);
    viewer->addText("Plane-Fitted Points", 10, 10, "plane_label", v3);

    // Set point size
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "green_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "plane_cloud");

    viewer->initCameraParameters();
    viewer->resetCamera();

    std::cout << "Visualizing point clouds. Press 'q' to exit..." << std::endl;

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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr green_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    green_cloud->width = points.size();
    green_cloud->height = 1;
    green_cloud->points.resize(green_cloud->width * green_cloud->height);

    for (size_t i = 0; i < points.size(); ++i) {
        green_cloud->points[i].x = points[i].x;
        green_cloud->points[i].y = points[i].y;
        green_cloud->points[i].z = points[i].z;
        uint32_t rgb = ((uint32_t)points[i].r << 16 | (uint32_t)points[i].g << 8 | (uint32_t)points[i].b);
        green_cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);
    }

    // Fit plane and remove outliers
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_filtered_cloud =
        fitPlaneAndFilterOutliers(green_cloud);

    // Save the filtered cloud
    pcl::io::savePCDFile("filtered_green_plane.pcd", *plane_filtered_cloud);

    // Visualize all stages
    visualizeCloud(cloud, green_cloud, plane_filtered_cloud);

    return 0;
}