#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <Eigen/Dense>

// Global configuration
const std::string INPUT_FILE = "Data/point_cloud_PCD_23310350_1080_23-10-2024-16-19-14.pcd";
const float GREEN_THRESHOLD = 1.2f;
const float PLANE_DISTANCE_THRESHOLD = 0.02f;
const float INITIAL_LINE_DISTANCE_THRESHOLD = 0.01f;
const int STATISTICAL_NEIGHBORS = 50;
const float STANDARD_DEV_MULTIPLIER = 2.0f;
const int MAX_LINES = 5;
const float POINT_PROPORTION_THRESHOLD = 0.1f;

struct Point {
    float x, y, z;
    uint8_t r, g, b;
};

struct LineSegment {
    Eigen::Vector3f start;
    Eigen::Vector3f end;
    float length;
};

// Function to fit a plane and filter outliers
pcl::PointCloud<pcl::PointXYZRGB>::Ptr fitPlaneAndFilterOutliers(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

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

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& idx : inliers->indices) {
        plane_cloud->points.push_back(cloud->points[idx]);
    }
    plane_cloud->width = plane_cloud->points.size();
    plane_cloud->height = 1;

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    sor.setInputCloud(plane_cloud);
    sor.setMeanK(STATISTICAL_NEIGHBORS);
    sor.setStddevMulThresh(STANDARD_DEV_MULTIPLIER);
    sor.filter(*filtered_cloud);

    return filtered_cloud;
}

// Function to fit multiple lines on the plane-filtered cloud
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> fitMultipleLines(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud) {

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> line_clouds;
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(plane_cloud);

    float line_distance_threshold = INITIAL_LINE_DISTANCE_THRESHOLD;
    int detected_lines = 0;
    int total_points = plane_cloud->width;

    for (int i = 0; i < MAX_LINES; ++i) {
        pcl::ModelCoefficients::Ptr line_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_LINE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(line_distance_threshold);
        seg.setInputCloud(plane_cloud);
        seg.segment(*inliers, *line_coefficients);

        if (inliers->indices.size() < POINT_PROPORTION_THRESHOLD * total_points) {
            std::cerr << "No more lines can be fitted on the plane after " << detected_lines << " lines." << std::endl;
            break;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr line_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*line_cloud);

        line_clouds.push_back(line_cloud);
        detected_lines++;

        extract.setNegative(true);
        extract.filter(*plane_cloud);

        line_distance_threshold *= 1.1;
    }

    return line_clouds;
}

// Function to fit a thin line segment to a point cloud
LineSegment fitLineSegment(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    Eigen::Vector3f direction(coefficients->values[3], coefficients->values[4], coefficients->values[5]);
    direction.normalize();

    Eigen::Vector3f point(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    float min_t = std::numeric_limits<float>::max();
    float max_t = std::numeric_limits<float>::lowest();

    for (const auto& pcl_point : cloud->points) {
        Eigen::Vector3f p(pcl_point.x, pcl_point.y, pcl_point.z);
        float t = direction.dot(p - point);
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }

    LineSegment segment;
    segment.start = point + direction * min_t;
    segment.end = point + direction * max_t;
    segment.length = (segment.end - segment.start).norm();

    return segment;
}

// Function to create rectangular prism between two line segments
void addPrismBetweenLines(
    pcl::visualization::PCLVisualizer::Ptr& viewer,
    const LineSegment& line1,
    const LineSegment& line2,
    const std::string& name,
    int viewport) {

    // Get midpoints of each line
    Eigen::Vector3f mid1 = (line1.start + line1.end) * 0.5f;
    Eigen::Vector3f mid2 = (line2.start + line2.end) * 0.5f;

    // Calculate average direction of the two lines for length
    Eigen::Vector3f dir1 = (line1.end - line1.start).normalized();
    Eigen::Vector3f dir2 = (line2.end - line2.start).normalized();

    // Make sure dirs are pointing in similar directions
    if (dir1.dot(dir2) < 0) {
        dir2 = -dir2;
    }

    Eigen::Vector3f avg_dir = (dir1 + dir2).normalized();

    // Width direction is perpendicular to average direction in horizontal plane
    Eigen::Vector3f width_dir = (mid2 - mid1);
    // Remove component parallel to average direction
    width_dir = width_dir - width_dir.dot(avg_dir) * avg_dir;
    width_dir.normalize();

    // Height direction is up (Z)
    Eigen::Vector3f up(0, 0, 1);

    // Calculate dimensions
    float width = (mid2 - mid1).norm() * 0.75f;  // 75% of distance between lines
    float height = width * 0.8f;  // 80% of width
    float length = (line1.length + line2.length) * 0.375f;  // 75% of average length

    // Calculate center point
    Eigen::Vector3f center = (mid1 + mid2) * 0.5f;

    // Calculate corners
    std::vector<Eigen::Vector3f> corners;
    for (int i = 0; i < 8; ++i) {
        Eigen::Vector3f corner = center;

        // Add/subtract half dimensions in appropriate directions
        if (i & 1) corner += width_dir * (width * 0.5f);
        else corner -= width_dir * (width * 0.5f);

        if (i & 2) corner += avg_dir * (length * 0.5f);
        else corner -= avg_dir * (length * 0.5f);

        if (i & 4) corner += up * (height * 0.5f);
        else corner -= up * (height * 0.5f);

        corners.push_back(corner);
    }

    // Find bounding box
    float x_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest();
    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();

    for (const auto& corner : corners) {
        x_min = std::min(x_min, corner.x());
        x_max = std::max(x_max, corner.x());
        y_min = std::min(y_min, corner.y());
        y_max = std::max(y_max, corner.y());
        z_min = std::min(z_min, corner.z());
        z_max = std::max(z_max, corner.z());
    }

    viewer->addCube(x_min, x_max, y_min, y_max, z_min, z_max, 0.8, 0.8, 0.8, name, viewport);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.7, name);
}

std::vector<std::pair<size_t, size_t>> findClosestLinePairs(const std::vector<LineSegment>& line_segments) {
    std::vector<std::pair<size_t, size_t>> pairs;
    if (line_segments.size() < 2) return pairs;

    // Constants for filtering
    const float MAX_ANGLE_DIFF = 0.3f;  // Maximum angle difference in radians (~17 degrees)
    const float MAX_DIST_MULTIPLIER = 1.5f;  // Maximum distance multiplier compared to closest pair

    // First find the minimum distance between any valid pair (lines that are nearly parallel)
    float min_valid_dist = std::numeric_limits<float>::max();

    for (size_t i = 0; i < line_segments.size(); ++i) {
        Eigen::Vector3f dir_i = (line_segments[i].end - line_segments[i].start).normalized();
        Eigen::Vector3f mid_i = (line_segments[i].start + line_segments[i].end) * 0.5f;

        for (size_t j = i + 1; j < line_segments.size(); ++j) {
            Eigen::Vector3f dir_j = (line_segments[j].end - line_segments[j].start).normalized();
            Eigen::Vector3f mid_j = (line_segments[j].start + line_segments[j].end) * 0.5f;

            // Check if lines are nearly parallel using dot product
            float angle = std::acos(std::abs(dir_i.dot(dir_j)));
            if (angle <= MAX_ANGLE_DIFF) {
                float dist = (mid_j - mid_i).norm();
                min_valid_dist = std::min(min_valid_dist, dist);
            }
        }
    }

    // If no valid pairs found, return empty vector
    if (min_valid_dist == std::numeric_limits<float>::max()) {
        return pairs;
    }

    // Find all pairs within acceptable distance and angle
    float max_allowed_dist = min_valid_dist * MAX_DIST_MULTIPLIER;

    for (size_t i = 0; i < line_segments.size(); ++i) {
        Eigen::Vector3f dir_i = (line_segments[i].end - line_segments[i].start).normalized();
        Eigen::Vector3f mid_i = (line_segments[i].start + line_segments[i].end) * 0.5f;

        // Find the closest valid neighbor for this line
        float min_dist = std::numeric_limits<float>::max();
        size_t best_j = i;

        for (size_t j = 0; j < line_segments.size(); ++j) {
            if (i == j) continue;

            Eigen::Vector3f dir_j = (line_segments[j].end - line_segments[j].start).normalized();
            Eigen::Vector3f mid_j = (line_segments[j].start + line_segments[j].end) * 0.5f;

            // Check if lines are nearly parallel
            float angle = std::acos(std::abs(dir_i.dot(dir_j)));
            if (angle <= MAX_ANGLE_DIFF) {
                float dist = (mid_j - mid_i).norm();
                if (dist < min_dist && dist <= max_allowed_dist) {
                    min_dist = dist;
                    best_j = j;
                }
            }
        }

        // Only add the pair if we found a valid neighbor and haven't added this pair yet
        if (best_j != i && i < best_j) {
            pairs.push_back({ i, best_j });
        }
    }

    return pairs;
}

// Visualization function
void visualizeCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& original_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& green_cloud,
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& line_clouds) {

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    int v1(0), v2(1), v3(2);
    viewer->createViewPort(0.0, 0.0, 0.33, 1.0, v1);
    viewer->createViewPort(0.33, 0.0, 0.66, 1.0, v2);
    viewer->createViewPort(0.66, 0.0, 1.0, 1.0, v3);

    // Add original and filtered clouds to first two viewports
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_original(original_cloud);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_green(green_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(original_cloud, rgb_original, "original_cloud", v1);
    viewer->addPointCloud<pcl::PointXYZRGB>(green_cloud, rgb_green, "green_cloud", v2);

    // Fit line segments and store them
    std::vector<LineSegment> line_segments;
    for (size_t i = 0; i < line_clouds.size(); ++i) {
        LineSegment segment = fitLineSegment(line_clouds[i]);
        line_segments.push_back(segment);

        // Visualize the line segment
        std::string line_name = "line_segment_" + std::to_string(i);
        viewer->addLine(
            pcl::PointXYZ(segment.start.x(), segment.start.y(), segment.start.z()),
            pcl::PointXYZ(segment.end.x(), segment.end.y(), segment.end.z()),
            1.0, 0.0, 0.0, line_name, v3
        );

        // Add the point cloud with a unique color
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> line_color(
            line_clouds[i],
            255 * (i % 3 == 0),  // R
            255 * (i % 3 == 1),  // G
            255 * (i % 3 == 2)   // B
        );
        viewer->addPointCloud<pcl::PointXYZRGB>(line_clouds[i], line_color,
            "line_cloud" + std::to_string(i), v3);
    }

    // Find and add prisms between closest line pairs
    auto closest_pairs = findClosestLinePairs(line_segments);
    for (size_t i = 0; i < closest_pairs.size(); ++i) {
        const auto& pair = closest_pairs[i];
        std::string prism_name = "prism_" + std::to_string(i);
        addPrismBetweenLines(viewer,
            line_segments[pair.first],
            line_segments[pair.second],
            prism_name, v3);
    }

    // Add viewport labels
    viewer->addText("Original Cloud", 10, 10, "original_label", v1);
    viewer->addText("Green Points", 10, 10, "green_label", v2);
    viewer->addText("Line Segments and Prisms", 10, 10, "line_label", v3);

    // Set up camera
    viewer->initCameraParameters();
    viewer->resetCamera();

    // Main visualization loop
    std::cout << "Visualizing point clouds. Press 'q' to exit..." << std::endl;
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main() {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(INPUT_FILE, *cloud) == -1) {
        std::cerr << "Couldn't read file " << INPUT_FILE << std::endl;
        return -1;
    }

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

    auto it = std::remove_if(points.begin(), points.end(),
        [](const Point& p) {
            return !(p.g > GREEN_THRESHOLD * p.r && p.g > GREEN_THRESHOLD * p.b);
        });
    points.erase(it, points.end());

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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_filtered_cloud = fitPlaneAndFilterOutliers(green_cloud);

    // Fit multiple lines on the plane-filtered points
    auto line_clouds = fitMultipleLines(plane_filtered_cloud);

    // Visualize all stages
    visualizeCloud(cloud, green_cloud, line_clouds);

    return 0;
}