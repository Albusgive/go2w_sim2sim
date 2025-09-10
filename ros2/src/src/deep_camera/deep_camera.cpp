#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class DepthImageProcessor : public rclcpp::Node {
public:
  DepthImageProcessor() : Node("depth_image_processor") {

    min_depth_ = 0.25;
    max_depth_ = 2.0;
    deep_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/camera/depth/image_rect_raw", 1,
        std::bind(&DepthImageProcessor::depth_callback, this,
                  std::placeholders::_1));
    // rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    //     "/camera/camera/color/image_raw", 1,
    //     std::bind(&DepthImageProcessor::rgb_callback, this,
    //               std::placeholders::_1));
  }

  ~DepthImageProcessor() { cv::destroyAllWindows(); }

private:
  void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImagePtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image = cv_ptr->image;
    cv::imshow("Image", image);
    cv::waitKey(1);
  }
  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {

    std::cout << "Received depth image: width=" << msg->width
              << " height=" << msg->height << "     encoding=" << msg->encoding
              << std::endl;

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);

    cv::Mat depth_image = cv_ptr->image;

    cv::Mat processed_image = process_depth_image(depth_image, msg->encoding);
    cv::Mat inv_image = processed_image.clone();
    for (int i = 0; i < inv_image.rows; i++) {
      uchar *p = inv_image.ptr<uchar>(i);
      for (int j = 0; j < inv_image.cols; j++) {
        if (p[j] != 0) {
          p[j] = 255 - p[j];
        }
      }
    }
    cv::imshow("Processed Depth Image", processed_image);
    cv::imshow("Inv Depth Image", inv_image);
    cv::waitKey(1);
  }

  cv::Mat process_depth_image(cv::Mat &depth_image,
                              const std::string &encoding) {
    cv::Mat float_image, normalized_image, display_image;
    if (encoding == "16UC1") {
      depth_image.convertTo(float_image, CV_32F, 1.0 / 1000.0); // 转换为米
    } else if (encoding == "32FC1") {
      depth_image.convertTo(float_image, CV_32F);
    } else {
      RCLCPP_WARN(this->get_logger(), "Unhandled image encoding: %s",
                  encoding.c_str());
      depth_image.convertTo(float_image, CV_32F);
    }
    cv::Mat mask;
    cv::inRange(float_image, min_depth_, max_depth_, mask);
    cv::Mat clamped_image = float_image.clone();
    clamped_image.setTo(min_depth_, float_image < min_depth_);
    clamped_image.setTo(max_depth_, float_image > max_depth_);
    normalized_image = (clamped_image - min_depth_) / (max_depth_ - min_depth_);

    normalized_image.convertTo(display_image, CV_8UC1, 255.0);

    // cv::applyColorMap(display_image, display_image, cv::COLORMAP_JET);
    return display_image;
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr deep_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  double min_depth_;
  double max_depth_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DepthImageProcessor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}