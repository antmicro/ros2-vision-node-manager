/*
 * Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <kenning_computer_vision_msgs/msg/box_msg.hpp>
#include <kenning_computer_vision_msgs/msg/segmentation_msg.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <gui_node/gui_node.hpp>
#include <gui_node/ros_data/ros_subscriber_data.hpp>
#include <gui_node/utils/detection.hpp>
#include <gui_node/widget/widget_detection.hpp>
#include <gui_node/widget/widget_rosout.hpp>
#include <gui_node/widget/widget_video.hpp>

namespace cvnode_manager
{
namespace gui
{

using namespace gui_node;
using SegmentationMsg = kenning_computer_vision_msgs::msg::SegmentationMsg;

class CVNodeManagerGUI
{
private:
    std::shared_ptr<GuiNode> gui_node_ptr; ///< Pointer to the GUI node

    /**
     * Prepares the display for the Kenning's instance segmentation message.
     *
     * @param instance_segmentation_msg The segmentation message.
     * @param bounding_boxes The vector of bounding boxes to fill.
     * @param filterclass The class to filter masks by.
     * @param threshold The threshold to filter masks by.
     * @return sensor_msgs::msg::Image The image to display.
     */
    sensor_msgs::msg::Image prep_display(
        SegmentationMsg::SharedPtr instance_segmentation_msg,
        std::vector<BoundingBox> &bounding_boxes,
        const std::string &filterclass,
        const float &threshold)
    {
        kenning_computer_vision_msgs::msg::BoxMsg box;
        float score;
        int color_idx;
        std::string class_name;

        int height = instance_segmentation_msg->frame.height;
        int width = instance_segmentation_msg->frame.width;

        cv::Mat frame(height, width, CV_8UC3, instance_segmentation_msg->frame.data.data());

        for (size_t i = 0; i < instance_segmentation_msg->boxes.size(); i++)
        {
            class_name = instance_segmentation_msg->classes[i];
            score = instance_segmentation_msg->scores[i] * 100;
            color_idx = i % im_colors.size();
            box = instance_segmentation_msg->boxes[i];
            bounding_boxes.push_back(
                BoundingBox(box.xmin, box.ymin, box.xmax, box.ymax, class_name, score, im_colors[color_idx]));

            std::transform(class_name.begin(), class_name.end(), class_name.begin(), ::tolower);
            if (score >= threshold && (filterclass.empty() || class_name.find(filterclass) != std::string::npos))
            {
                // Apply mask
                cv::Mat mask(
                    instance_segmentation_msg->masks[i].dimension[0],
                    instance_segmentation_msg->masks[i].dimension[1],
                    CV_8UC1,
                    instance_segmentation_msg->masks[i].data.data());
                cv::resize(mask, mask, cv::Size(width, height));
                cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
                cv::Scalar cv_color = cv::Scalar(
                    im_colors[color_idx].Value.z * 255,
                    im_colors[color_idx].Value.y * 255,
                    im_colors[color_idx].Value.x * 255);
                mask.setTo(cv_color, mask);
                cv::addWeighted(mask, 0.4, frame, 1.0, 0, frame);
            }
        }
        return instance_segmentation_msg->frame;
    }

    /// ImGui color definitions
    const std::vector<ImColor> im_colors{
        ImColor(ImVec4(0.96f, 0.26f, 0.21f, 1.0f)),
        ImColor(ImVec4(0.91f, 0.12f, 0.39f, 1.0f)),
        ImColor(ImVec4(0.61f, 0.15f, 0.69f, 1.0f)),
        ImColor(ImVec4(0.4f, 0.23f, 0.72f, 1.0f)),
        ImColor(ImVec4(0.25f, 0.32f, 0.71f, 1.0f)),
        ImColor(ImVec4(0.13f, 0.59f, 0.95f, 1.0f)),
        ImColor(ImVec4(0.01f, 0.66f, 0.96f, 1.0f)),
        ImColor(ImVec4(0.0f, 0.74f, 0.83f, 1.0f)),
        ImColor(ImVec4(0.0f, 0.59f, 0.53f, 1.0f)),
        ImColor(ImVec4(0.3f, 0.69f, 0.31f, 1.0f)),
        ImColor(ImVec4(0.55f, 0.76f, 0.29f, 1.0f)),
        ImColor(ImVec4(0.8f, 0.86f, 0.22f, 1.0f)),
        ImColor(ImVec4(1.0f, 0.92f, 0.23f, 1.0f)),
        ImColor(ImVec4(1.0f, 0.76f, 0.03f, 1.0f)),
        ImColor(ImVec4(1.0f, 0.6f, 0.0f, 1.0f)),
        ImColor(ImVec4(1.0f, 0.34f, 0.13f, 1.0f)),
        ImColor(ImVec4(0.47f, 0.33f, 0.28f, 1.0f)),
        ImColor(ImVec4(0.62f, 0.62f, 0.62f, 1.0f)),
        ImColor(ImVec4(0.38f, 0.49f, 0.55f, 1.0f))};

public:
    CVNodeManagerGUI(const rclcpp::NodeOptions &options)
    {
        using MsgImageSharedPtr = sensor_msgs::msg::Image::SharedPtr;
        using RosImageSubscriberData = RosSubscriberData<sensor_msgs::msg::Image, sensor_msgs::msg::Image::SharedPtr>;
        using RosSegmentationSubscriberData = RosSubscriberData<SegmentationMsg, SegmentationMsg::SharedPtr>;

        gui_node_ptr = std::make_shared<GuiNode>(options, "gui_node");

        // Widget to display instance segmentations
        std::shared_ptr<RosSegmentationSubscriberData> subscriber_instance_segmentation =
            std::make_shared<RosSegmentationSubscriberData>(
                gui_node_ptr,
                "output_segmentations",
                [](const SegmentationMsg::SharedPtr msg) -> SegmentationMsg::SharedPtr { return msg; });
        gui_node_ptr->addRosData("output_subscriber", subscriber_instance_segmentation);

        std::shared_ptr<DetectionWidget> instance_segmentation_widget = std::make_shared<DetectionWidget>(
            gui_node_ptr,
            "[Sub] Instance Segmentation stream",
            "output_subscriber",
            [this](
                std::shared_ptr<GuiNode> gui_node_ptr,
                sensor_msgs::msg::Image &msg,
                std::vector<BoundingBox> &boxes,
                const std::string &filterclass,
                const float &threshold) -> void
            {
                std::shared_ptr<RosSegmentationSubscriberData> subscriber_instance_segmentation =
                    gui_node_ptr->getRosData("output_subscriber")->as<RosSegmentationSubscriberData>();
                SegmentationMsg::SharedPtr instance_segmentation_msg = subscriber_instance_segmentation->getData();

                msg = prep_display(instance_segmentation_msg, boxes, filterclass, threshold);
            });
        gui_node_ptr->addWidget("instance_segmentation_widget", instance_segmentation_widget);

        // Create a widget to display the video stream
        std::shared_ptr<RosImageSubscriberData> subscriber_video = std::make_shared<RosImageSubscriberData>(
            gui_node_ptr,
            "input_frame",
            [](const MsgImageSharedPtr msg) -> MsgImageSharedPtr { return msg; });
        gui_node_ptr->addRosData("input_subscriber", subscriber_video);

        std::shared_ptr<VideoWidget> video_widget = std::make_shared<VideoWidget>(
            gui_node_ptr,
            "[Sub] Input stream",
            "input_subscriber",
            [](std::shared_ptr<GuiNode> gui_node_ptr, sensor_msgs::msg::Image &msg) -> void
            {
                std::shared_ptr<RosImageSubscriberData> subscriber_video =
                    gui_node_ptr->getRosData("input_subscriber")->as<RosImageSubscriberData>();
                msg = *subscriber_video->getData().get();
            });
        gui_node_ptr->addWidget("input_widget", video_widget);

        gui_node_ptr->prepare("Instance segmentation");
    }

    rclcpp::node_interfaces::NodeBaseInterface::SharedPtr get_node_base_interface()
    {
        return gui_node_ptr->get_node_base_interface();
    }
};

} // namespace gui
} // namespace cvnode_manager

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(cvnode_manager::gui::CVNodeManagerGUI)
