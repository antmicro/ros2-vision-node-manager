// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <unordered_map>

#include <kenning_computer_vision_msgs/msg/segmentation_msg.hpp>
#include <kenning_computer_vision_msgs/srv/manage_cv_node.hpp>
#include <kenning_computer_vision_msgs/srv/runtime_protocol_srv.hpp>
#include <kenning_computer_vision_msgs/srv/segment_cv_node_srv.hpp>

namespace cvnode_manager
{

/**
 * Node to manage testing process of the computer vision system.
 */
class CVNodeManager : public rclcpp::Node
{
private:
    /**
     * Callback for the service to manage the CVNode-like node.
     *
     * @param request Request of the service for registering the node.
     * @param response Response of the service for registering the node.
     */
    void manage_node_callback(
        const kenning_computer_vision_msgs::srv::ManageCVNode::Request::SharedPtr request,
        kenning_computer_vision_msgs::srv::ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for the service to register the CVNode-like node.
     *
     * @param request Request of the service for registering the node.
     * @param response Response of the service for registering the node.
     */
    void register_node_callback(
        const kenning_computer_vision_msgs::srv::ManageCVNode::Request::SharedPtr request,
        kenning_computer_vision_msgs::srv::ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for the service to unregister the CVNode-like node.
     *
     * @param request Request of the service for unregistering the node.
     * @param response Response of the service for unregistering the node.
     */
    void unregister_node_callback(
        const kenning_computer_vision_msgs::srv::ManageCVNode::Request::SharedPtr request,
        [[maybe_unused]] kenning_computer_vision_msgs::srv::ManageCVNode::Response::SharedPtr response);

    /**
     * Callback for communication with dataprovider.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void dataprovider_callback(
        const std::shared_ptr<rmw_request_id_t> header,
        const kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Request::SharedPtr request);

    /**
     * Extracts input data from bytes-encoded.
     *
     * @param input_data_b Bytes-encoded input data.
     *
     * @return request Request with data to distribute. If error occurred, message type is set to ERROR.
     */
    kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Request::SharedPtr
    extract_images(std::vector<uint8_t> &input_data_b);

    /**
     * Broadcasts request to registered CVNode-like node.
     * Responses with 'OK' to DataProvider when received confirmation, 'ERROR' otherwise.
     *
     * @param header Header of the service request.
     * @param request Request to broadcast.
     */
    void async_broadcast_request(
        const std::shared_ptr<rmw_request_id_t> header,
        const kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Request::SharedPtr request);

    /**
     * Broadcasts request to registered CVNode-like node.
     * Responses with 'OK' to DataProvider when received confirmation, 'ERROR' otherwise.
     *
     * @param header Header of the service request.
     * @param request Request to broadcast.
     * @param callback Callback to be executed with every received confirmation.
     */
    void async_broadcast_request(
        const std::shared_ptr<rmw_request_id_t> header,
        const kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Request::SharedPtr request,
        std::function<kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Response(
            const kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Response::SharedPtr)> callback);

    /**
     * Converts SegmentationMsg to json.
     *
     * @param segmentation Segmentation to be extracted.
     *
     * @return Array of JSON segmentation representations.
     */
    nlohmann::json segmentation_to_json(const kenning_computer_vision_msgs::msg::SegmentationMsg &segmentation);

    /**
     * Synthetic testing scenario.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void synthetic_scenario(
        const std::shared_ptr<rmw_request_id_t> header,
        const kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Request::SharedPtr request);

    /**
     * Creates a client to a service.
     *
     * @param service_name Name of the service.
     * @param client Client of the service to be created and initialized.
     *
     * @tparam TClient Type of the service client.
     *
     * @return True if the client was created and initialized successfully.
     */
    template <class TClient>
    bool initialize_service_client(const std::string &service_name, typename rclcpp::Client<TClient>::SharedPtr &client)
    {
        if (service_name.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "%s", "The process service name is empty");
            return false;
        }

        client = this->create_client<TClient>(service_name);

        if (!client->wait_for_service(std::chrono::seconds(1)))
        {
            RCLCPP_ERROR(this->get_logger(), "The service '%s' is not available", service_name.c_str());
            return false;
        }
        return true;
    }

    /**
     * Aborts further processing.
     *
     * Reports error to the registered CVNode-like node.
     */
    void abort();

    /// Service to register the CVNode
    rclcpp::Service<kenning_computer_vision_msgs::srv::ManageCVNode>::SharedPtr manage_service;

    /// Client to communicate with Kenning
    rclcpp::Service<kenning_computer_vision_msgs::srv::RuntimeProtocolSrv>::SharedPtr dataprovider_service;

    /// Publisher to publish input data
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr input_publisher;

    /// Publisher to publish output data
    rclcpp::Publisher<kenning_computer_vision_msgs::msg::SegmentationMsg>::SharedPtr output_publisher;

    bool dataprovider_initialized = false;   ///< Flag indicating whether the dataprovider is initialized
    std::mutex dataprovider_mutex;           ///< Mutex for dataprovider
    std::condition_variable dataprovider_cv; ///< Condition variable for dataprovider

    /// Registered CVNode-like node
    std::tuple<std::string, rclcpp::Client<kenning_computer_vision_msgs::srv::SegmentCVNodeSrv>::SharedPtr> cv_node =
        std::make_tuple("", nullptr);

    /// Testing scenario function
    void (CVNodeManager::*inference_scenario_func)(
        const std::shared_ptr<rmw_request_id_t>,
        const kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Request::SharedPtr) = nullptr;

    std::shared_ptr<nlohmann::json> output_json; ///< JSON storing output segmentations

public:
    /**
     * Constructor.
     *
     * @param options Node options.
     */
    CVNodeManager(const rclcpp::NodeOptions &options);
};

} // namespace cvnode_manager
