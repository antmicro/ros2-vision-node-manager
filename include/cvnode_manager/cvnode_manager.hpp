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
     * Callback for communication with DataProvider.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void dataprovider_callback(
        const std::shared_ptr<rmw_request_id_t> header,
        const kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Request::SharedPtr request);

    /**
     * Holds DataProvider initialization up until inference should be started.
     *
     * @param header Header of the service request.
     */
    void initialize_dataprovider(const std::shared_ptr<rmw_request_id_t> header);

    /**
     * Extracts input data from bytes-encoded and forwards it to the CVNode-like node.
     *
     * @param header Header of the service request.
     * @param request Request of the service.
     */
    void forward_data_request(
        const std::shared_ptr<rmw_request_id_t> header,
        const kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Request::SharedPtr request);

    /**
     * Forwards output request to the CVNode-like node. Extracts output data from the response
     * and forwards it back to the DataProvider.
     *
     * @param header Header of the service request.
     */
    void forward_output_request(const std::shared_ptr<rmw_request_id_t> header);

    /**
     * Initializes testing scenario strategy from ROS2 parameter.
     *
     * @return True if initialization was successful, false otherwise.
     */
    bool set_scenario();

    /**
     * Extracts input data from bytes-encoded.
     *
     * @param input_data_b Bytes-encoded input data.
     *
     * @return Request with data to distribute. If error occurred, message type is set to ERROR.
     */
    kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Request::SharedPtr
    extract_images(std::vector<uint8_t> &input_data_b);

    /**
     * Converts output segmentations from the CVNode-like node to bytes-encoded state.
     *
     * @param response CVNode-like node response with segmentations attached.
     *
     * @return Dataprovider response with output data from the CVNode-like node.
     *         If error occurred, message type is set to ERROR.
     */
    kenning_computer_vision_msgs::srv::RuntimeProtocolSrv::Response segmentations_to_output_data(
        const kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Response::SharedPtr response);

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
            const kenning_computer_vision_msgs::srv::SegmentCVNodeSrv::Response::SharedPtr)> callback = nullptr);

    /**
     * Synthetic testing scenario.
     *
     * @param header Header of the service request.
     */
    void synthetic_inference_scenario(const std::shared_ptr<rmw_request_id_t> header);

    /**
     * Real-world testing scenario where last request is aborted if new one is received.
     *
     * @param header Header of the service request.
     */
    void real_world_last_inference_scenario(const std::shared_ptr<rmw_request_id_t> header);

    /**
     * Real-world testing scenario where new request is aborted if last one is not finished yet.
     *
     * @param header Header of the service request.
     */
    void real_world_first_inference_scenario(const std::shared_ptr<rmw_request_id_t> header);

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
     * Reports error to the registered CVNode-like node and DataProvider.
     *
     * @param header Header of the service request.
     * @param error_msg Error message to be logged.
     */
    void abort(const std::shared_ptr<rmw_request_id_t> header, const std::string &error_msg);

    /// Service to register the CVNode
    rclcpp::Service<kenning_computer_vision_msgs::srv::ManageCVNode>::SharedPtr manage_service;

    /// Client to communicate with Kenning
    rclcpp::Service<kenning_computer_vision_msgs::srv::RuntimeProtocolSrv>::SharedPtr dataprovider_service;

    /// Publisher to publish input data
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr input_publisher;

    /// Publisher to publish output data
    rclcpp::Publisher<kenning_computer_vision_msgs::msg::SegmentationMsg>::SharedPtr output_publisher;

    bool dataprovider_initialized = false;   ///< Flag indicating whether the DataProvider is initialized
    std::mutex dataprovider_mutex;           ///< Mutex for DataProvider
    std::condition_variable dataprovider_cv; ///< Condition variable for DataProvider

    /// Registered CVNode-like node
    std::tuple<std::string, rclcpp::Client<kenning_computer_vision_msgs::srv::SegmentCVNodeSrv>::SharedPtr> cv_node =
        std::make_tuple("", nullptr);

    // Shared future from last request
    rclcpp::Client<kenning_computer_vision_msgs::srv::SegmentCVNodeSrv>::SharedFuture cv_node_future =
        rclcpp::Client<kenning_computer_vision_msgs::srv::SegmentCVNodeSrv>::SharedFuture();

    /// Testing scenario function
    std::function<void(const std::shared_ptr<rmw_request_id_t>)> inference_scenario_func = nullptr;

    nlohmann::json measurements = nlohmann::json(); ///< Measurements from inference
    std::chrono::steady_clock::time_point start;    ///< Start time of the inference
    std::chrono::steady_clock::time_point end;      ///< End time of the inference
public:
    /**
     * Constructor.
     *
     * @param options Node options.
     */
    CVNodeManager(const rclcpp::NodeOptions &options);
};

} // namespace cvnode_manager
