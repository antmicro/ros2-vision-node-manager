// Copyright 2022-2023 Antmicro <www.antmicro.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cvnode_manager/cvnode_manager.hpp>
#include <mutex>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <thread>

namespace cvnode_manager
{

CVNodeManager::CVNodeManager(const rclcpp::NodeOptions &options) : Node("cvnode_manager", options)
{
    manage_service = create_service<ManageCVNode>(
        "cvnode_register",
        std::bind(&CVNodeManager::manage_node_callback, this, std::placeholders::_1, std::placeholders::_2));

    prepare_service = create_service<Trigger>(
        "cvnode_prepare",
        [this](const std::shared_ptr<rmw_request_id_t> header, const Trigger::Request::SharedPtr request)
        { std::thread([this, header, request]() { prepare_node(header, request); }).detach(); });

    measurements_service = create_service<Trigger>(
        "cvnode_measurements",
        std::bind(&CVNodeManager::upload_measurements, this, std::placeholders::_1, std::placeholders::_2));

    dataprovider_server = rclcpp_action::create_server<SegmentationAction>(
        this,
        "cvnode_process",
        std::bind(&CVNodeManager::handle_test_process_request, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&CVNodeManager::handle_test_process_cancel, this, std::placeholders::_1),
        [this](const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentationAction>> goal_handle)
        { std::thread([this, goal_handle]() { handle_test_process_start_processing(goal_handle); }).detach(); });

    rcl_interfaces::msg::ParameterDescriptor descriptor;

    // Parameter bool to publish visualizations
    descriptor.description = "Publishes input and output data to dedicated topics if set to true";
    descriptor.additional_constraints = "Must be a boolean value";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
    declare_parameter("publish_visualizations", false, descriptor);

    // Parameter int to set the timeout for the real-world scenarios
    descriptor.description = "Timeout for the real-world scenarios in milliseconds";
    descriptor.additional_constraints = "Must be an integer value greater than 0";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
    declare_parameter("inference_timeout_ms", 1000, descriptor);

    // Parameter string to choose testing strategy
    descriptor.description = "Testing scenario to run";
    descriptor.additional_constraints = "Must be one of: real_world_last, real_world_first, synthetic";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_STRING;
    declare_parameter("scenario", "synthetic", descriptor);

    // Parameter bool to preserve last output
    descriptor.description = "Preserves last frame output if timeout is reached during inference";
    descriptor.additional_constraints = "Must be a boolean value";
    descriptor.read_only = false;
    descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
    declare_parameter("preserve_output", true, descriptor);

    // Publishers for input and output data
    gui_input_publisher = create_publisher<sensor_msgs::msg::Image>("input_frame", 1);
    gui_output_publisher = create_publisher<SegmentationMsg>("output_segmentations", 1);

    // Real world timer which is manually triggered
    real_world_timer = create_wall_timer(
        std::chrono::milliseconds(get_parameter("inference_timeout_ms").as_int()),
        [this]()
        {
            real_world_timer->cancel();
            for (auto &output : cvnode_response->output)
            {
                if (get_parameter("publish_visualizations").as_bool())
                {
                    gui_output_publisher->publish(output);
                }
                cvnode_response_data.back().data.push_back(output);
            }

            if (!is_input_data_available())
            {
                publish_data();
                return;
            }

            if (get_parameter("scenario").as_string() == "real_world_first")
            {
                execute_real_world_first_inference_scenario();
            }
            else
            {
                execute_real_world_last_inference_scenario();
            }

            real_world_timer->reset();
        });
    real_world_timer->cancel();
}

rclcpp_action::GoalResponse CVNodeManager::handle_test_process_request(
    [[maybe_unused]] const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const SegmentationAction::Goal> goal)
{
    RCLCPP_DEBUG(get_logger(), "Received action request with data");
    if (goal->input.empty())
    {
        RCLCPP_ERROR(get_logger(), "Received empty input");
        return rclcpp_action::GoalResponse::REJECT;
    }

    cvnode_request_data = goal->input;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse CVNodeManager::handle_test_process_cancel(
    [[maybe_unused]] const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentationAction>> goal_handle)
{
    RCLCPP_DEBUG(get_logger(), "Received request to cancel goal");
    RCLCPP_DEBUG(get_logger(), "Cleaning up CVNode");
    Trigger::Request::SharedPtr request = std::make_shared<Trigger::Request>();
    cv_node.cleanup->async_send_request(request);
    return rclcpp_action::CancelResponse::ACCEPT;
}

void CVNodeManager::handle_test_process_start_processing(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentationAction>> goal_handle)
{
    std::string scenario = get_parameter("scenario").as_string();
    cvnode_response_data.clear();
    cvnode_response_data.push_back(VideoSegmentationMsg());
    request_handle = goal_handle;
    if (scenario == "synthetic")
    {
        RCLCPP_DEBUG(get_logger(), "Executing synthetic scenario");
        execute_synthetic_inference_scenario();
        return;
    }
    else if (scenario == "real_world_first")
    {
        RCLCPP_DEBUG(get_logger(), "Executing real-world scenario with first frame");
        execute_real_world_first_inference_scenario();
        real_world_timer->reset();
        return;
    }
    else if (scenario == "real_world_last")
    {
        RCLCPP_DEBUG(get_logger(), "Executing real-world scenario with last frame");
        execute_real_world_last_inference_scenario();
        real_world_timer->reset();
        return;
    }
    else
    {
        abort_request("Unknown scenario");
        return;
    }
}

void CVNodeManager::prepare_node(
    const std::shared_ptr<rmw_request_id_t> header,
    [[maybe_unused]] const std::shared_ptr<Trigger::Request> request)
{
    if (cv_node.process == nullptr)
    {
        std::mutex dataprovider_mutex;
        std::unique_lock<std::mutex> lk(dataprovider_mutex);
        RCLCPP_DEBUG(get_logger(), "Waiting for inference to start");
        cvnode_wait_cv.wait(lk);
    }

    RCLCPP_DEBUG(get_logger(), "Preparing node");
    if (!cv_node.prepare->wait_for_service(std::chrono::seconds(1)))
    {
        RCLCPP_ERROR(get_logger(), "Node to prepare is not ready");
        Trigger::Response response;
        response.success = false;
        response.message = "Node to prepare is not ready";
        prepare_service->send_response(*header, response);
        return;
    }
    else
    {
        cv_node.prepare->async_send_request(
            request,
            [this, header](rclcpp::Client<Trigger>::SharedFuture future)
            {
                auto response = future.get();
                Trigger::Response runtime_response;
                if (!response->success)
                {
                    RCLCPP_ERROR(get_logger(), "Error while loading the model");
                    runtime_response.success = false;
                    prepare_service->send_response(*header, runtime_response);
                    return;
                }
                // Reset measurements
                measurements = nlohmann::json();
                measurements.emplace("target_inference_step", nlohmann::json::array());
                measurements.emplace("target_inference_step_timestamp", nlohmann::json::array());

                RCLCPP_DEBUG(get_logger(), "CVNode prepared successfully");
                runtime_response.success = true;
                prepare_service->send_response(*header, runtime_response);
                return;
            });
    }
}

void CVNodeManager::upload_measurements(
    const std::shared_ptr<rmw_request_id_t> header,
    const Trigger::Request::SharedPtr request)
{
    RCLCPP_DEBUG(get_logger(), "Cleaning up CVNode");
    cv_node.cleanup->async_send_request(
        request,
        [this, header](rclcpp::Client<Trigger>::SharedFuture future)
        {
            auto response = future.get();
            response->message = measurements.dump();
            if (!response->success)
            {
                RCLCPP_ERROR(get_logger(), "Error while cleaning up the node");
                measurements_service->send_response(*header, *response);
                return;
            }
            else if (response->message.empty())
            {
                RCLCPP_ERROR(get_logger(), "No measurements to upload");
                response->success = false;
                response->message = "No measurements to upload";
                return;
            }

            RCLCPP_DEBUG(get_logger(), "CVNode cleaned up successfully. Uploading measurements.");
            measurements_service->send_response(*header, *response);
            return;
        });
}

bool CVNodeManager::is_input_data_available()
{
    if (!cvnode_request_data.empty())
    {
        if (!cvnode_request_data.back().frames.empty())
        {
            return true;
        }
    }
    return false;
}

void CVNodeManager::publish_data()
{
    auto result = std::make_shared<SegmentationAction::Result>();
    result->success = true;
    result->output = cvnode_response_data;
    request_handle->succeed(result);
    cvnode_response_data.clear();
}

void CVNodeManager::abort_request(const std::string &message)
{
    auto result = std::make_shared<SegmentationAction::Result>();
    result->success = false;
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    request_handle->abort(result);
    request_handle.reset();
}

void CVNodeManager::prepare_cvnode_request()
{
    cvnode_request->input.clear();
    if (cvnode_request_data.front().frames.empty())
    {
        cvnode_request_data.erase(cvnode_request_data.begin());
        cvnode_response_data.push_back(VideoSegmentationMsg());
    }
    cvnode_request->input.push_back(cvnode_request_data.front().frames.front());
    cvnode_request_data.front().frames.erase(cvnode_request_data.front().frames.begin());
    if (get_parameter("publish_visualizations").as_bool())
    {
        gui_input_publisher->publish(cvnode_request->input.front());
    }
}

void CVNodeManager::execute_synthetic_inference_scenario()
{
    using namespace std::chrono;
    if (!is_input_data_available())
    {
        abort_request("No input data to process");
        return;
    }
    prepare_cvnode_request();

    start = steady_clock::now();
    cv_node.process->async_send_request(
        cvnode_request,
        [this](const rclcpp::Client<SegmentCVNodeSrv>::SharedFuture future)
        {
            cvnode_response = future.get();
            if (!cvnode_response->success)
            {
                abort_request("Error while processing data");
                return;
            }
            end = steady_clock::now();

            // Save measurements
            float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
            measurements.at("target_inference_step").push_back(duration);
            measurements.at("target_inference_step_timestamp")
                .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);

            for (auto &output : cvnode_response->output)
            {
                if (get_parameter("publish_visualizations").as_bool())
                {
                    gui_output_publisher->publish(output);
                }
                cvnode_response_data.back().data.push_back(output);
            }

            if (!is_input_data_available())
            {
                publish_data();
                return;
            }

            execute_synthetic_inference_scenario();
        });
}

void CVNodeManager::execute_real_world_last_inference_scenario()
{
    using namespace std::chrono;

    if (!is_input_data_available())
    {
        abort_request("No input data to process");
        return;
    }
    prepare_cvnode_request();

    // Local copy of request_id
    current_request_id += 1;
    const uint32_t request_id = current_request_id;

    // Send inference request
    start = steady_clock::now();
    cv_node.process->async_send_request(
        cvnode_request,
        [this, request_id](const rclcpp::Client<SegmentCVNodeSrv>::SharedFuture future)
        {
            if (current_request_id != request_id)
            {
                RCLCPP_DEBUG(get_logger(), "Timeout was reached, ignoring inference output");
                return;
            }
            cvnode_response = future.get();
            if (!cvnode_response->success)
            {
                abort_request("Error while processing data");
                return;
            }
            end = steady_clock::now();

            // Measure inference time
            float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
            measurements.at("target_inference_step").push_back(duration);
            measurements.at("target_inference_step_timestamp")
                .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);
        });
}

void CVNodeManager::execute_real_world_first_inference_scenario()
{
    using namespace std::chrono;

    if (!is_input_data_available())
    {
        abort_request("No input data to process");
        return;
    }
    prepare_cvnode_request();

    if (processing_request)
    {
        RCLCPP_DEBUG(get_logger(), "Request is already in progress. Skipping...");
        return;
    }
    processing_request = true;

    // Send inference request
    start = steady_clock::now();
    cv_node.process->async_send_request(
        cvnode_request,
        [this](const rclcpp::Client<SegmentCVNodeSrv>::SharedFuture future)
        {
            cvnode_response = future.get();
            processing_request = false;
            if (!cvnode_response->success)
            {
                abort_request("Error while processing data");
                return;
            }
            end = steady_clock::now();

            // Measure inference time
            float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
            measurements.at("target_inference_step").push_back(duration);
            measurements.at("target_inference_step_timestamp")
                .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);
        });
}

void CVNodeManager::manage_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    ManageCVNode::Response::SharedPtr response)
{
    if (request->type == request->REGISTER)
    {
        register_node_callback(request, response);
    }
    else if (request->type == request->UNREGISTER)
    {
        unregister_node_callback(request, response);
    }
    else
    {
        RCLCPP_ERROR(get_logger(), "Unknown type of the service");
        response->status = false;
        response->message = "Unknown type of the service";
    }
    return;
}

void CVNodeManager::register_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    ManageCVNode::Response::SharedPtr response)
{
    std::string node_name = request->node_name;
    response->status = false;
    RCLCPP_DEBUG(get_logger(), "Registering the '%s' node", node_name.c_str());

    if (cv_node.process != nullptr)
    {
        response->message = "There is already a node registered";
        RCLCPP_WARN(
            get_logger(),
            "Could not register the node '%s' because there is already a node registered",
            node_name.c_str());
        return;
    }

    // Create clients to communicate with the node
    rclcpp::Client<Trigger>::SharedPtr cvnode_prepare;
    if (!initialize_service_client<Trigger>(request->prepare_srv_name, cvnode_prepare))
    {
        response->message = "Could not initialize the prepare service client";
        RCLCPP_ERROR(get_logger(), "Could not initialize the prepare service client");
        return;
    }
    rclcpp::Client<SegmentCVNodeSrv>::SharedPtr cvnode_process;
    if (!initialize_service_client<SegmentCVNodeSrv>(request->process_srv_name, cvnode_process))
    {
        response->message = "Could not initialize the process service client";
        RCLCPP_ERROR(get_logger(), "Could not initialize the process service client");
        return;
    }
    rclcpp::Client<Trigger>::SharedPtr cvnode_cleanup;
    if (!initialize_service_client<Trigger>(request->cleanup_srv_name, cvnode_cleanup))
    {
        response->message = "Could not initialize the cleanup service client";
        RCLCPP_ERROR(get_logger(), "Could not initialize the cleanup service client");
        return;
    }

    response->status = true;
    response->message = "The node is registered";
    cv_node = CVNode(node_name, cvnode_prepare, cvnode_process, cvnode_cleanup);

    cvnode_wait_cv.notify_one();
    RCLCPP_DEBUG(get_logger(), "The node '%s' is registered", node_name.c_str());
}

void CVNodeManager::unregister_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    [[maybe_unused]] ManageCVNode::Response::SharedPtr response)
{
    std::string node_name = request->node_name;

    RCLCPP_DEBUG(get_logger(), "Unregistering the node '%s'", node_name.c_str());
    if (cv_node.name != node_name)
    {
        RCLCPP_WARN(get_logger(), "The node '%s' is not registered", node_name.c_str());
        return;
    }

    cv_node = CVNode();
    RCLCPP_DEBUG(get_logger(), "The node '%s' is unregistered", node_name.c_str());
    return;
}

} // namespace cvnode_manager

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(cvnode_manager::CVNodeManager)
