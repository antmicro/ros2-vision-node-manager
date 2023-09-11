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

    prepare_service = create_service<std_srvs::srv::Trigger>(
        "cvnode_prepare",
        [this](const std::shared_ptr<rmw_request_id_t> header, const std_srvs::srv::Trigger::Request::SharedPtr request)
        { std::thread([this, header, request]() { prepare_node(header, request); }).detach(); });

    measurements_service = create_service<std_srvs::srv::Trigger>(
        "cvnode_measurements",
        [this](const std::shared_ptr<rmw_request_id_t> header, const std_srvs::srv::Trigger::Request::SharedPtr request)
        { std::thread([this, header, request]() { upload_measurements(header, request); }).detach(); });

    dataprovider_server = rclcpp_action::create_server<SegmentationAction>(
        this,
        "cvnode_process",
        std::bind(&CVNodeManager::handle_test_process_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&CVNodeManager::handle_test_process_cancel, this, std::placeholders::_1),
        std::bind(&CVNodeManager::handle_test_process_accepted, this, std::placeholders::_1));

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
}

rclcpp_action::GoalResponse CVNodeManager::handle_test_process_goal(
    [[maybe_unused]] const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const SegmentationAction::Goal> goal)
{
    RCLCPP_DEBUG(get_logger(), "Received action request with data");
    if (goal->input.empty())
    {
        RCLCPP_ERROR(get_logger(), "Received empty input");
        return rclcpp_action::GoalResponse::REJECT;
    }

    cvnode_request->input = goal->input;
    if (get_parameter("publish_visualizations").as_bool())
    {
        for (auto &input : cvnode_request->input)
        {
            gui_input_publisher->publish(input);
        }
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse CVNodeManager::handle_test_process_cancel(
    [[maybe_unused]] const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentationAction>> goal_handle)
{
    RCLCPP_DEBUG(get_logger(), "Received request to cancel goal");
    RCLCPP_DEBUG(get_logger(), "Cleaning up CVNode");
    std_srvs::srv::Trigger::Request::SharedPtr request = std::make_shared<std_srvs::srv::Trigger::Request>();
    cv_node.cleanup->async_send_request(request);
    return rclcpp_action::CancelResponse::ACCEPT;
}

void CVNodeManager::handle_test_process_accepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentationAction>> goal_handle)
{
    std::thread(
        [this, goal_handle]()
        {
            std::string scenario = get_parameter("scenario").as_string();
            SegmentationAction::Result::SharedPtr result = std::make_shared<SegmentationAction::Result>();
            if (scenario == "synthetic")
            {
                RCLCPP_DEBUG(get_logger(), "Executing synthetic scenario");
                if (!execute_synthetic_inference_scenario())
                {
                    RCLCPP_ERROR(get_logger(), "Failed execution of synthetic scenario");
                    result->success = false;
                    goal_handle->abort(result);
                    return;
                }
            }
            else if (scenario == "real_world_first")
            {
                RCLCPP_DEBUG(get_logger(), "Executing real-world scenario with first frame");
                if (!execute_real_world_first_inference_scenario())
                {
                    RCLCPP_ERROR(get_logger(), "Failed execution of real-world scenario with first frame");
                    result->success = false;
                    goal_handle->abort(result);
                    return;
                }
            }
            else if (scenario == "real_world_last")
            {
                RCLCPP_DEBUG(get_logger(), "Executing real-world scenario with last frame");
                if (!execute_real_world_last_inference_scenario())
                {
                    RCLCPP_ERROR(get_logger(), "Failed execution of real-world scenario with last frame");
                    result->success = false;
                    goal_handle->abort(result);
                    return;
                }
            }
            else
            {
                RCLCPP_ERROR(get_logger(), "Unknown scenario '%s'", scenario.c_str());
                result->success = false;
                goal_handle->abort(result);
                return;
            }
            result->output = cvnode_response->output;
            if (!get_parameter("preserve_output").as_bool())
            {
                cvnode_response->output.clear();
            }

            if (get_parameter("publish_visualizations").as_bool())
            {
                for (auto &output : result->output)
                {
                    gui_output_publisher->publish(output);
                }
            }
            RCLCPP_DEBUG(get_logger(), "Successfully executed scenario");
            goal_handle->succeed(result);
        })
        .detach();
}

void CVNodeManager::prepare_node(
    const std::shared_ptr<rmw_request_id_t> header,
    [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Trigger::Request> request)
{
    if (cv_node.process == nullptr)
    {
        std::mutex dataprovider_mutex;
        std::unique_lock<std::mutex> lk(dataprovider_mutex);
        RCLCPP_DEBUG(get_logger(), "Waiting for inference to start");
        cvnode_wait_cv.wait(lk);
    }

    std_srvs::srv::Trigger::Response response;
    RCLCPP_DEBUG(get_logger(), "Preparing node");
    if (!cv_node.prepare->wait_for_service(std::chrono::seconds(1)))
    {
        RCLCPP_ERROR(get_logger(), "Node to prepare is not ready");
        response.success = false;
        response.message = "Node to prepare is not ready";
        prepare_service->send_response(*header, response);
        return;
    }
    else
    {
        cv_node.prepare->async_send_request(
            request,
            [this, header](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future)
            {
                auto response = future.get();
                std_srvs::srv::Trigger::Response runtime_response;
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
    [[maybe_unused]] const std_srvs::srv::Trigger::Request::SharedPtr request)
{
    std_srvs::srv::Trigger::Response response = std_srvs::srv::Trigger::Response();
    response.message = measurements.dump();
    if (response.message.empty())
    {
        response.success = false;
        response.message = "No measurements to upload";
        measurements_service->send_response(*header, response);
        return;
    }
    response.success = true;
    RCLCPP_DEBUG(get_logger(), "Cleaning up CVNode");
    cv_node.cleanup->async_send_request(request);
    RCLCPP_DEBUG(get_logger(), "Uploading measurements");
    measurements_service->send_response(*header, response);
}

bool CVNodeManager::execute_synthetic_inference_scenario()
{
    using namespace std::chrono;

    if (cvnode_request->input.empty())
    {
        RCLCPP_ERROR(get_logger(), "No input data to process");
        return false;
    }

    start = steady_clock::now();
    cvnode_future = cv_node.process->async_send_request(cvnode_request).future.share();
    cvnode_future.wait();
    cvnode_response = cvnode_future.get();
    cvnode_future = std::shared_future<SegmentCVNodeSrv::Response::SharedPtr>();
    if (!cvnode_response->success)
    {
        RCLCPP_ERROR(get_logger(), "Error while processing data");
        return false;
    }
    end = steady_clock::now();

    // Save measurements
    float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
    measurements.at("target_inference_step").push_back(duration);
    measurements.at("target_inference_step_timestamp")
        .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);
    return true;
}

bool CVNodeManager::execute_real_world_last_inference_scenario()
{
    using namespace std::chrono;

    if (cvnode_request->input.empty())
    {
        RCLCPP_ERROR(get_logger(), "No input data to process");
        return false;
    }

    /// Send inference request
    start = steady_clock::now();
    cvnode_future = cv_node.process->async_send_request(cvnode_request).future.share();
    cvnode_request->input.clear();
    if (cvnode_future.wait_for(milliseconds(get_parameter("inference_timeout_ms").as_int())) ==
        std::future_status::ready)
    {
        cvnode_response = cvnode_future.get();
        if (!cvnode_response->success)
        {
            RCLCPP_ERROR(get_logger(), "Error while processing data.");
            return false;
        }
    }
    end = steady_clock::now();

    // Measure inference time
    float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
    measurements.at("target_inference_step").push_back(duration);
    measurements.at("target_inference_step_timestamp")
        .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);

    // Reset future to make sure that it is not reused
    cvnode_future = std::future<SegmentCVNodeSrv::Response::SharedPtr>();
    return true;
}

bool CVNodeManager::execute_real_world_first_inference_scenario()
{
    using namespace std::chrono;

    if (!cvnode_future.valid())
    {
        if (cvnode_request->input.empty())
        {
            RCLCPP_ERROR(get_logger(), "No data to process");
            return false;
        }
        start = steady_clock::now();
        cvnode_future = cv_node.process->async_send_request(cvnode_request).future.share();
        cvnode_request->input.clear();
    }
    if (cvnode_future.wait_for(milliseconds(get_parameter("inference_timeout_ms").as_int())) ==
        std::future_status::ready)
    {
        cvnode_response = cvnode_future.get();
        if (!cvnode_response->success)
        {
            RCLCPP_ERROR(get_logger(), "Error while processing data");
            return false;
        }
        end = steady_clock::now();
        cvnode_future = std::shared_future<SegmentCVNodeSrv::Response::SharedPtr>();

        float duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
        measurements.at("target_inference_step").push_back(duration);
        measurements.at("target_inference_step_timestamp")
            .push_back(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 1000.0);
    }
    return true;
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
    RCLCPP_DEBUG(get_logger(), "[REGISTER] Registering the '%s' node", node_name.c_str());

    if (cv_node.process != nullptr)
    {
        response->message = "There is already a node registered";
        RCLCPP_WARN(
            get_logger(),
            "[REGISTER] Could not register the node '%s' because there is already a node registered",
            node_name.c_str());
        return;
    }

    // Create clients to communicate with the node
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr cvnode_prepare;
    if (!initialize_service_client<std_srvs::srv::Trigger>(request->prepare_srv_name, cvnode_prepare))
    {
        response->message = "Could not initialize the prepare service client";
        RCLCPP_ERROR(get_logger(), "[REGISTER] Could not initialize the prepare service client");
        return;
    }
    rclcpp::Client<SegmentCVNodeSrv>::SharedPtr cvnode_process;
    if (!initialize_service_client<SegmentCVNodeSrv>(request->process_srv_name, cvnode_process))
    {
        response->message = "Could not initialize the process service client";
        RCLCPP_ERROR(get_logger(), "[REGISTER] Could not initialize the process service client");
        return;
    }
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr cvnode_cleanup;
    if (!initialize_service_client<std_srvs::srv::Trigger>(request->cleanup_srv_name, cvnode_cleanup))
    {
        response->message = "Could not initialize the cleanup service client";
        RCLCPP_ERROR(get_logger(), "[REGISTER] Could not initialize the cleanup service client");
        return;
    }

    response->status = true;
    response->message = "The node is registered";
    cv_node = CVNode(node_name, cvnode_prepare, cvnode_process, cvnode_cleanup);

    cvnode_wait_cv.notify_one();
    RCLCPP_DEBUG(get_logger(), "[REGISTER] The node '%s' is registered", node_name.c_str());
}

void CVNodeManager::unregister_node_callback(
    const ManageCVNode::Request::SharedPtr request,
    [[maybe_unused]] ManageCVNode::Response::SharedPtr response)
{
    std::string node_name = request->node_name;

    RCLCPP_DEBUG(get_logger(), "[UNREGISTER] Unregistering the node '%s'", node_name.c_str());
    if (cv_node.name != node_name)
    {
        RCLCPP_WARN(get_logger(), "[UNREGISTER] The node '%s' is not registered", node_name.c_str());
        return;
    }

    cv_node = CVNode();
    RCLCPP_DEBUG(get_logger(), "[UNREGISTER] The node '%s' is unregistered", node_name.c_str());
    return;
}

} // namespace cvnode_manager

#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(cvnode_manager::CVNodeManager)
