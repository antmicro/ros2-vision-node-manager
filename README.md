# ROS 2 CVNodeManager

Copyright (c) 2022-2023 [Antmicro](https://www.antmicro.com)

`CVNodeManager` is a ROS 2 node responsible for managing nodes running computer vision algorithms.

It performs:

* switching between nodes to collect predictions,
* evaluating the predictions of computer vision models.

It utilizes the [kenning_computer_vision_msgs](https://github.com/antmicro/ros2-kenning-computer-vision-msgs) package for communication with other nodes, and [GUINode](https://github.com/antmicro/ros2-gui-node) for data visualization.

## Building the CVNodeManager

Project dependencies:

* [ROS 2 Humble](https://docs.ros.org/en/humble/index.html)
* [kenning_computer_vision_msgs](https://github.com/antmicro/ros2-kenning-computer-vision-msgs)
* [GuiNode](https://github.com/antmicro/ros2-gui-node) (optional)

The `CVNodeManager` defines two composable nodes:

* `cvnode_manager` - the main node responsible for managing inference scenarios and testing.
* `cvnode_manager_gui` - visualizes the input data and results of the inference testing.

To build the `CVNodeManager` node, run the following command from the root of the repository:

```bash
colcon build --packages-select cvnode_manager
```

Use `BUILD_GUI` CMake option to build the `cvnode_manager_gui` target as well:

```bash
colcon build --packages-select cvnode_manager --cmake-args ' -DBUILD_GUI=ON'
```

Those composable nodes can later be wrapped in `ComposableNodeContainer` and started, e.g. with ROS 2 launch files.
For example, to run the `CVNodeManager` node, define a launch file with the following content:

```python
from launch_ros.actions import ComposableNodeContainer
...
cvnode_manager_node = ComposableNodeContainer(
    name='cvnode_manager_container',
    namespace='',
    package='rclcpp_components',
    executable='component_container',
    composable_node_descriptions=[
        ComposableNode(
            package='cvnode_manager',
            plugin='cvnode_manager::CVNodeManager',
            name='cvnode_manager_node',
            parameters=[{
                'publish_visualizations': LaunchConfiguration('publish_visualizations'),
                'inference_timeout_ms': LaunchConfiguration('inference_timeout_ms'),
                'scenario': LaunchConfiguration('scenario'),
                'preserve_output': LaunchConfiguration('preserve_output'),
            }],
        )
    ],
    output='both',
)
```

Similarly, to run the `CVNodeManagerGUI` node, add next lines to the launch file:
```python
cvnode_manager_gui_node = ComposableNodeContainer(
    name='cvnode_manager_gui_container',
    namespace='',
    package='rclcpp_components',
    executable='component_container',
    composable_node_descriptions=[
        ComposableNode(
            package='cvnode_manager',
            plugin='cvnode_manager::gui::CVNodeManagerGUI',
            name='cvnode_manager_gui_node',
        ),
    ],
    output='both',
)
```

## Configuration

`CVNodeManager` defines the following parameters:

* `scenario` - Defines the testing scenario to be used. Possible values are `synthetic`, `real_world_first` and `real_world_last`.
  Default value is `synthetic`.
  More about testing scenarios can be found in the [Testing scenarios](#testing-scenarios) section.
* `publish_results` - Defines whether the node should publish the input data and results of the testing scenario.
  Possible values are `true` and `false`, default value is `false`.
* `preserve_output` - Defines whether the node should preserve the output data of the testing scenario.
  Used only for `Real-World` scenarios where it defines behavior of the `CVNodeManager` node when the timeout is reached.
  Possible values are `true` and `false`.
  The default value is `false` which means that the node will respond empty results when the timeout is reached.
* `inference_timeout_ms` - Defines the timeout in milliseconds for the `Real-World` scenarios.
  This parameter is ignored for `Synthetic` scenario.
  The default value is `1000`.

All parameters can be set via command line arguments or via a launch configuration file.

Also, communication with the `CVNodeManager` node is done via ROS 2 services and action:

* `cvnode_register` - Registers a new `CVNode` for inference testing via `kenning_computer_vision_msgs.srv.ManageCVNode` service.
* `cvnode_prepare` - Prepares registered `CVNode` for inference testing and utilizes `std_srvs.srv.Trigger` service type.
* `cvnode_process` - Processes a single batch of data via `kenning_computer_vision_msgs.action.SegmentationAction` action.
* `cvnode_measurements` - Returns the measurements of the inference testing and performs cleanup by utilizing `std_srvs.srv.Trigger` service type.

All services and action names can be remapped inside the launch file or via command line if needed.

## Testing scenarios

The `CVNodeManager` node introduces two families of testing scenarios:

* `Synthetic` - In this approach, `CVNodeManager` sends input data to a given topic, and waits for results before sending new data.
  This approach tells how model performs on sequences of data when no time constraints are present.
* `Real-world` - In this approach, the input data is sent to a given topic at a fixed rate.
  If the model is unable to deliver predictions, either the previous predictions are used, or empty predictions are returned.
  This demonstrates how predictions of the overall system are affected by time constraints.
  There are two variants of the scenario:
  * `real_world_first` will always try to finish processing the initial batch of data,
  * `real_world_last` will always try to finish processing the last batch of data.
