# ROS2 CVNodeManager

Copyright (c) 2022-2023 [Antmicro](https://www.antmicro.com)

`CVNodeManager` is a ROS2 node responsible for managing inference testing scenarios and flow of data between `DataProvider` and tested `CVNode`.
It utilizes the [kenning_computer_vision_msgs](https://github.com/antmicro/ros2-kenning-computer-vision-msgs) package for communication with other nodes, and [GUINode](https://github.com/antmicro/ros2-gui-node) for data visualization.

## Building the CVNodeManager

Project dependencies:

* [ROS2 Humble](https://docs.ros.org/en/humble/index.html)
* [kenning_computer_vision_msgs](https://github.com/antmicro/ros2-kenning-computer-vision-msgs)
* [GuiNode](https://github.com/antmicro/ros2-gui-node) [optional]

The `CVNodeManager` defines two targets that can be built separately:

* `cvnode_manager` - the main node responsible for managing inference testing scenarios.
* `cvnode_manager_gui` - a GUI node an only purpose is to visualize the input data and results of the inference testing.

To build the `CVNodeManager` node, run the following command from the root of the repository:
```bash
colcon build --packages-select cvnode_manager
```

Use `BUILD_GUI` cmake option to build the `cvnode_manager_gui` target as well:
```bash
colcon build --packages-select cvnode_manager --cmake-args ' -DBUILD_GUI=ON'
```

Those targets can later be used and executed through ROS2 launch files.
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
Possible values are `true` and `false` and defaults to `false`.
* `preserve_output` - Defines whether the node should preserve the output data of the testing scenario.
Used only for `Real-World` scenarios where it defines behavior of the `CVNodeManager` node when the timeout is reached.
Possible values are `true` and `false` and defaults to `false` which means that the node will respond empty results when the timeout is reached.
* `inference_timeout_ms` - Defines the timeout in milliseconds for the `Real-World` scenarios.
This parameter is ignored for `Synthetic` scenario and defaults to `1000`.

All parameters can be set via command line arguments or via a launch configuration file.

Also, communication with the `CVNodeManager` node is done via ROS2 services and action:

* `cvnode_register` - Registers a new `CVNode` for inference testing via `kenning_computer_vision_msgs.srv.ManageCVNode` service.
* `cvnode_prepare` - Prepares registered `CVNode` for inference testing and utilizes `std_srvs.srv.Trigger` service type.
* `cvnode_process` - Processes a single batch of data via `kenning_computer_vision_msgs.action.SegmentationAction` action.
* `cvnode_measurements` - Returns the measurements of the inference testing and performs cleanup by utilizing `std_srvs.srv.Trigger` service type.

All services and action names can be remapped inside the launch file or via command line if needed.

## Testing scenarios

The `CVNodeManager` node introduces two families of testing scenarios:

* `Synthetic` - This family is represented by a single testing scenario called `synthetic` and is better suited for inference testing where accuracy is the main concern.
This scenario is designed to test the inference accuracy of the node under test by passing a single batch of data to the node and measuring the inference results.
* `Real-World` - This family is represented by two testing scenarios called `real_world_first` and `real_world_last` and is better suited for inference testing where latency is the main concern.
This scenario is designed to test inference accuracy of the node under test with strictly specified latency by passing batches of data to the node and setting a timeout for the inference results.
Timeout also serves as a frame rate limiter as output data is only passed to the `DataProvider` node in the pipeline after the timeout reached.

The difference between `real_world_first` and `real_world_last` is that the `first` scenario will always try to finish processing the initial batch of data, while the `last` scenario will always try to finish processing the last batch of data.
