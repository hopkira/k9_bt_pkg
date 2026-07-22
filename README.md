# K9 visible behaviour-tree shell

This is the first, deliberately non-functional increment of the K9 executive.

It contains the complete hierarchy but uses deterministic placeholders. It has
no K9 service clients, hardware access, speech calls, or blocking waits.

## Copy into the package

```bash
cp k9_bt_shell.py ~/k9_ws/src/k9_bt_pkg/k9_bt_pkg/k9_bt_shell.py
chmod +x ~/k9_ws/src/k9_bt_pkg/k9_bt_pkg/k9_bt_shell.py
```

## Fastest test: run it directly

```bash
cd ~/k9_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

python3 src/k9_bt_pkg/k9_bt_pkg/k9_bt_shell.py
```

If ROS 2 Jazzy was built or installed somewhere else, source that setup file
instead of `/opt/ros/jazzy/setup.bash`.

## View it

In a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/k9_ws/install/setup.bash

py-trees-tree-viewer
```

For a terminal view:

```bash
py-trees-tree-watcher /k9_bt_shell/snapshots
```

If your installed viewer discovers the tree dynamically, the topic argument is
not required:

```bash
py-trees-tree-watcher
```

## Add a ros2 run entry point

In `k9_bt_pkg/setup.py`, add this item inside `console_scripts`:

```python
'k9_bt_shell = k9_bt_pkg.k9_bt_shell:main',
```

Then rebuild:

```bash
cd ~/k9_ws
colcon build --symlink-install --packages-select k9_bt_pkg
source install/setup.bash
ros2 run k9_bt_pkg k9_bt_shell
```

## Expected active path

The initial active state is:

```text
K9 Root
├── Battery Supervisor
└── Safety Executive
    └── Normal Operation
        ├── Audio State Manager
        │   ├── Process Audio Events
        │   └── Maintain Effective Audio State
        │       └── Maintain NotListening
        ├── Dialogue Manager
        │   └── Dialogue Idle
        ├── Chess Manager
        │   └── Chess Idle
        └── Expression Manager
            └── NotListening Expression
```

The root and all four normal-operation managers remain RUNNING.
