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

# K9 behaviour-tree shell — blackboard increment

This increment preserves the visible behaviour-tree hierarchy and adds a
central, namespaced blackboard containing the state required by the current K9
executive design.

## Files

- `k9_bt_shell.py` — the visible tree and ROS node.
- `k9_blackboard.py` — canonical keys, states, schema, defaults and typed access.

## Install

```bash
cp k9_bt_shell.py \
  ~/k9_ws/src/k9_bt_pkg/k9_bt_pkg/k9_bt_shell.py

cp k9_blackboard.py \
  ~/k9_ws/src/k9_bt_pkg/k9_bt_pkg/k9_blackboard.py

chmod +x \
  ~/k9_ws/src/k9_bt_pkg/k9_bt_pkg/k9_bt_shell.py
```

Your existing `setup.py` console entry remains:

```python
'k9_bt_shell = k9_bt_pkg.k9_bt_shell:main',
```

Build and run:

```bash
cd ~/k9_ws

source /opt/ros/jazzy/setup.bash

colcon build \
  --symlink-install \
  --packages-select k9_bt_pkg

source install/setup.bash

ros2 run k9_bt_pkg k9_bt_shell
```

## Inspect the blackboard

List every registered key:

```bash
py-trees-blackboard-watcher --list
```

Stream the complete blackboard:

```bash
py-trees-blackboard-watcher
```

Stream selected values:

```bash
py-trees-blackboard-watcher \
  /k9/system/status \
  /k9/audio/desired_mode \
  /k9/audio/effective_mode \
  /k9/dialogue/intent \
  /k9/chess/state
```

Include recent read/write activity:

```bash
py-trees-blackboard-watcher --activity
```

Include blackboard information in the tree watcher:

```bash
py-trees-tree-watcher -b
```

## Schema groups

The blackboard contains fundamental Python values beneath:

```text
/k9/system/*
/k9/safety/*
/k9/battery/*
/k9/audio/*
/k9/dialogue/*
/k9/speech/*
/k9/chess/*
/k9/expression/*
```

The shell initialises the system to:

```text
/k9/system/mode                    NORMAL
/k9/system/ready                   true
/k9/system/status                  RUNNING
/k9/safety/emergency_active        false
/k9/audio/desired_mode             WAITING_FOR_HOTWORD
/k9/audio/effective_mode           NOT_LISTENING
/k9/dialogue/state                 IDLE
/k9/dialogue/intent                NONE
/k9/speech/state                   IDLE
/k9/chess/state                    IDLE
/k9/expression/effective           NOT_LISTENING
```

## Design rules for later behaviours

Later behaviours should import `BlackboardKey` and attach their own client with
only the access they require:

```python
self.blackboard = self.attach_blackboard_client(
    name="Emergency Condition",
    namespace="k9",
)
self.blackboard.register_key(
    key=BlackboardKey.SAFETY_EMERGENCY_BUTTON_PRESSED,
    access=py_trees.common.Access.READ,
)
```

Avoid hard-coded strings outside `k9_blackboard.py`. This makes misspelled keys
less likely and keeps the watcher output stable while the implementation grows.
