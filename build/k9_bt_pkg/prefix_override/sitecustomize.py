import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/hopkira/k9_ws/src/k9_bt_pkg/install/k9_bt_pkg'
