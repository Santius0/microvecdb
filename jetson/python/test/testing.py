import sys
# sys.path.append("/usr/lib/python3.6")
sys.path.append("/usr/lib/python3/dist-packages")

print(sys.path)

import numpy as np
import microvecdb as mvdb_c

mvdb_c.hello(3)