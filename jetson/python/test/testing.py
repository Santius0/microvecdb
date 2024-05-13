import sys
sys.path.append("/usr/lib/python3.6")
sys.path.append("/usr/lib/python3")

print(sys.path)

import microvecdb as mvdb_c

mvdb_c.hello(3)