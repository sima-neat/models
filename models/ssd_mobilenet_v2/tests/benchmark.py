import sys

import pyneat

model = pyneat.Model(sys.argv[1])
report = model.benchmark()
report.save_json(sys.argv[2])
