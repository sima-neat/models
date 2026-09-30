import sys

import pyneat

model = pyneat.Model(sys.argv[1])
model.benchmark(output_path=sys.argv[2])
