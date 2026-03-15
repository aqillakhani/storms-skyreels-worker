"""Create stub modules for torchcodec and torchao to avoid import crashes."""
import site, os

sp = site.getsitepackages()[0]

# torchcodec stub
p = os.path.join(sp, "torchcodec")
os.makedirs(p, exist_ok=True)
with open(os.path.join(p, "__init__.py"), "w") as f:
    f.write("# stub\n")

# torchao stub with quantization submodule
p = os.path.join(sp, "torchao")
os.makedirs(p, exist_ok=True)
with open(os.path.join(p, "__init__.py"), "w") as f:
    f.write("# stub\n")

q = os.path.join(p, "quantization")
os.makedirs(q, exist_ok=True)
with open(os.path.join(q, "__init__.py"), "w") as f:
    f.write("def float8_weight_only(*a, **k): pass\ndef quantize_(*a, **k): pass\n")

print("stubs created: torchcodec, torchao")
