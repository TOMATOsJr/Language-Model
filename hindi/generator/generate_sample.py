import importlib.util
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def _load_local_module(module_name: str, relative_name: str):
    module_path = os.path.join(CURRENT_DIR, relative_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {relative_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


reasoning = _load_local_module("hi_sample_reasoning", "reasoning.py")
renderer = _load_local_module("hi_sample_renderer", "renderer.py")

generate_reasoning_example = reasoning.generate_reasoning_example
render_sample = renderer.render_sample


def main() -> None:
    domains = ["height", "age", "price", "weight", "quantity", "distance"]
    for index, domain in enumerate(domains):
        example = generate_reasoning_example(domain, seed=100 + index)
        rendered = render_sample(example)
        print(domain, rendered)


if __name__ == "__main__":
    main()
