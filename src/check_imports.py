import importlib

for name in ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "torch"]:
    try:
        m = importlib.import_module(name)
        print(f"OK       {name} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"BLOCKED  {name}: {str(e)[:120]}")