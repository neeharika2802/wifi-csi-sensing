"""
step11_build_nlos.py - Step 11a: build the through-wall (NLOS) dataset.

Uses exactly the same pipeline and settings as step8_build_dataset.py, only the
folder is different. The result is saved as data/processed/NLOS_PIFA_cut12.npz.
"""
import step8_build_dataset as b

b.SCENARIO = "NLOS"     # folder name inside data/raw/wallhack1.8k (Windows ignores upper/lower case)
b.DATA = b.ROOT / "data" / "raw" / "wallhack1.8k" / b.SCENARIO / b.SYSTEM

if not b.DATA.exists():
    parent = b.ROOT / "data" / "raw" / "wallhack1.8k"
    print(f"Folder not found: {b.DATA}")
    if parent.exists():
        print(f"Folders inside {parent}: {sorted(p.name for p in parent.iterdir())}")
        scen = parent / b.SCENARIO
        if scen.exists():
            print(f"Folders inside {scen}: {sorted(p.name for p in scen.iterdir())}")
    raise SystemExit("Send me the lines above and I will fix the folder names.")

b.main()