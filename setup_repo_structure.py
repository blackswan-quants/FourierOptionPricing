import os
import json
import subprocess

# === CONFIG ===
members = [
    ("galante-implementation", "Achille Galante"),
    ("bottan-implementation", "Bottan"),
    ("coppetti-implementation", "Coppetti"),
    ("lorello-implementation", "Lorello"),
    ("piana-validation", "Davide Piana")
]

base_dirs = [
    "src",
    "notebooks",
    "tests"
]

src_files = [
    "characteristic_functions.py",
    "fourier_integral.py",
    "fft_pricer.py",
    "validation.py"
]

notebook_files = [
    "sensitivity_study.ipynb",
    "analytical_validation.ipynb"
]

test_files = [
    "test_pricer.py"
]

config_data = {
    "S0": 100,
    "r": 0.05,
    "sigma": 0.2,
    "T": 1.0,
    "alpha": 1.25,
    "eta": 0.25,
    "N": 4096
}

# === FUNCTIONS ===
def create_structure():
    print("📁 Creating folder structure...")
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)

    for f in src_files:
        with open(os.path.join("src", f), "w") as file:
            file.write(f"# Placeholder for {f} — Fourier Option Pricing Project (Sprint 2)\n")

    for f in notebook_files:
        with open(os.path.join("notebooks", f), "w") as file:
            file.write(f"# Notebook placeholder: {f}\n")

    for f in test_files:
        with open(os.path.join("tests", f), "w") as file:
            file.write(f"# Tests placeholder: {f}\n")

    with open("config.json", "w") as f:
        json.dump(config_data, f, indent=4)

    print("✅ Folder structure and files created successfully.")


def create_branches():
    print("\n🌿 Creating branches for each member...")
    for branch, name in members:
        print(f"→ Creating branch: {branch} ({name})")
        subprocess.run(["git", "checkout", "main"])
        subprocess.run(["git", "checkout", "-b", branch])
        subprocess.run(["git", "push", "-u", "origin", branch])
    print("✅ All branches created and pushed to remote.")


# === MAIN ===
if __name__ == "__main__":
    create_structure()

    create_branches_option = input("\nDo you want to create Git branches for each member? (y/n): ").strip().lower()
    if create_branches_option == "y":
        create_branches()
    else:
        print("Skipped branch creation. You can still run it later.")