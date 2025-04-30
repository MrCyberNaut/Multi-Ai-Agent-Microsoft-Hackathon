"""Helper script to run the Streamlit app with proper Python module imports."""
import os
import sys
import subprocess
import importlib.util

def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit app."""
    # Check and install required packages
    required_packages = ["streamlit", "langgraph", "serpapi", "openai", "python-dotenv"]
    
    all_installed = True
    for package in required_packages:
        if not check_package(package):
            print(f"{package} not found, installing...")
            success = install_package(package)
            if not success:
                all_installed = False
                print(f"Failed to install {package}. Please install it manually with 'pip install {package}'")
    
    if not all_installed:
        print("\nSome packages could not be installed automatically.")
        print("Please run the following command to install all requirements:")
        print("pip install -r requirements.txt")
        return
    
    # Run the Streamlit app using Python module
    print("Starting Streamlit app...")
    try:
        subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        print("\nIf you see 'streamlit not found', try these steps:")
        print("1. Install streamlit: pip install streamlit")
        print("2. Make sure your Python environment is properly activated")
        print("3. Try running the app with: python -m streamlit run app.py")
        print("4. If using PowerShell, you might need to use: python -m streamlit run .\\app.py")

if __name__ == "__main__":
    run_streamlit_app() 