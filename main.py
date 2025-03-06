import sys
from pathlib import Path
import os
import streamlit as st

# Set page config as the first Streamlit command
#st.set_page_config(layout="wide")

# Dynamically find the project root
project_root = Path(__file__).resolve().parent
#st.write("Project root:", project_root)

# Add `src` to the Python path
src_path = project_root / "src"
#st.write("src_path:", src_path)

if src_path.exists() and str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Print the sys.path to verify the src directory is added
#st.write("sys.path:", sys.path)

# Print the current working directory to verify the correct path
#st.write("Current working directory:", os.getcwd())

# Print the contents of the src directory to verify the correct path
#st.write("Contents of src directory:", os.listdir(src_path))

# Import and run the frontend script
try:
    import frontend
    st.write("Successfully imported frontend.py")
except ImportError as e:
    st.write(f"Error importing frontend.py: {e}")




