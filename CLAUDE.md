# CLAUDE.md - Coding Guidelines for NTT_Hackathon

## Python Commands
- Run Jupyter notebook: `jupyter notebook *.ipynb`
- Run Streamlit dashboard: `streamlit run challenge_two/dashboard.py`
- Install dependencies: `pip install pandas numpy streamlit plotly torch scikit-learn`

## Code Style Guidelines
- **Imports**: Group in order: core libraries (pandas, numpy) → utility libraries → visualization → ML libraries
- **Formatting**: 4-space indentation, 120 character line limit
- **Naming**: 
  - Constants: UPPERCASE (e.g., WORKDIR, DOWNSAMPLE)
  - Variables: snake_case (e.g., train_data, model_output)
  - Functions: snake_case with descriptive action verbs
- **Documentation**: Use docstrings for functions/classes, inline comments for complex logic
- **Data Analysis**: Label plots, include units, document data transformations
- **Error Handling**: Use try/except for data loading and model operations
- **Organization**: Keep data exploration in EDA directory, challenges in separate folders

## Project Structure
This repository contains data analysis for energy market challenges organized in separate directories for each challenge.