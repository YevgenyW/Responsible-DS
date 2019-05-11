# Responsible-DS
This repo created for responsible data science project.

### Project structure
- `model`: contains all files related to model(s)
- `src`: contains all files relates to framework

### Environment installation
Create conda environment and install dependencies:
`conda env create -f environment.yml`

Install dependencies into an existing environment:
`conda env update -n responsible-ds -f environment.yml --prune`

In order to add a package to `environment.yml` you need to install it locally and run `conda env export > environment.yml`.

### Other
For more details about conda virtual environment:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Useful article how setup this repo in Google Colab:
https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c
