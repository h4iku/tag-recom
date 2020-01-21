# How to run:

1. Install [Python 3.6+](https://www.python.org/).

2. Clone this repository and create a venv:
    ```
    git clone https://github.com/h4iku/tag-recom.git
    cd tag-recom
    python -m venv env
    ./env/Scripts/activate
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
    
4. Download the datasets from [here](https://www.mediafire.com/file/bpc0h4uateua899/data.zip/file), and unzip them in the root directory of the cloned repository:

    ```
    tag-recom/data
    ├── apple
    ├── askubuntu
    ├── codereview
    ├── dba
    ├── serverfault
    ├── softwareengineering
    ├── stackoverflow
    ├── stats
    ├── superuser
    ├── tex
    └── wordpress
    ```
    
5. Run the main module:

    ```
    cd tag_recommender
    python main.py
    ```

    Change the value of the `DATASET` variable in `datasets.py` to choose different datasets. There are also some boolean flags in `main.py` to control the execution of different parts of the program.
