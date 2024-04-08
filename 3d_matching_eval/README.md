# 3D Matching

## Data Preparation

Classes **columns**, **doors**, **walls** are accepted and will be parsed.

**Files format regex:**

`(?P<model>.*)_(?P<floor>.*)_(?P<classname>columns|doors|walls).json`.

**Examples:**

`MedOffice_F1_doors.json`, `MedOffice_F2_columns.json`, `Minas_Tirith_Outerwall_walls.json`.

All the files must be placed together in one directory or .zip archive.

**Notes:**

* Each JSON files is a List of structures (Dict).
* All the measures, locations and points must be measured in meters! Rotation parameter measures in degrees.
* All parameters are optional with 0 default value. You may not specify them

### Schemas

``` python
""" Columns.
    Have only one location point and 3D measures:
        Width - X,
        Depth - Y,
        Height - Z.
    Rotation parameter is used to rotation the structure around Z-axis."""
{
    "width": 0.3047,    # Optional,
    "depth": 0.2031,    # default
    "height": 2.7432,   # is 0
    "loc": [
        5.8154,         # X-axis
        9.8700,         # Y-axis
        0.0             # Z-axis
    ],
    "rotation": 0.0     # Degrees (Optional: default is 0)
}

""" Doors.
    The same schema as the one above.
    Translation and rotation parameters are optional."""
{
    "width": 1.0668,
    "depth": 0.1778,
    "height": 2.2097,
    "loc": [
        4.2008,
        17.7022,
        -0.911
    ],
    "rotation": 359.9999,
}

""" Walls.
    Consists of the two points and width, height dimensions.

        •-------------•
       /|            /|         z    y
      / |           / |         |   /
     /  |          /  |  height |  / width (around ep-st vector)
    •-------------• --•         | /
    |  /          |  /          |/
    | st----------|-ep          •----------x
    |/            |/               length (ep-st)
    •-------------•

    Height measure for both points is assumed to be the same.
    Translation parameters are optional.
"""
{
    "start_pt": [
        35.9333,
        22.5964,
        -0.9111
    ],
    "end_pt": [
        35.9330,
        17.7753,
        -0.9111
    ],
    "width": 0.2032,    # X or Y-axis respectively.
    "height": 3.9529,   # Z-axis
}
```

## Run in Docker

You may want to configure some parameters in `src/config.py` before buid.

```bash
# Clone the repository.
git clone git@github.com:cv4aec/3d-matching-eval.git
cd 3d-matching-eval

# Build Docker image.
docker build -t cvpr-2023-matching .

# Run image and evaluate.
# NOTE: Put all your JSON data in ./data/.
docker run \
    -v {full/path/to/ground-truth/directory/}:/data/ \
    -v {full/path/to/users-models/directory/}:/predicted/ \
    -v {full/path/to/output/directory/}:/code/output \
    -it 3d-matching-eval /bin/bash

python main.py ../data/{reference_model}.json ../predicted/{user_model}.json --output/match.json

# Example:
python main.py ../data/OfficeLab01_Allfloors_columns.json ../predicted/OfficeLab01_Allfloors_columns.json --output output/match.json
```

**Note**: you need to pass a directory path or a file path to the script.

* When the path is a (like `./data`) directory the script searches for all models inside one level deep.
* When the path is a file (like `*_columns.json`) the script uses regular expressions to extract the model name of the file. Then it searches for all the data for that model inside the directory. It means that if you pass `../data/OfficeLab01_Allfloors_columns.json` file to the script, it will search for all the files with `OfficeLab01` model name around.
