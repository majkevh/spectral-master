import numpy as np
import os
import h5py
import pandas as pd
import gudhi as gd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_orbit(num_pts_per_orbit: int, param: float) -> np.ndarray:
    """
    Generate a set of points representing an orbit based on a simple dynamic system.

    The function iteratively generates points in a 2D space using a modification of 
    the logistic map, a classic example of how complex, chaotic behaviour can arise from 
    very simple non-linear dynamical equations.

    Parameters:
    - num_pts_per_orbit (int): The number of points to generate for the orbit.
    - param (float): The parameter that controls the dynamics of the orbit. Typically
      a value between 0 and 5.

    Returns:
    - np.ndarray: A 2D array of shape (num_pts_per_orbit, 2) where each row represents 
      the x and y coordinates of a point in the orbit.

    Raises:
    - ValueError: If num_pts_per_orbit is not positive or if param is outside an expected range.
    """

    if num_pts_per_orbit <= 0:
        raise ValueError("num_pts_per_orbit must be a positive integer")
    if not (0 <= param <= 5):
        raise ValueError("param must be between 0 and 4")

    orbit_points = np.zeros([num_pts_per_orbit, 2])
    current_x, current_y = np.random.rand(), np.random.rand()

    for point_index in range(num_pts_per_orbit):
        current_x = (current_x + param * current_y * (1. - current_y)) % 1
        current_y = (current_y + param * current_x * (1. - current_x)) % 1
        orbit_points[point_index, :] = [current_x, current_y]

    return orbit_points


def compute_persistence(dataset: str) -> pd.DataFrame:
    """
    Generates persistence diagrams and features for a given dataset.

    This function creates persistence diagrams for different parameter values of a dynamical system 
    (orbit generation) and stores them in an HDF5 file. It also compiles a DataFrame of labels 
    for each generated orbit. Implementation based on PersLay https://github.com/MathieuCarriere/perslay.

    Parameters:
    - dataset (str): Name of the dataset, expected to be either "ORBIT5K" or "ORBIT100K".
    - path_dataset (str): The path where the dataset is stored. If empty, a default path is used.

    Returns:
    - pd.DataFrame: A DataFrame containing labels for each orbit.

    Raises:
    - AssertionError: If the dataset is not in the expected options.
    - FileNotFoundError: If the required directories do not exist.
    """

    assert dataset in ["ORBIT5K", "ORBIT100K"], "Dataset must be 'ORBIT5K' or 'ORBIT100K'"

    default_path = "./data/" + dataset + "/"

    hdf5_file_path = os.path.join(default_path , dataset + ".hdf5")
    if os.path.isfile(hdf5_file_path):
        os.remove(hdf5_file_path)

    diag_file = h5py.File(hdf5_file_path, "w")
    [diag_file.create_group(filtration) for filtration in ["H0", "H1"]]

    labels_data = []
    count = 0
    num_diag_per_param = 1000 if "5K" in dataset else 20000
    r_values = [2.5, 3.5, 4.0, 4.1, 4.3]

    for label, rho in enumerate(r_values):
        print(f"Generating {num_diag_per_param} dynamical particles and PDs for r = {rho}...")
        for _ in range(num_diag_per_param):
            orbit_points = generate_orbit(num_pts_per_orbit=1000, param=rho)
            alpha_complex = gd.AlphaComplex(points=orbit_points)
            simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=1e50)
            simplex_tree.persistence()

            for dim in range(2):  # Assuming we're interested in dimensions 0 and 1
                persistence_intervals = np.array(simplex_tree.persistence_intervals_in_dimension(dim))
                diag_file[f"H{dim}"].create_dataset(name=str(count), data=persistence_intervals)

            labels_data.append({"label": label, "pcid": count})
            count += 1

    labels_df = pd.DataFrame(labels_data)
    labels_df.set_index("pcid", inplace=True)
    labels_df[["label"]].to_csv(default_path +  dataset+ ".csv")


def get_data(dataset: str,  verbose: bool = True):
    """
    Loads data from a given dataset, including persistence diagrams and features.

    The function reads persistence diagrams stored in an HDF5 file and feature data from a CSV file.
    It supports filtering specific types of filtrations and provides verbose output. Implementation based on PersLay https://github.com/MathieuCarriere/perslay.

    Parameters:
    - dataset (str): The name of the dataset to load.
    - path_dataset (str): The path to the dataset directory. Uses a default path if empty.
    - filtrations (list): A list of filtrations to load. Loads all if empty.
    - verbose (bool): If True, prints detailed information about the loaded dataset.

    Returns:
    - tuple: A tuple containing the loaded persistence diagrams, labels, and number of data.

    Raises:
    - FileNotFoundError: If the dataset files do not exist.
    """

    default_path = os.path.join("./data", dataset)

    hdf5_file_path = os.path.join(default_path, dataset + ".hdf5")
    if not os.path.isfile(hdf5_file_path):
        raise FileNotFoundError(f"The HDF5 file for dataset '{dataset}' not found at '{hdf5_file_path}'")

    diag_file = h5py.File(hdf5_file_path, "r")

    diagrams_dict = {}
    for filtration in list(diag_file.keys()):
        diagrams = [np.array(diag_file[filtration][str(diag)]) for diag in range(len(diag_file[filtration].keys()))]
        diagrams_dict[filtration] = diagrams

    feature_file_path = os.path.join(default_path, dataset + ".csv")
    if not os.path.isfile(feature_file_path):
        raise FileNotFoundError(f"The feature file for dataset '{dataset}' not found at '{feature_file_path}'")

    feature_data = pd.read_csv(feature_file_path, index_col=0)
    labels = LabelEncoder().fit_transform(feature_data["label"])
    one_hot_labels = OneHotEncoder(sparse=False, categories="auto").fit_transform(labels[:, np.newaxis])

    if verbose:
        print(f"Dataset: {dataset}")
        print(f"Number of observations: {one_hot_labels.shape[0]}")
        print(f"Number of classes: {one_hot_labels.shape[1]}")

    get_id_class = lambda x:np.where(x==1)[0][0]
    labels = np.apply_along_axis(get_id_class, 1, one_hot_labels)
    return diagrams_dict, labels, one_hot_labels.shape[0]



def max_measures(data_dict: dict) -> tuple:
    """
    Finds the maximum x and y values across all arrays in the provided dictionary.

    The function iterates through each array in the dictionary and finds the maximum x and y values,
    considering only finite values for y.

    Parameters:
    - data_dict (dict): A dictionary where each key maps to a list of numpy arrays. Each array
      is expected to have two columns (representing x and y coordinates).

    Returns:
    - tuple: A tuple containing the maximum x value and the maximum y value found across all arrays.

    Raises:
    - ValueError: If any array in the dictionary does not have two columns.
    """

    max_x, max_y = float('-inf'), float('-inf')

    for arrays in data_dict.values():
        for array in arrays:
            if array.size == 0:
                continue

            if array.shape[1] != 2:
                raise ValueError("Each array must have exactly two columns representing x and y coordinates.")

            current_max_x = np.max(array[:, 0])
            current_max_y = np.max(array[:, 1][np.isfinite(array[:, 1])])

            max_x = max(max_x, current_max_x)
            max_y = max(max_y, current_max_y)

    return max_x, max_y



def evaluate_classifier_performance(data: np.ndarray, labels: np.ndarray, n_runs: int = 10, test_size: float = 0.3, verbose: bool = True) -> tuple:
    """
    Evaluates the performance of a Random Forest classifier on the given dataset.

    The function performs multiple runs of training and testing the classifier, 
    each time with a different split of training and testing data. It calculates 
    and returns the mean accuracy and standard deviation of these runs.

    Parameters:
    - data (np.ndarray): The feature data.
    - labels (np.ndarray): The labels corresponding to the data.
    - n_runs (int): The number of runs for training and testing the classifier. Default is 10.
    - test_size (float): The proportion of the dataset to include in the test split. Default is 0.3.
    - verbose (bool): If True, prints the accuracy of each run. Default is True.

    Returns:
    - tuple: A tuple containing the mean accuracy and standard deviation across all runs.

    Raises:
    - ValueError: If the test_size is not between 0 and 1.
    """

    if not (0 < test_size < 1):
        raise ValueError("test_size must be a value between 0 and 1.")

    classifier = RandomForestClassifier(n_estimators=100)
    all_runs_accuracy = []

    for run in range(n_runs):
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=test_size)
        classifier.fit(X_train, Y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions)
        all_runs_accuracy.append(accuracy)

        if verbose:
            print(f"Run {run + 1}: Accuracy = {accuracy}")

    mean_accuracy = np.mean(all_runs_accuracy)
    std_dev_accuracy = np.std(all_runs_accuracy)


    print(f"Overall Mean Accuracy across {n_runs} runs: {mean_accuracy}")
    print(f"Standard Deviation across {n_runs} runs: {std_dev_accuracy}")

    return mean_accuracy, std_dev_accuracy
