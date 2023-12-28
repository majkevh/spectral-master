import numpy as np
import os
import h5py
import pandas as pd
from itertools import product, combinations
import gudhi as gd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import logging
from spectral.signals import Signals
from sklearn.metrics import accuracy_score
from typing import Callable, Any, Dict, List, Tuple
import shutil
from scipy.sparse import csgraph
from scipy.io import loadmat
from scipy.linalg import eigh

def load_and_prepare_data(graph_folder: str, graph_dtypes: List[str], filtrations: List[str]) -> Dict:
    """
    Loads and prepares data from graph files. 

    Parameters:
    graph_folder (str): Path to the folder containing graph files.
    graph_dtypes (List[str]): List of graph data types.
    filtrations (List[str]): List of filtrations.

    Returns:
    Dict: A dictionary of all diagrams.
    """
    num_elements = len(os.listdir(os.path.join(graph_folder, "mat")))
    array_indices = np.arange(num_elements)
    all_diags = {}

    for dtype, gid, filt in product(graph_dtypes, array_indices, filtrations):
        all_diags[(dtype, filt, gid)] = load_csv_to_array(
            os.path.join(graph_folder, f"diagrams/{dtype}/graph_{gid:06}_filt_{filt}.csv")
        )

    return all_diags, array_indices

def extract_features_from_graphs(graph_folder: str, 
                                    all_diagrams: Dict, 
                                    signals_objects: Dict[Tuple[str, str], Any]) -> List[Dict]:
    """
    Extracts Persistence Signals features from graphs stored in a specified folder. 

    Parameters:
    graph_folder (str): Path to the folder containing graph files.
    all_diagrams (Dict): Dictionary of persistence diagrams.
    signals_objects (Dict[Tuple[str, str], Any]): Dictionary of Persistence Signals objects for feature extraction.
    verbose (bool, optional): If True, prints information about the process. Defaults to False.

    Returns:
    List[Dict]: A list of dictionaries, each representing a feature with its corresponding properties.

    Raises:
    FileNotFoundError: If the specified graph folder does not exist.
    Exception: For any other errors encountered during the feature extraction process.
    """
    features = []

    try:
        for graph_name in os.listdir(os.path.join(graph_folder, "mat")):
            name = graph_name.split("_")
            label = int(name[name.index("lb")+1])
            gid = int(name[name.index("gid") + 1]) - 1

            for (data_type, filt), signal_obj in signals_objects.items():
                diag_features = signal_obj(all_diagrams[(data_type, filt, gid)])
                for idx_center, value in enumerate(diag_features):
                    features.append({
                        "index": gid, 
                        "type": f"{data_type}-{filt}", 
                        "center": idx_center, 
                        "value": value, 
                        "label": label
                    })

        return features

    except FileNotFoundError:
        raise FileNotFoundError(f"Graph folder not found: {graph_folder}")
    except Exception as e:
        raise Exception(f"An error occurred in extract_features_from_graphs: {e}")




def predict_and_evaluate(learner: Any, 
                         test_features: Any, 
                         score: Callable = accuracy_score, 
                         verbose: bool = False) -> float:
    """
    Predicts and evaluates the performance of a given learning model. 

    Parameters:
    learner (Any): The learning model to be used for prediction.
    test_features (Any): The test data features.
    score (Callable, optional): The scoring function to evaluate the model. Defaults to accuracy_score.
    verbose (bool, optional): If True, prints the evaluation score. Defaults to False.

    Returns:
    float: The evaluation score of the model on the test data.

    Raises:
    Exception: If an error occurs during prediction or evaluation.
    """
    try:
        x_test = test_features.apply(lambda _: _["value"].values)
        y_test = LabelEncoder().fit_transform(
            test_features.apply(lambda _: _["label"].values[0])
        )
        y_test_pred = learner.predict(list(x_test))
        test_score = score(y_test, y_test_pred)

        if verbose:
            print(f"  (Test score) {score.__name__}: {test_score:.2f}")

        return test_score

    except Exception as e:
        raise Exception(f"An error occurred in predict_and_evaluate: {e}")


def fit_and_evaluate_model(learner: Any, 
                           train_features: Any, 
                           folder:int,
                           score: Callable = accuracy_score, 
                           verbose: bool = False) -> Any:
    """
    Fits the given learner to the training data and optionally evaluates its performance.

    Parameters:
    learner (Any): The learning model to be fitted.
    training_features (Any): The training data features.
    score (Callable, optional): The scoring function to evaluate the model after fitting. Defaults to accuracy_score.
    verbose (bool, optional): If True, prints additional information about the training process. Defaults to False.

    Returns:
    Any: The fitted learner.

    Raises:
    Exception: If an error occurs during the fitting process.
    """
    try:
        x_train = train_features.apply(lambda _: _["value"].values)
        y_train = LabelEncoder().fit_transform(
            train_features.apply(lambda _: _["label"].values[0])
        )
        learner.fit(list(x_train), y_train)

        if verbose:
            y_train_pred = learner.predict(list(x_train))
            train_score = score(y_train, y_train_pred)
            unique_sizes, counts = np.unique(list(map(len, x_train)), return_counts=True)
            print(f"Fold {folder}, embedding has size:", dict(zip(unique_sizes, counts)))
            print(f"  (Train score) {score.__name__}: {train_score:.2f}")

        return learner

    except Exception as e:
        raise Exception(f"An error occurred in fit_and_evaluate_model: {e}")


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
    for each generated orbit. 

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
    It supports filtering specific types of filtrations and provides verbose output. 

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



def evaluate_classifier_orbits(data: np.ndarray, labels: np.ndarray, n_runs: int = 10, test_size: float = 0.3, verbose: bool = True) -> tuple:
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




def load_csv_to_array(file_name: str) -> np.ndarray:
    """
    Loads a CSV file into a numpy array.

    Parameters:
    file_name (str): The name of the file to be loaded.

    Returns:
    np.ndarray: A 2D numpy array containing the data from the CSV file. 
                Returns a default array [[0, 0]] if the CSV is empty or has only headers.
    
    Raises:
    FileNotFoundError: If the specified file does not exist.
    ValueError: If the file is not a valid CSV file.
    """
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data_array = np.loadtxt(file_name, delimiter=',', ndmin=2)
            if not data_array.any():
                data_array = np.array([[0, 0]])
            
            if w:
                for warning in w:
                    logging.warning(str(warning.message))

        return data_array

    except FileNotFoundError:
        logging.error(f"File not found: {file_name}")
        raise

    except ValueError as e:
        logging.error(f"Error reading the file {file_name}: {e}")
        raise


def evaluate_model(dataset, alg, grid, waw, graph_folder, all_diags, array_indices, length, sampling, graph_dtypes, filtrations, repeat, verbose):
    """
    Evaluate the model based on the provided algorithm and parameters. 

    Other parameters: Contextual parameters for model evaluation.

    Returns:
    None
    """
    precisions = []
    for _ in range(repeat):
        objs = create_signal_objects(graph_dtypes, filtrations, dataset, alg, waw, grid)
        np.random.shuffle(array_indices)
        test_scores = perform_cross_validation(objs, graph_folder, all_diags, array_indices, length, sampling, graph_dtypes, filtrations, verbose)
        precisions.append(np.mean(test_scores))

    print(f"DATA: {dataset}, ALG: {alg}, GRID: {grid}, MEAN: {np.mean(precisions)}, MAX: {np.max(precisions)}, STD: {np.std(precisions)}, WAVE: {waw}")

def create_signal_objects(graph_dtypes, filtrations, dataset, alg, waw, grid):
    """
    Create signal objects based on the provided parameters.

    Other parameters: Parameters needed for signal object creation.

    Returns:
    Dictionary of signal objects.
    """
    maxima = {"PROTEINS":(0.9093726873397827, 0.9093728065490723),
        "DHFR":(0.9076708555221558, 0.9088095426559448),
        "MUTAG":(0.9071017503738403, 0.9086105227470398),
        "COX2":(0.9076706171035767, 0.908805251121521),
        "IMDB-BINARY": (0.9061643481254578, 0.9063771963119507),
        "IMDB-MULTI": (0.9063586592674255, 0.9063833355903625)}
    objs = {}
    for dtype, filt in product(graph_dtypes, filtrations):
        objs[(dtype, filt)] = Signals(global_max=maxima[dataset], function=alg, wave=waw, resolution=grid)
    return objs

def perform_cross_validation(objs, graph_folder, all_diags, array_indices, length, sampling, graph_dtypes, filtrations, verbose):
    """
    Perform cross-validation on the dataset.

    Other parameters: Parameters needed for cross-validation.

    Returns:
    List of test scores for each fold.
    """
    test_scores = []
    for k in range(10):
        test_indices = array_indices[np.arange(start=k * length, stop=(k + 1) * length)]
        train_indices = np.setdiff1d(array_indices, test_indices)

        for dtype, filt in product(graph_dtypes, filtrations):
            objs[(dtype, filt)].fit([all_diags[(dtype, filt, gid)] for gid in train_indices])

        feats = pd.DataFrame(extract_features_from_graphs(graph_folder=graph_folder, all_diagrams=all_diags, signals_objects=objs),
                             columns=["index", "type", "center", "value", "label"])

        fitted_learner = fit_and_evaluate_model(learner=RandomForestClassifier(n_estimators=100),
                                                    train_features=feats[np.isin(feats[sampling], train_indices)].groupby([sampling]),
                                                    verbose=verbose, folder = k+1)
        test_score = predict_and_evaluate(learner=fitted_learner,
                                              test_features=feats[np.isin(feats[sampling], test_indices)].groupby([sampling]), verbose=verbose)
        test_scores.append(test_score)
    return test_scores


def build_filtered_simplex(A, filtration_val, edge_threshold=0):
    """
    Constructs a filtered simplex tree from a given adjacency matrix.

    Parameters:
    A (numpy.ndarray): Adjacency matrix representing the graph.
    filtration_val (numpy.ndarray): Filtration values to be assigned to vertices.
    edge_threshold (float): Threshold value for edges to be included in the simplex.

    Returns:
    gd.SimplexTree: The constructed filtered simplex tree.
    """
    num_vertices = A.shape[0]
    st = gd.SimplexTree()
    [st.insert([i], filtration=-1e10) for i in range(num_vertices)]
    for i, j in combinations(range(num_vertices), r=2):
        if A[i, j] > edge_threshold:
            st.insert([i, j], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    return st


def compute_extended_persistence(dataset, filtrations):
    """
    Computes Extended Persistence for a set of graphs. 

    Parameters:
    dataset (str): Name of the dataset containing the graph files.
    filtrations (list): List of filtration values to be used in TDA.

    Outputs diagrams for each graph in the dataset using specified filtrations.
    """
    graph_folder = "./data/" + dataset + "/"
    diag_repo = graph_folder + "diagrams/"
    if os.path.exists(diag_repo) and os.path.isdir(diag_repo):
        shutil.rmtree(diag_repo)
    [os.makedirs(diag_repo + dtype) for dtype in [""] +  ["dgmOrd0", "dgmExt0", "dgmRel1", "dgmExt1"]]

    pad_size = 1
    for graph_name in os.listdir(graph_folder + "mat/"):
        A = np.array(loadmat(graph_folder + "mat/" + graph_name)["A"], dtype=np.float32)
        pad_size = np.max((A.shape[0], pad_size))

    print("Dataset:", dataset)
    print("Number of observations:", (len(os.listdir(graph_folder + "mat/"))))
    for graph_name in os.listdir(graph_folder + "mat/"):
        A = np.array(loadmat(graph_folder + "mat/" + graph_name)["A"], dtype=np.float32)
        name = graph_name.split("_")
        gid = int(name[name.index("gid") + 1]) - 1
        
        egvals, egvectors = eigh(csgraph.laplacian(A, normed=True))
        for filtration in filtrations:
            time = float(filtration.split("-")[0])
            filtration_val = np.square(egvectors).dot(np.diag(np.exp(-time * egvals))).sum(axis=1)
            st = build_filtered_simplex(A, filtration_val)
            st.extend_filtration()
            dgms = st.extended_persistence(min_persistence=1e-5)
            [np.savetxt(diag_repo + "%s/graph_%06i_filt_%s.csv" % (dtype, gid, filtration), [pers[1] for pers in diag],
                        delimiter=',')
             for diag, dtype in zip(dgms, ["dgmOrd0", "dgmExt0", "dgmRel1", "dgmExt1"])]
    
    