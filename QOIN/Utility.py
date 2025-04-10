import copy
import pickle
import warnings

from quri_parts.circuit.noise import DepolarizingNoise, NoiseInstruction, ThermalRelaxationNoise
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)
import random
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import ktrain
from ktrain import tabular

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
#from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.circuit.random import random_circuit
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from qiskit import qasm2
from qiskit.quantum_info import hellinger_distance
from lightgbm import LGBMClassifier, early_stopping
from quri_parts.qiskit.circuit.qiskit_circuit_converter import circuit_from_qiskit
import quri_parts.circuit.gate_names as gate_names
import quri_parts.circuit.noise as quri_noise
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
from quri_parts.qulacs.sampler import create_qulacs_density_matrix_concurrent_sampler

qurnoises = [
    # Single qubit noise
    quri_noise.BitFlipNoise(
        error_prob=0.004,
        qubit_indices=[0, 2],  # Qubit 0 or 2
        target_gates=[gate_names.H, gate_names.CNOT],  # H or CNOT gates
    ),
    quri_noise.DepolarizingNoise(
        error_prob=0.001,
        qubit_indices=[],  # All qubits
        target_gates=[gate_names.X, gate_names.CNOT,gate_names.Identity,gate_names.SqrtX, gate_names.RZ]  # X or CNOT gates
    ),
    quri_noise.PhaseFlipNoise(
        error_prob=0.002,
        qubit_indices=[1, 0],  # Qubit 0 or 1
        target_gates=[]  # All kind of gates
    ),
    quri_noise.BitPhaseFlipNoise(
        error_prob=0.001,
        qubit_indices=[],  # All qubits
        target_gates=[],  # All kind of gates
    ),

    # Multi qubit noise
    quri_noise.PauliNoise(
        pauli_list=[[1, 2], [2, 3]],
        prob_list=[0.001, 0.002],
        qubit_indices=[1, 2],  # 2 qubit gates applying to qubits (1, 2) or (2, 1)
        target_gates=[gate_names.CNOT]  # CNOT gates
    ),

    # Circuit noise
    quri_noise.DepthIntervalNoise([quri_noise.PhaseFlipNoise(0.001)], depth_interval=5),
    quri_noise.MeasurementNoise([quri_noise.BitFlipNoise(0.004), quri_noise.DepolarizingNoise(0.003)]),
]


qurnoises2 = [
    # Single qubit noise
    quri_noise.DepolarizingNoise(
        error_prob=0.02,
        qubit_indices=[],
        target_gates=[gate_names.Identity, gate_names.SqrtX, gate_names.X, gate_names.RZ],  # H or CNOT gates
    ),
    quri_noise.DepolarizingNoise(
        error_prob=0.1,
        qubit_indices=[],
        target_gates=[gate_names.CNOT],  # CNOT gates
    )
]


quri_noise_model = quri_noise.NoiseModel(qurnoises2)
quri_noise_ideal = quri_noise.NoiseModel()
quri_noise_sim = create_qulacs_density_matrix_concurrent_sampler(quri_noise_model)
quri_ideal_sim = create_qulacs_density_matrix_concurrent_sampler(quri_noise_ideal)

def convert_to_quri(qc):
    index = 0
    # iterate until all reset gates are removed if more than one gate was added
    while (index != -1):
        index = -1
        for i,q in enumerate(qc):
            if q.operation.name=='reset':
                index = i
        if index!=-1:
            del qc.data[index]

    qc.remove_final_measurements()

    try:
        circuit_from_qiskit(qc)
    except Exception as e:
        print(qc.draw())
    return circuit_from_qiskit(qc)


def add_measurements(QC, qubits_to_measure, basis):
    qc = QC.copy()
    if isinstance(qubits_to_measure,int):

        if basis=="X":
            qc.h(qubits_to_measure)
        elif basis=="Y":
            qc.sdg(qubits_to_measure)
            qc.h(qubits_to_measure)

        qc.measure(qubits_to_measure,qubits_to_measure)
    else:
        for m in qubits_to_measure:
            if basis=="X":
                qc.h(m)
            elif basis=="Y":
                qc.sdg(m)
                qc.h(m)
            qc.measure(m,m)
    return qc

def add_measurements_removal(QC):
    qc = QC.copy()
    qc.remove_final_measurements()
    qc.measure_all()
    return qc

def execute_circuit(
        qc,
        ideal_run=True,
        noisy_run=True,
        shots=1024,
        noise_model=None
):
    """
    Run a given quantum circuit qc on a simulator.
    - If ideal_run is True, run on an ideal simulator.
    - If noisy_run is True, run on a simulator with noise_model.
    - If both are True, return a dict with {'ideal': ..., 'noisy': ...}.
    - If only one is True, return the corresponding counts.
    """

    # Validate input
    if not ideal_run and not noisy_run:
        raise ValueError("At least one of ideal_run or noisy_run must be True.")

    # Prepare placeholders for results
    ideal_counts = None
    noisy_counts = None

    # Run on ideal simulator if requested
    if ideal_run:
        sim_ideal = AerSimulator()  # No noise model
        result_ideal = sim_ideal.run(qc, shots=shots).result()
        ideal_counts = result_ideal.get_counts()

    # Run on noisy simulator if requested
    if noisy_run:
        sim_noise = AerSimulator(noise_model=noise_model)
        result_noisy = sim_noise.run(qc, shots=shots).result()
        noisy_counts = result_noisy.get_counts()

    # Return appropriately based on user's request
    if ideal_run and noisy_run:
        return {"ideal": ideal_counts, "noisy": noisy_counts}
    elif ideal_run:
        return ideal_counts
    else:
        return noisy_counts
        
        
def execute_circuit_quri(qc,ideal_run=True,noisy_run=True,shots=1024):
    
    """
    This function is responsible for handling circuit execution in both ideal and noisy environment
    """
    
    if ideal_run and not noisy_run:
        if isinstance(qc,list):
            circ_tnoise = [(convert_to_quri(c),shots) for c in qc]
            counts = quri_ideal_sim(circ_tnoise)
            ideal_counts = []
            for _counts,qub in zip(counts, qc):
                qubits = qub.num_qubits
                ideal_counts.append(dict([(bin(k)[2::].zfill(qubits),v) for k,v in _counts.items()]))
            
        else:
            qubits = qc.num_qubits
            circ_tnoise = convert_to_quri(qc)
            ideal_counts = quri_ideal_sim([(circ_tnoise ,shots)])[0]
            ideal_counts = dict([(bin(k)[2::].zfill(qubits),v) for k,v in ideal_counts.items()])
        return ideal_counts
    
    elif not ideal_run and noisy_run:
        if isinstance(qc,list):
            circ_tnoise = [(convert_to_quri(c),shots) for c in qc]
            counts = quri_noise_sim(circ_tnoise)
            noise_counts = []
            for _counts,qub in zip(counts, qc):
                qubits = qub.num_qubits
                noise_counts.append(dict([(bin(k)[2::].zfill(qubits),v) for k,v in _counts.items()]))
            
        else:
            qubits = qc.num_qubits
            circ_tnoise = convert_to_quri(qc)
            noise_counts = quri_noise_sim([(circ_tnoise ,shots)])[0]
            noise_counts = dict([(bin(k)[2::].zfill(qubits),v) for k,v in noise_counts.items()])
        return noise_counts
    
    elif ideal_run and noisy_run:
        if isinstance(qc,list):
            circ_tnoise = [(convert_to_quri(c),shots) for c in qc]
            counts = quri_ideal_sim(circ_tnoise)
            ideal_counts = []
            for _counts,qub in zip(counts, qc):
                qubits = qub.num_qubits
                ideal_counts.append(dict([(bin(k)[2::].zfill(qubits),v) for k,v in _counts.items()]))
            
        else:
            qubits = qc.num_qubits
            circ_tnoise = convert_to_quri(qc)
            ideal_counts = quri_ideal_sim([(circ_tnoise ,shots)])[0]
            ideal_counts = dict([(bin(k)[2::].zfill(qubits),v) for k,v in ideal_counts.items()])
            
        
        if isinstance(qc,list):
            circ_tnoise = [(convert_to_quri(c),shots) for c in qc]
            counts = quri_noise_sim(circ_tnoise)
            noise_counts = []
            for _counts,qub in zip(counts, qc):
                qubits = qub.num_qubits
                noise_counts.append(dict([(bin(k)[2::].zfill(qubits),v) for k,v in _counts.items()]))
            
        else:
            qubits = qc.num_qubits
            circ_tnoise = convert_to_quri(qc)
            noise_counts = quri_noise_sim([(circ_tnoise ,shots)])[0]
            noise_counts = dict([(bin(k)[2::].zfill(qubits),v) for k,v in noise_counts.items()])
        
        return {"ideal":ideal_counts,"noisy":noise_counts}
    else:
        raise Exception("Not Implemented")
        

def calculate_features(noisy_count: list, ideal_count: list = None):
    
    """
    This function converts noisy counts to feature data
    if ideal counts are given then it will also add the ground truth for each state
    """
    
    if isinstance(noisy_count,list):
        
        total_noisy_runs = defaultdict(list)
        for d in noisy_count:
            for key, value in d.items():
                total_noisy_runs[key].append(value/sum(d.values()))
        
        feature_result = {}
        
        for each_state in total_noisy_runs:
            median_prob = np.round(np.median(total_noisy_runs[each_state]),3)
            odds = [v/(1-v) if v != 1 else 1 for v in total_noisy_runs[each_state]]
            median_odds = np.round(np.median(odds),3)
            probf = [1-v for v in total_noisy_runs[each_state]]
            median_probf = np.round(np.median(probf),3)
            feature_result[each_state] = {"POS":median_prob,"ODR":median_odds,"POF":median_probf}


        if ideal_count!=None:
            total_ideal_runs = defaultdict(list)
            for d in ideal_count:
                for key, value in d.items():
                    total_ideal_runs[key].append(value/sum(d.values()))

            for each_state in feature_result:
                if each_state in total_ideal_runs.keys():
                    feature_result[each_state]["ideal_prob"] = np.round(np.median(total_ideal_runs[each_state]),3)
                else:
                    feature_result[each_state]["ideal_prob"] = -1
                
        return feature_result

    else:
        raise Exception("Unknow type for counts")


def input_generation_for_QOIN(property_list: list,size=12000):
    """
    This function generates required inputs and respective circuits for given properties
    """
    
    All_input_list = []
    for _ in tqdm(range(size//len(property_list))):
        for one_property in property_list:

            property_obj = one_property()
            MAX_ATTEMPTS = 100

            input_generators = property_obj.get_input_generators()
            
            found = False

            for attempt_idx in range(MAX_ATTEMPTS):
                # we need as many seeds as input generators,
                seeds = tuple(random.randint(0, 2 ** 31 - 1) for _ in input_generators)

                # use the seeds to generate the inputs
                inputs = [generator.generate(seeds[i]) for i, generator in enumerate(input_generators)]

                # check the preconditions
                if property_obj.preconditions(*inputs):
                    # next for loop iteration
                    found=True
                    break

            if found:
                qubits_to_measure_and_circuits = property_obj.operations(*inputs)
                All_input_list.append(qubits_to_measure_and_circuits)
        
    return All_input_list


        
def QOIN_data_generation(property_list: list, case_study_name, basis="Z", datasize=12000):
    """
    This function generates feature data for QOIN training from input circuits for a given basis
    
    Values for basis is Z,X,and Y
    
    """
    data = pd.DataFrame(columns=["POS","ODR","POF","ideal_prob"])
    
    All_input_list = input_generation_for_QOIN(property_list,datasize)
    
    for single_input in tqdm(All_input_list):
        qubits_to_measure = single_input[0]
        circuit_to_consider = single_input[1]
        
        qc = add_measurements(circuit_to_consider, qubits_to_measure, basis)
        
        execution_result = execute_circuit([qc for _ in range(10)],ideal_run=True,noisy_run=True)
        features = calculate_features(execution_result["noisy"],execution_result["ideal"])
        
        for state in features:
            data = data._append(features[state], ignore_index = True)
    
    data.to_csv(f"QOIN/data/{case_study_name}-{basis}.csv",index=False)
    print(f"data saved at QOIN/data/{case_study_name}-{basis}.csv")


def QOIN_data_generation_file(path, case_study_name, noise_model=None):
    """
    This function generates feature data for QOIN training from input circuits for a given basis

    Values for basis is Z,X,and Y

    """
    print("data generation file")
    PATH = os.path.dirname(os.path.abspath(""))

    files = os.listdir(path)

    data = pd.DataFrame(columns=["POS", "ODR", "POF", "ideal_prob"])

    qcs = []
    for file in files:
        print(f"loading {file}")
        with open(path + "/" + file, 'rb') as f:
            qcs.extend(pickle.load(f))

    for qc in tqdm(qcs):
        execution_result = execute_circuit([copy.deepcopy(qc) for _ in range(10)], ideal_run=True, noisy_run=True,
                                           noise_model=noise_model)

        features = calculate_features(execution_result["noisy"], execution_result["ideal"])

        for state in features:
            data = data._append(features[state], ignore_index=True)

    data.to_csv(f"{PATH}/QOIN/data/{case_study_name}-Z.csv", index=False)
    print(f"data saved at QOIN/data/{case_study_name}-Z.csv")


def QOIN_data_generation_file_quri(path,case_study_name,shots=1024):
    """
    This function generates feature data for QOIN training from input circuits for a given basis
    
    Values for basis is Z,X,and Y
    
    """

    PATH = os.path.dirname(os.path.abspath(""))

    files = os.listdir(path)
    
    data = pd.DataFrame(columns=["POS","ODR","POF","ideal_prob"])
    
    All_input_list = [path+"/"+x for x in files]

    print(All_input_list)
    
    for single_input in tqdm(All_input_list):
  
        qc = qasm2.load(single_input,custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

        try:
            execution_result_ideal = execute_circuit_quri([copy.deepcopy(qc) for _ in range(10)],ideal_run=True,noisy_run=False,shots=shots)
            execution_result_noisy = execute_circuit_quri([copy.deepcopy(qc) for _ in range(10)],ideal_run=False,noisy_run=True,shots=shots)

            features = calculate_features(execution_result_noisy,execution_result_ideal)

            for state in features:
                data = data._append(features[state], ignore_index = True)
        except Exception as e:
            print(e)
            print("exception")
            print(single_input)
    
    data.to_csv(f"{PATH}/QOIN/data/{case_study_name}-Z.csv",index=False)
    print(f"data saved at QOIN/data/{case_study_name}-Z.csv")


def Qoin_train(case_study_name,basis):
    
    """
    This function trains the  model for given file and saves it in given path
    """
    PATH = os.path.dirname(os.path.abspath(""))

    data = pd.read_csv(f"{PATH}/QOIN/data/{case_study_name}-{basis}.csv")
    trn, val, preproc = tabular.tabular_from_df(data, is_regression=True, 
                                                 label_columns='ideal_prob', random_state=42,verbose=0)
    model = tabular.tabular_regression_model('mlp', trn)
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=128)
    learner.autofit(1e-3,early_stopping=10,reduce_on_plateau=2)
    ktrain.get_predictor(learner.model, preproc).save(f"{PATH}/QOIN/models/{case_study_name}-{basis}")
    print(f"model saved to QOIN/models/{case_study_name}-{basis}")


def Qoin_train_clf(case_study_name, basis):
    PATH = os.path.dirname(os.path.abspath(""))

    data = pd.read_csv(f"{PATH}/QOIN/data/{case_study_name}-{basis}.csv")
    regression_data = copy.deepcopy(data)
    data.loc[data["ideal_prob"] != -1, "ideal_prob"] = 0
    data.loc[data["ideal_prob"] == 0, "ideal_prob"] = 1

    y = data["ideal_prob"]
    data.drop(["ideal_prob"], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42,stratify=y)

    gbmr = LGBMClassifier()
    gbmr.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             callbacks=[early_stopping(1000)])

    preds = gbmr.predict(data)
    regression_data = regression_data.loc[preds == 1]
    y = regression_data["ideal_prob"]
    regression_data.drop(["ideal_prob"], axis=1, inplace=True)

    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        'n_estimators': 10000,
        'learning_rate': 0.09,
        'num_leaves': 240,
        'max_depth': 12,
        'min_data_in_leaf': 400,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'feature_fraction': 0.3,
        "verbose": -1
    }

    reg = LinearRegression()
    reg.fit(regression_data, y)
    preds = reg.predict(regression_data)
    print(mean_absolute_error(preds, y))

    y_pred = gbmr.predict(X_test)
    mse = accuracy_score(y_pred, y_test)

    with open(f"{PATH}/QOIN/models/clf{case_study_name}.models", 'wb') as f:
        pickle.dump({"c": gbmr, "r": reg}, f)

    return {"satus": f"QOIN/models/clf{case_study_name}.models", "regression_error": mse}

def Qoin_load(case_study_name,basis):
    PATH = os.path.dirname(os.path.abspath(""))
    print(PATH)
    predictor = ktrain.load_predictor(f"{PATH}/QOIN/models/{case_study_name}-{basis}")
    return predictor

def Qoin_load_clf(case_study_name):
    PATH = os.path.dirname(os.path.abspath(""))
    print(PATH)
    with open(f"{PATH}/QOIN/models/clf{case_study_name}.models", 'rb') as f:
        model = pickle.load(f)
    return model


def get_filtered_result(model,qubits_to_measure,QC,basis):
    
    qc = add_measurements(QC, qubits_to_measure, basis)

    noisy_counts = execute_circuit([qc for _ in range(10)],ideal_run=False,noisy_run=True)
    feature_dict = calculate_features(noisy_counts)
    df = pd.DataFrame.from_dict(feature_dict).T.reset_index()
    preds = model.predict(df[["POS","ODR","POF"]])
    preds[preds < 0] = 0
    preds[preds>1] = 1
    preds = preds/np.sum(preds)
    df["preds"] = np.round(preds,3)
    return dict(zip(df['index'], df['preds']))


def filtered_result(model, qc, basis="Z", noise_model=None):
    noisy_counts = execute_circuit([qc for _ in range(10)], ideal_run=False, noisy_run=True, noise_model=noise_model)
    feature_dict = calculate_features(noisy_counts)
    df = pd.DataFrame.from_dict(feature_dict).T.reset_index()

    print("before predict")
    print(df)
    preds = model.predict(df[["POS", "ODR", "POF"]], verbose=0)
    print("raw preds")
    print(preds)
    preds[preds < 0] = 0
    preds[preds > 1] = 1
    print("preds")
    print(preds)
    preds = preds / np.sum(preds)
    df["preds"] = np.round(preds, 3)
    return dict(zip(df['index'], df['preds']))


def filtered_result_clf(model, qc, noise_model=None):
    noisy_counts = execute_circuit([qc for _ in range(10)], ideal_run=False, noisy_run=True, noise_model=noise_model)
    feature_dict = calculate_features(noisy_counts)
    df = pd.DataFrame.from_dict(feature_dict).T.reset_index()

    clf_df = copy.deepcopy(df)
    reg = model["r"]
    clf = model["c"]
    clf_df.drop(["index"], axis=1, inplace=True)
    c_preds = clf.predict(clf_df,num_iteration=clf.best_iteration_)
    if df.loc[c_preds==1].shape[0]>0:
        df = df.loc[c_preds == 1]
        states = df["index"].values
        df.drop(["index"], axis=1, inplace=True)
        r_preds = reg.predict(df[["POS","ODR","POF"]])
        result = {}
        for i in range(len(r_preds)):
            if float(r_preds[i])<0:
                pass
            elif float(r_preds[i])>1:
                result[states[i]] = 0.8
            else:
                result[states[i]] = float(r_preds[i])

        result = dict([(k,round(v/sum(result.values()),ndigits=4)) for k,v in result.items()])

        # if results dict empty because all neg, return noisy
        if result:
            pass
        else:
            for s,v in zip(states,df['POS'].values):
                result[s] = float(v)

    else:
        result = {}
        # checks if all predictions unlikely, if true returns noisy
        if df[df["POS"]<=0.01].shape[0]==df.shape[0]:
            for i in range(df.shape[0]):
                result[df.loc[i, "State"]] = float(df.loc[i, "POS"])
        else:
            for i in range(df.shape[0]):
                if df.loc[i,"POS"] > 0.01:
                    result[df.loc[i,"State"]] = float(df.loc[i,"POS"])

        result = dict([(k, round(v / sum(result.values()), ndigits=4)) for k, v in result.items()])
    if result:
        n = 1
    else:
        n = 2
    return result


def filtered_result_quri(model, qc, basis="Z"):
    noisy_counts = execute_circuit_quri([qc for _ in range(10)], ideal_run=False, noisy_run=True)
    feature_dict = calculate_features(noisy_counts)
    df = pd.DataFrame.from_dict(feature_dict).T.reset_index()
    preds = model.predict(df[["POS", "ODR", "POF"]])
    preds[preds < 0] = 0
    preds[preds > 1] = 1
    preds = preds / np.sum(preds)
    df["preds"] = np.round(preds, 3)
    return dict(zip(df['index'], df['preds']))

def filtered_result_clf_quri(model, qc,shots=1024):
    noisy_counts = execute_circuit_quri([qc for _ in range(10)], ideal_run=False, noisy_run=True,shots=shots)
    feature_dict = calculate_features(noisy_counts)
    df = pd.DataFrame.from_dict(feature_dict).T.reset_index()

    clf_df = copy.deepcopy(df)
    reg = model["r"]
    clf = model["c"]
    clf_df.drop(["index"], axis=1, inplace=True)
    c_preds = clf.predict(clf_df,num_iteration=clf.best_iteration_)
    if df.loc[c_preds==1].shape[0]>0:
        df = df.loc[c_preds == 1]
        states = df["index"].values
        df.drop(["index"], axis=1, inplace=True)
        r_preds = reg.predict(df[["POS","ODR","POF"]])
        #r_preds = df["POS"].values
        result = {}
        for i in range(len(r_preds)):
            if float(r_preds[i])<0:
                pass
            elif float(r_preds[i])>1:
                result[states[i]] = 0.8
            else:
                result[states[i]] = float(r_preds[i])

        result = dict([(k,round(v/sum(result.values()),ndigits=4)) for k,v in result.items()])

    else:
        result = {}
        for i in range(df.shape[0]):
            if df.loc[i,"POS"] > 0.01:
                result[df.loc[i,"State"]] = float(df.loc[i,"POS"])

        result = dict([(k, round(v / sum(result.values()), ndigits=4)) for k, v in result.items()])

    return result


def get_filtered_result_file(model,name):
    qc = qasm2.load(name,custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    
    
    #qc = add_measurements_removal(qc)
    
    noisy_counts = execute_circuit([qc for _ in range(10)],ideal_run=False,noisy_run=True)
    feature_dict = calculate_features(noisy_counts)
    df = pd.DataFrame.from_dict(feature_dict).T.reset_index()
    preds = model.predict(df[["POS","ODR","POF"]])
    preds[preds < 0] = 0
    preds[preds>1] = 1
    preds = preds/np.sum(preds)
    df["preds"] = np.round(preds,3)
    return dict(zip(df['index'], df['preds']))


def hellinger_distance(dist_p, dist_q):

    p_normed = {}
    for key, val in dist_p.items():
        p_normed[key] = val

    q_normed = {}
    for key, val in dist_q.items():
        q_normed[key] = val

    total = 0
    for key, val in p_normed.items():
        if key in q_normed.keys():
            total += (np.sqrt(val) - np.sqrt(q_normed[key])) ** 2
            del q_normed[key]
        else:
            total += val
    total += sum(q_normed.values())

    dist = np.sqrt(total) / np.sqrt(2)

    return dist

def qiskit_hellinger(dist_p,dist_q):
    return hellinger_distance(dist_p,dist_q)