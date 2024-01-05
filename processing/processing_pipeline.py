import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from multiprocessing import Pool

from processing.operations import Operation, get_aggregation, decode_signal


class AccelerometerSignalProcessingPipeline:
    def __init__(self, operation_ids, aggregation_id):
        self.list_of_operations = operation_ids
        self.aggregation_id = aggregation_id
        self.aggregation = get_aggregation(aggregation_id)

        self.pipe = [Operation(op_id) for op_id in self.list_of_operations]

    def __str__(self):
        s = ""
        for op in self.pipe:
            s += str(op)
        if len(s) == 0:
            s += "raw"
        s += "-{}".format(self.aggregation_id)
        return s

    def process_signal(self, signal):
        for op in self.pipe:
            signal = op.compute(signal)
        return self.aggregation(signal)

    def compute_df(self, df):
        y = []
        for _, row in df.iterrows():
            if row["raw_accelerometer_signal"] is np.nan:
                acc_raw = np.zeros(6)
            else:
                acc_raw = np.fromstring(row["raw_accelerometer_signal"], dtype=np.float32, sep=",")
            y.append(self.process_signal(acc_raw))
        return y
    

def execute(task):
    aspp, df = task
    return str(aspp), aspp.compute_df(df)
    

class GeneralASPP:
    def __init__(self, operations_to_consider, aggergations_to_consider, complexity: int):
        self.complexity = complexity

        param_space = {"op_{}".format(i+1): operations_to_consider for i in range(complexity)}
        param_space["aggregations"] = aggergations_to_consider
        self.configs = list(ParameterGrid(param_space))
    
    def __len__(self):
        return len(self.configs)
    
    def create(self):
        print("[INFO] Generating gASPP-{}...".format(self.complexity))
        list_of_aspp = []
        for conf in self.configs:
            operation_ids = [conf["op_{}".format(i+1)] for i in range(self.complexity)]
            aspp = AccelerometerSignalProcessingPipeline(operation_ids, conf["aggregations"])
            list_of_aspp.append(aspp)
        return list_of_aspp
    
    def compute_df(self, df):
        list_of_aspp = self.create()
        print("[INFO] Evaluating...")
        task_list = [[aspp, df] for aspp in list_of_aspp]
        with Pool() as p:
            results = p.map(execute, task_list)
        processed = {aspp_id: y for aspp_id, y in results}
        return processed
