from pprint import pprint

from . import hellaswag
from . import lambada
from . import piqa
from . import openbookqa
from . import copa
from . import storycloze
from . import arc
from . import csqa
from . import boolq
from . import rte
from . import cb
from . import sst
from . import trec
from . import agn
from . import lama

TASK_REGISTRY = {
    "storycloze": storycloze.StoryCloze,
    "csqa": csqa.CommonsenseQA,
    "rte": rte.RTE, 
    "cb": cb.CommitmentBank, 
    'sst2': sst.SST2,
    'sst5': sst.SST5,
    'trec': trec.TREC, 
    'agn': agn.AGNews,
    "boolq": boolq.BoolQ,
    "copa": copa.COPA,
    "lambada": lambada.LAMBADA,
    "piqa": piqa.PIQA,
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    "hellaswag": hellaswag.HellaSwag,
    "openbookqa": openbookqa.OpenBookQA,
    "lama": lama.LAMA
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as e:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise NotImplementedError("Task Not Supported Yet: {}".format(task_name))
        # raise KeyError(f"Missing task {task_name}")
        

def get_task_dict(task_name_list):
    return {
        task_name: get_task(task_name)()
        for task_name in task_name_list
    }
