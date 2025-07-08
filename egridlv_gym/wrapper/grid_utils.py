import os
import pickle
import copy
import simbench as sb
import pandapower as pp
import numpy as np
import pandas as pd


def get_simbench_grid():
    """
    Returns the simbench grid for the 1-LV-rural1--2-sw network.
    """
    try:
        base_path = os.path.join(os.path.dirname(__file__), "../../data")
        os.makedirs(base_path, exist_ok=True)
        file_path = os.path.join(base_path, "1-LV-rural1--2-sw.pkl")
        with open(file_path, "rb") as f:
            net = pickle.load(f)
        return net
    except FileNotFoundError:
        net = sb.get_simbench_net("1-LV-rural1--2-sw")
        with open(file_path, "wb") as f:
            pickle.dump(net, f)
        return net


def min_max_normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def extract_feeder(
    input_grid: pp.auxiliary.pandapowerNet, busses: list, keys: list
) -> pp.auxiliary.pandapowerNet:
    """
    Extracts the single feeder branch from the input grid.
    """
    grid = copy.deepcopy(input_grid)
    output_grid = pp.create_empty_network()
    for key in keys:
        if key == "bus":
            grid[key] = grid[key].drop(busses)
            grid[key] = grid[key].reset_index()
            output_grid[key] = copy.deepcopy(grid[key])
        elif key == "ext_grid":
            output_grid[key] = copy.deepcopy(grid[key])
        elif key == "storage":
            for bool_list in [bus != grid[key]["bus"] for bus in busses]:
                grid[key]["bus"] = grid[key][["bus"]].where(bool_list, np.nan, axis=0)
                grid[key] = grid[key].dropna(subset="bus")
            grid[key] = grid[key].reset_index(drop=True)
            grid[key]["bus"] = grid[key]["bus"].replace(
                grid["bus"]["index"].tolist(), grid["bus"].index.tolist()
            )
            grid[key]["bus"] = grid[key]["bus"].astype(int)
            output_grid[key] = copy.deepcopy(grid[key])
        elif key == "line":
            for bool_list in [bus != grid[key]["from_bus"] for bus in busses]:
                grid[key]["from_bus"] = grid[key][["from_bus"]].where(
                    bool_list, np.nan, axis=0
                )
                grid[key] = grid[key].dropna(subset="from_bus")
            grid[key] = grid[key].reset_index(drop=True)
            grid[key]["from_bus"] = grid[key]["from_bus"].replace(
                grid["bus"]["index"].tolist(), grid["bus"].index.tolist()
            )
            grid[key]["from_bus"] = grid[key]["from_bus"].astype(int)

            for bool_list in [bus != grid[key]["to_bus"] for bus in busses]:
                grid[key]["to_bus"] = grid[key][["to_bus"]].where(
                    bool_list, np.nan, axis=0
                )
                grid[key] = grid[key].dropna(subset="to_bus")
            grid[key] = grid[key].reset_index(drop=True)
            grid[key]["to_bus"] = grid[key]["to_bus"].replace(
                grid["bus"]["index"].tolist(), grid["bus"].index.tolist()
            )
            grid[key]["to_bus"] = grid[key]["to_bus"].astype(int)

            output_grid[key] = copy.deepcopy(grid[key])
        elif key == "trafo":
            grid[key]["lv_bus"] = grid[key]["lv_bus"].replace(
                grid["bus"]["index"].tolist(), grid["bus"].index.tolist()
            )
            grid[key]["lv_bus"] = grid[key]["lv_bus"].astype(int)
            output_grid[key] = copy.deepcopy(grid[key])
        elif key == "bus_geodata":
            grid[key] = grid[key].drop(busses)
            grid[key] = grid[key].reset_index(drop=True)
            output_grid[key] = copy.deepcopy(grid[key])
        else:
            for bool_list in [bus != grid[key]["bus"] for bus in busses]:
                grid[key]["bus"] = grid[key][["bus"]].where(bool_list, np.nan, axis=0)
                grid[key] = grid[key].dropna(subset="bus")
            grid[key] = grid[key].reset_index(drop=True)
            grid[key]["bus"] = grid[key]["bus"].replace(
                grid["bus"]["index"].tolist(), grid["bus"].index.tolist()
            )
            grid[key]["bus"] = grid[key]["bus"].astype(int)

            output_grid[key] = copy.deepcopy(grid[key])

    output_grid["bus"] = output_grid["bus"].drop(["index"], axis=1)
    return output_grid


def fix_bus_names(names: list) -> list:
    return [
        (name[: name.rfind(" ")] + " " + str(i) if not name.endswith(str(i)) else name)
        for i, name in enumerate(names)
    ]


def create_ts_dataframe(feeder_grid, element_type, ts_element_p):
    element_ids = [
        int(name.split()[-1]) for name in getattr(feeder_grid, element_type)["name"]
    ]
    index_to_id = {
        index: element_id - 1 for index, element_id in enumerate(element_ids)
    }
    id_to_index = {k: v for k, v in index_to_id.items()}
    return pd.DataFrame({i: ts_element_p[id_to_index[i]] for i in id_to_index})
