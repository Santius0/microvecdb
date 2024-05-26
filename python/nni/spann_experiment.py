import nni

def evaluate_spann(spann_params: dict):
    return 0.0


if __name__ == "__main__":
    params = nni.get_next_parameter()
    print(f"Running with parameters: {params}")
    performance = evaluate_spann(params)
    print(f"Hyperparameters: {params}")
    print(f"Final Performance: {performance}")
    nni.report_final_result(performance)