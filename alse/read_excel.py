import torch, pandas

def read_excel(file_path, x_names, y_names):
                                                        # NaN to 0
    consolidated_data = pandas.read_excel(f"{file_path}").fillna(0).loc
    # Input parameters
    input_param = []
    for xname in x_names:
        input_param.append(torch.tensor(consolidated_data[:, f"{xname}"]))

    # Output parameters
    output_param = []
    for yname in y_names:
        output_param.append((torch.tensor((consolidated_data[:, f"{yname}"]))).unsqueeze(-1))

    X = torch.stack(tuple(input_param), -1)
    return X, *output_param

