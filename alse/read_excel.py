import torch, pandas


def read_excel(file_path):

    consolidated_data = pandas.read_excel(f"{file_path}").loc
    # Input parameters
    power = torch.tensor(consolidated_data[:, "P (W)"])
    velocity = torch.tensor(consolidated_data[:, "V (mm/min)"])

    # Output parameters
    pow_cap = (torch.tensor(consolidated_data[:, "powder_cap"]) / 100).unsqueeze(-1)
    width = (torch.tensor(consolidated_data[:, "width (mm)"])).unsqueeze(-1)
    height = (torch.tensor(consolidated_data[:, "height (mm)"])).unsqueeze(-1)
    wth = width / height
    wth = torch.nan_to_num(wth, nan=torch.rand(1).item())

    X = torch.stack((power, velocity), -1)
    return X, width, pow_cap, wth
