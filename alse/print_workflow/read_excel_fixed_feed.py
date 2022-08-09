import torch, pandas


def parse_data(raw_data, param):
    d = torch.tensor([])
    for i in range(len(raw_data)):
        d = torch.cat((d, torch.tensor(raw_data[i][:, f"{param}"])))
    return d


# Right now I am hardcoding 3 excels, we can change that in the future
def read_excel():
    ############### Save this section for later ##############

    # path = []
    # for i in range(3):
    #     path.append(input(f"Path to excel No. {i+1}: "))

    ############### Save this section for later ##############
    print("Name for the excel sheets are hardcoded in this version")
    path1 = "../test_data/8_4_data.xlsx"
    # path2 = "../test_data/4340_3mm_5.24gmin.xlsx"
    # path3 = "../test_data/4340_3mm_10.47gmin.xlsx"
    path = [path1]
    consolidated_data = []
    consolidated_data.append(pandas.read_excel(f"{path[0]}").loc)
    # Input parameters
    power = parse_data(consolidated_data, "P")
    velocity = parse_data(consolidated_data, "V")
    # Hardcoded
    # ones = torch.ones(40)
    # print("Spot size and feed rate are hardcoded in this version")
    # spot_size = torch.cat((ones*2.5, ones*3, ones*3))
    # feed_rate = torch.cat((ones*5.24, ones*5.24, ones*10.47))
    # Output parameters
    # width = parse_data(consolidated_data, "widths avg (mm)").unsqueeze(-1)
    pow_cap = (parse_data(consolidated_data, "powder_cap") / 100).unsqueeze(-1)
    width = (parse_data(consolidated_data, "width")).unsqueeze(-1)
    height = (parse_data(consolidated_data, "height")).unsqueeze(-1)
    wth = width / height
    wth = torch.nan_to_num(wth, nan=torch.rand(1).item())
    # _height = parse_data(consolidated_data, "heights avg (mm)").bool().long().unsqueeze(-1)
    # whratio = (parse_data(consolidated_data, "width")/parse_data(consolidated_data, "heights avg (mm)")).unsqueeze(-1)
    # adhere = _height
    # width_to_height = width/height.unsqueeze(-1)

    X = torch.stack((power, velocity), -1)
    return X, width, pow_cap, wth
