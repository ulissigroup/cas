import torch, pandas

def parse_data(raw_data, param):
    d = torch.tensor([])
    for i in range(len(raw_data)):
        d = torch.cat((d, torch.tensor(raw_data[i][:,f"{param}"])))
    return d

#Right now I am hardcoding 3 excels, we can change that in the future
def read_excel(path):
    path = []
    for i in range(3):
        path.append(input(f"Path to excel No. {i+1}: "))
    consolidated_data = []
    for i in range(3):
        consolidated_data.append(pandas.read_excel(f'{path[i]}').loc)    
    #Input parameters
    power = parse_data(consolidated_data, "P (W)")
    velocity = parse_data(consolidated_data, "V (mm/min)")
    # Hardcoded
    ones = torch.ones(40)
    print("Reminder: spot size and feed rate are hardcoded in this version")
    spot_size = torch.cat((ones*2.5, ones*3, ones*3))
    feed_rate = torch.cat((ones*5.24, ones*5.24, ones*10.47))
    #Output parameters
    width = parse_data(consolidated_data, "widths avg (mm)").unsqueeze(-1)
    pow_cap = (parse_data(consolidated_data, "powder capt %")/100).unsqueeze(-1)
    _height = parse_data(consolidated_data, "heights avg (mm)").bool().long().unsqueeze(-1)
    adhere = torch.cat(([_height]*4),-1)

    X = torch.stack((power, velocity, spot_size, feed_rate), -1)
    return X, width, pow_cap, adhere