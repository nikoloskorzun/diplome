from PyEMD import EMD

def get_EMD(series):
    try:
        emd = EMD() #for my baggy enviroment
    except Exception:
        emd = EMD.EMD()

    return emd(series)