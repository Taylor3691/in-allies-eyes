def get_metadata(data):
    pids, cams = [], []
    for _, pid, camid in data:
        pids += [pid]
        cams += [camid]
    pids = set(pids)
    cams = set(cams)
    num_pids = len(pids)
    num_cams = len(cams)
    num_imgs = len(data)
    return num_pids, num_imgs, num_cams