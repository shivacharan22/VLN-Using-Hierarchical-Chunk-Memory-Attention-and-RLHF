import MatterSim
import time
import math
import numpy as np
import pandas as pd
from PIL import Image
import bz2
import pickle
import _pickle as cPickle
WIDTH = 800
HEIGHT = 600
VFOV = math.radians(60)
sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(False)
sim.setBatchSize(1)
sim.initialize()

data_con = pd.DataFrame(columns = ['scan_id', 'viewpoint_id', 'image', 'language', 'action'])
df = pd.read_json('R2R_train.json')
scanisls = df['scan'].unique()
for scanis in scanisls:
    view_pointls = df.loc[df['scan'] == scanis]
    for idx,(view_point_ids,instr) in enumerate(zip(view_pointls['path'],view_pointls['instructions'])):
        for i in range(len(view_point_ids)-1):
            sim.newEpisode([scanis], [view_point_ids[i]], [0], [0])
            state = sim.getState()[0]
            rgb = np.array(state.rgb, copy=False)
            image = rgb[:, :, ::-1]
            locations = state.navigableLocations
            for j in range(len(locations)):
                if locations[j].viewpointId == view_point_ids[i+1]:
                    data_con = data_con.append({'scan_id': scanis, 'viewpoint_id': view_point_ids[i], 'image': image, 'language': instr[2], 'action': 1}, ignore_index=True)
                    print({'scan_id': scanis, 'viewpoint_id': view_point_ids[i], 'image': image, 'language': instr[2], 'action': 1})
                    break
                else:
                    data_con = data_con.append({'scan_id': scanis, 'viewpoint_id': view_point_ids[i], 'image': image, 'language': instr[2], 'action': 0}, ignore_index=True)
                    print({'scan_id': scanis, 'viewpoint_id': view_point_ids[i], 'image': image, 'language': instr[2], 'action': 0})

def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
    
compressed_pickle('Tar_data', data_con) 