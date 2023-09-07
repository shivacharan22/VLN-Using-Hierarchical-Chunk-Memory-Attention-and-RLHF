import MatterSim
import time
import math
import numpy as np
import pandas as pd
from PIL import Image
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
for l in scanisls:
    view_pointls = df.loc[df['scan'] == l]
    for idx,(view_point_ids,instr) in enumerate(zip(view_pointls['path'],view_pointls['instructions'])):
        for i in range(len(view_point_ids)-1):
            sim.newEpisode([l], [view_point_ids[i]], [0], [0])
            state = sim.getState()[0]
            rgb = np.array(state.rgb, copy=False)
            image = rgb[:, :, ::-1]
            locations = state.navigableLocations
            count = 0
            if len(locations) < 2:
                continue
            while locations[1].viewpointId != view_point_ids[i+1]:
                    count+=1
                    sim.makeAction([0], [1.0], [0])
                    state = sim.getState()[0]
                    locations = state.navigableLocations
                    if len(locations) < 2:
                        break
            if len(locations) < 2:
                    continue
            sim.makeAction([1], [0], [0])
            state = sim.getState()[0]
            rgb = np.array(state.rgb, copy=False)
            image = rgb[:, :, ::-1]
            locations = state.navigableLocations
            data_con = data_con.append({'scan_id': l, 'viewpoint_id': view_point_ids[i], 'image': image, 'language': instr[2], 'action': "right_{}".format(count)}, ignore_index=True)

data_con.to_csv('data_n.csv')