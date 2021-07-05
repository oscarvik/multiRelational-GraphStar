from datetime import datetime as dtime
import numpy as np
def get_now():
    now = dtime.now()
    date = str(now.date())
    time = str(now.time()).split(".")[0]
    return date + "_" + time + str(np.random.randint(0,100))