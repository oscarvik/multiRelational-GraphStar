from datetime import datetime as dtime

def get_now():
    now = dtime.now()
    date = str(now.date())
    time = str(now.time()).split(".")[0]
    return date + " " + time