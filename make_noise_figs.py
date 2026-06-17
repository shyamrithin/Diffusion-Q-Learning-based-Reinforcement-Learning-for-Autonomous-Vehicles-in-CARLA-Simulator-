# ==========================================================
# plot_all_noise_traj.py
# Generates ONE trajectory-on-Town03 figure per agent per
# noise level (separate figures, not overlaid).
#
# Output: noise_traj_<agent>_<level>.png  (e.g. noise_traj_dqle_0.png)
# Covers: dqle/sac/ppo  x  0/5/10/15 %  (12 figures, those that exist)
#
# Each figure: faint Town03 road network + green reference
# route + the agent's path coloured by step (blue->red) +
# START/END markers + collision/offroad ring on terminal step.
#
# Requires CARLA up (map outline + reference). Run from
# ~/Carla/RLCarla. Uses episode 1 CSV of each cell.
# CARLA 0.9.16 | Python 3.10
# ==========================================================
import os, sys, csv, types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUTE      = "route_3_straight"
ROUTE_KEY  = "route_3_straight"
AGENTS     = ["dqle","sac","ppo"]
LEVELS     = [0,5,10,15]
SUFFIX     = {0:"",5:"_noise5",10:"_noise10",15:"_noise15"}
ALABEL     = {"dqle":"DQL-E","sac":"SAC","ppo":"PPO"}

# ---- robust GlobalRoutePlanner import ---------------------
def _load_grp():
    nav=os.path.expanduser("~/Carla/PythonAPI/carla/agents/navigation")
    if not os.path.isdir(nav): raise ImportError("CARLA navigation not found")
    ar=os.path.dirname(nav)
    pa=types.ModuleType("agents"); pa.__path__=[ar]; sys.modules["agents"]=pa
    pn=types.ModuleType("agents.navigation"); pn.__path__=[nav]
    sys.modules["agents.navigation"]=pn
    from agents.navigation.global_route_planner import GlobalRoutePlanner as G
    return G

import carla
import route_utils as ru
GlobalRoutePlanner=_load_grp()

# ---- connect once: map outline + reference ----------------
client=carla.Client("localhost",2000); client.set_timeout(20.0)
world=client.get_world(); cmap=world.get_map()
wps=cmap.generate_waypoints(4.0)
RX=[w.transform.location.x for w in wps]
RY=[w.transform.location.y for w in wps]
spec=ru.ROUTES[ROUTE_KEY]
grp=GlobalRoutePlanner(cmap,2.0)
route=grp.trace_route(cmap.get_spawn_points()[spec["spawn"]].location,
                      cmap.get_spawn_points()[spec["dest"]].location)
REFX=[w.transform.location.x for w,_ in route]
REFY=[w.transform.location.y for w,_ in route]

def load(agent,level):
    path=os.path.join("results",f"{agent}_empty{SUFFIX[level]}",
                      f"{ROUTE}_ep01.csv")
    if not os.path.exists(path): return None
    xs,ys,st,co,of=[],[],[],[],[]
    with open(path) as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x"])); ys.append(float(row["y"]))
            st.append(int(row["step"]))
            co.append(int(row.get("collision",0)))
            of.append(int(row.get("offroad",0)))
    return (np.array(xs),np.array(ys),np.array(st),co,of)

def make_fig(agent,level):
    d=load(agent,level)
    if d is None:
        print(f"  (skip {agent} {level}% — no csv)"); return
    xs,ys,st,co,of=d
    fig,ax=plt.subplots(figsize=(11,11))
    ax.scatter(RX,RY,s=1,color="#dddddd",zorder=1)
    ax.plot(REFX,REFY,"-",color="#11aa11",lw=5,alpha=0.5,zorder=2,
            label="Reference route")
    ax.scatter(REFX[-1],REFY[-1],s=320,marker="*",color="#11aa11",
               edgecolors="black",zorder=6,label="Destination")
    sc=ax.scatter(xs,ys,c=st,cmap="coolwarm",s=14,zorder=4)
    ax.plot(xs,ys,"-",color="gray",lw=0.5,alpha=0.5,zorder=3)
    plt.colorbar(sc,label="step (blue=start -> red=end)")
    ax.scatter(xs[0],ys[0],s=200,marker="o",color="lime",
               edgecolors="black",zorder=7,label="Start")
    ax.scatter(xs[-1],ys[-1],s=240,marker="X",color="red",
               edgecolors="black",zorder=7,label="End")
    term=[i for i in range(len(co)) if co[i] or of[i]]
    if term:
        ax.scatter(xs[term],ys[term],s=300,facecolors="none",
                   edgecolors="red",linewidths=3,zorder=8,
                   label="collision/offroad")
    # zoom to trajectory + reference
    m=20
    ax.set_xlim(min(xs.min(),min(REFX))-m,max(xs.max(),max(REFX))+m)
    ax.set_ylim(min(ys.min(),min(REFY))-m,max(ys.max(),max(REFY))+m)
    ax.set_xlabel("CARLA x (m)"); ax.set_ylabel("CARLA y (m)")
    ax.set_title(f"{ALABEL[agent]} on R3 — {level}% observation noise\n"
                 f"{len(xs)} steps")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out=f"noise_traj_{agent}_{level}.png"
    fig.savefig(out,dpi=120,bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")

for ag in AGENTS:
    print(f"{ALABEL[ag]}:")
    for lv in LEVELS:
        make_fig(ag,lv)
print("done.")