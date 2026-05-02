import h5py
import numpy as np

dataset_path = "easycarla_offline_dataset.hdf5"

with h5py.File(dataset_path, "r") as f:
    print("\nKeys in dataset:")
    for key in f.keys():
        print("-", key)

    obs = f["observations"]
    actions = f["actions"]
    rewards = f["rewards"]
    next_obs = f["next_observations"]
    dones = f["done"]

    print("\nShapes:")
    print("Observations:", obs.shape)
    print("Actions:", actions.shape)
    print("Rewards:", rewards.shape)
    print("Next Obs:", next_obs.shape)
    print("Done:", dones.shape)

    print("\nAction Stats:")
    print("Min:", np.min(actions[:], axis=0))
    print("Max:", np.max(actions[:], axis=0))
    print("Mean:", np.mean(actions[:], axis=0))

    print("\nReward Stats:")
    print("Min:", np.min(rewards[:]))
    print("Max:", np.max(rewards[:]))
    print("Mean:", np.mean(rewards[:]))

    print("\nDone count:", np.sum(dones[:]))

    print("\nNaN Check:")
    print("Obs NaN:", np.isnan(obs[:]).any())
    print("Actions NaN:", np.isnan(actions[:]).any())
    print("Rewards NaN:", np.isnan(rewards[:]).any())
