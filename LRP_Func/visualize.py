import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def sort_score_list(r: np.array, namelist:list, MaxToMin: bool):
    if MaxToMin == True:
        sorted_r = np.zeros((len(r)))
        sorted_namelist = []
        temp_r = r
        for i in range(len(r)):
            index = np.where( temp_r == np.max(temp_r))
            nameindex = np.where(r == temp_r[index])
            sorted_r[i] = temp_r[index]
            temp_r = np.delete(temp_r,index)
            sorted_namelist.append(namelist[int(nameindex[0])])
    else:
        sorted_r = np.zeros((len(r)))
        sorted_namelist = []
        temp_r = r
        for i in range(len(r)):
            index = np.where(temp_r == np.min(temp_r))
            nameindex = np.where(r == temp_r[index])
            sorted_r[i] = temp_r[index]
            temp_r = np.delete(temp_r, index)
            sorted_namelist.append(namelist[int(nameindex[0])])

    return sorted_r, sorted_namelist

def plot_relevance_scores(r: torch.tensor, outfile:str) -> None:
    """Plots results from layer-wise relevance propagation next to original image.

    Method currently accepts only a batch size of one.

    Args:
        x: Original image.
        r: Relevance scores for original image.
        name: Image name.
        config: Dictionary holding configuration.

    Returns:
        None.

    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    r_min = r.min()
    r_max = r.max()
    r = r/ (r_max - r_min)
    im = axes.imshow(r, cmap="coolwarm")
    #axes.set_axis_off()
    fig.colorbar(im)
    fig.tight_layout()
    plt.savefig(outfile, transparent = True)
    plt.close(fig)

def plot_importance(r: np.array, namelist: list, outfile: str)-> None:

    r = np.round(r,4)*100
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10,7))
    plt.barh(range(len(r)), r, height=0.7, color='#00FFFF', alpha=0.8)
    plt.yticks(range(len(r)), namelist)

    plt.xlim(0, max(np.ceil(max(r)+3),np.ceil(1.2*max(r))))
    plt.xlabel("Importance(%)")
    plt.title("Variables Importance")
    for x, y in enumerate(r):
        plt.text(y + 0.002, x - 0.001, '%s' % np.round(y,4)+'%')
    plt.savefig(outfile, transparent = True)
    plt.close()