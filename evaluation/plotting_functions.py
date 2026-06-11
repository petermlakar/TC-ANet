import torch

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import List, Tuple, Dict, Optional

from os.path import join, exists
from os import listdir, mkdir, makedirs

from tqdm import tqdm

from evaluation.metrics import rank_histogram, crps_expectation, energy_score

def axiseq(a):
    a.set_aspect((a.get_xlim()[1] - a.get_xlim()[0])/(a.get_ylim()[1] - a.get_ylim()[0]))

def plot_stratify(models: Dict, config: Dict, station_metadata: Dict):

    alt: torch.Tensor = torch.tensor([station_metadata[i]['alt'] for i in station_metadata])
    lon: torch.Tensor = torch.tensor([station_metadata[i]['lon'] for i in station_metadata])
    lat: torch.Tensor = torch.tensor([station_metadata[i]['lat'] for i in station_metadata])
    idx: torch.Tensor = torch.tensor([station_metadata[i]['index'] for i in station_metadata])

    width: float = 0.2

    h_v, h_e = torch.histogram(alt, bins = 4) 

    score_reference: torch.Tensor = torch.stack([model['crps_per_station'] for model in models if model['label'] == 'ECMWF'][0], dim = 0).mean(dim = 0)

    alt_mu: torch.Tensor = [alt[torch.logical_and(alt >= h_e[0] if i == 1 else alt > h_e[i - 1], alt <= h_e[i])].mean() for i in range(1, len(h_e))]
    alt_sd: torch.Tensor = [alt[torch.logical_and(alt >= h_e[0] if i == 1 else alt > h_e[i - 1], alt <= h_e[i])].std()  for i in range(1, len(h_e))]
    alt_cn: torch.Tensor = [torch.logical_and(alt >= h_e[0] if i == 1 else alt > h_e[i - 1], torch.logical_and(alt <= h_e[i], ~torch.isnan(score_reference))).sum() for i in range(1, len(h_e))]

    f, a = plt.subplots(1, dpi = 300, figsize = (10, 5))

    legend_labels: List = []
    legend_bplots: List = []

    for i, e in enumerate(h_e[1:]): 

        for j, model in enumerate(models):

            if model['label'] == 'ECMWF':
                continue

            score: torch.Tensor = torch.stack(model['crps_per_station'], dim = 0).mean(dim = 0)
            idx: torch.Tensor = torch.logical_and(~torch.isnan(score), torch.logical_and(alt >= h_e[0] if i == 0 else alt > h_e[i] , alt <= e))

            score = (1 - score[idx]/score_reference[idx])*100
    
            bplot = a.boxplot(score, positions = [i + j*width], widths = [width], patch_artist = True, showfliers = False)
            for patch in bplot['boxes']:
                patch.set_facecolor(model['color'])

            if i == 0:
                legend_bplots.append(bplot['boxes'][0])
                legend_labels.append(model['label'])

    a.set_xticks([i + width*(len(models) - 1)/2 - width/2 for i in range(len(h_e) - 1)], labels = [f'{alt_mu[i]:.2f} $\pm$ {alt_sd[i]:.2f}\n({alt_cn[i]})' for i in range(len(h_e) - 1)])
    a.grid()

    a.tick_params(axis = 'both', which = 'major', labelsize = 15)
    a.tick_params(axis = 'both', which = 'minor', labelsize = 15)

    a.legend(legend_bplots, legend_labels, loc = 'best', fontsize = 15)

    a.set_ylabel('CRPSS [%]', fontsize = 18)
    a.set_xlabel(r'$\mu_{\text{altitude}}$ $\pm$ $\sigma_{\text{altitude}}$ [m]' + '\n' + r'(S$_\text{altitude}$)', fontsize = 18)
    a.set_title(config['varName'], fontsize = 20)

    f.tight_layout()
    f.savefig(join(config['outputPath'], 'stratified.png'), bbox_inches = 'tight')
    plt.close(f)


def plot_map(models: Dict, config: Dict, station_metadata: Dict):

    extent: List[int] = [2.0, 11.14, 45, 54.5]

    fig = plt.figure(figsize = (10, 10), dpi = 300)

    ax = plt.axes(projection = ccrs.PlateCarree())
    ax.set_extent(extent, crs = ccrs.PlateCarree())

    # Background features
    ax.add_feature(cfeature.LAND, facecolor = "lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor = "lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth = 0.8)
    ax.add_feature(cfeature.BORDERS, linestyle = ":", linewidth = 0.8)

    # Optional administrative borders
    #ax.add_feature(cfeature.STATES, linewidth = 0.3)

    # Gridlines
    gl = ax.gridlines(
        draw_labels = True,
        linewidth = 0.5,
        color = "gray",
        alpha = 0.5,
        linestyle = "--")

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    lon: torch.Tensor = torch.tensor([station_metadata[i]['lon'] for i in station_metadata])
    lat: torch.Tensor = torch.tensor([station_metadata[i]['lat'] for i in station_metadata])
    idx: torch.Tensor = torch.tensor([station_metadata[i]['index'] for i in station_metadata])

    scores_max: torch.Tensor = torch.stack([torch.stack(model['crps_per_station'], dim = 0).nanmean(dim = 0) for model in models], dim = 1).argmin(dim = 1)
   
    for i in scores_max.unique():

        _i: torch.Tensor = scores_max == i

        sc = ax.scatter(
            lon[_i],
            lat[_i],
            marker = models[i]['marker'],
            c = models[i]['color'],
            label = f"{models[i]['label']} [{(100*_i.sum()/len(_i)):.2f} %]",
            s = 60,
            edgecolors = "black",
            linewidths = 0.5,
            transform = ccrs.PlateCarree(),
            zorder = 10)

    ax.legend(loc = 'best', scatterpoints = 1, fontsize = 16)

    ax.set_xlabel('Longitude', fontsize = 20)
    ax.set_ylabel('Latitude', fontsize = 20)
    ax.set_title(config['varName'], fontsize = 20)

    fig.tight_layout()
    fig.savefig(join(config['outputPath'], 'map.png'), bbox_inches = 'tight')
    plt.close(fig)

def plot_reliability(models: Dict, config: Dict):

    for i, (q, _) in enumerate(models[0]['reliability']):

        f, a = plt.subplots(1, dpi = 300, figsize = (5, 5))

        a.grid()
        a.set_xlabel(f"Predicted rate of\n{config['varName']} $=$ {q:.2f} {config['varUnit']}", fontsize = 15)
        a.set_ylabel(f'Observed rate', fontsize = 18)
        a.set_xlim(0.0, 1.0)
        a.set_ylim(0.0, 1.0)

        _x: torch.Tensor = torch.linspace(0, 1, steps = 10)
        _y: torch.Tensor = torch.linspace(0, 1, steps = 10)

        a.plot(_x, _y, linestyle = 'dashed', color = 'black')

        a.fill_between(_x, 0.0, _y, color = 'sandybrown', alpha = 0.2)
        a.fill_between(_x, _y, 1.0, color = 'lightskyblue', alpha = 0.2)

        a.tick_params(axis = 'both', which = 'major', labelsize = 18)
        a.tick_params(axis = 'both', which = 'minor', labelsize = 18)

        for model in models:
            a.plot(model['reliability'][i][1][0], model['reliability'][i][1][1], label = model['label'], color = model['color'], linewidth = 2, marker = model['marker'], markevery = 5 if model['reliability'][i][1][0].shape[0] > 20 else 1)
        
        a.legend(loc = 'best')

        f.tight_layout()
        f.savefig(join(config['outputPath'], f'realiability_{q:.2f}.png'), bbox_inches = 'tight')
        plt.close(f)

def plot_distribution(models: Dict, y: torch.Tensor, config: Dict):

    path: str = join(config['outputPath'], 'dist_station')

    makedirs(path)

    _y = y[~y.isnan()]
    _y = _y[_y > 0]

    f, a = plt.subplots(len(models), 1, dpi = 300, figsize = (10, 5*len(models)))

    for i, model in enumerate(models):
            a[i].grid()
            a[i].set_ylim(0.0, 1.0)

            h = a[i].hist(_y, bins = len(_y.unique())//10, density = True)
            a[i].hist(model['predictions'].flatten(), bins = h[1], color = model['color'], alpha = 0.8, density = True, label = model['label'])
            a[i].legend()

    f.tight_layout()
    f.savefig(join(path, 'total.png'), bbox_inches = 'tight')
    plt.close(f)

    for s in range(y.shape[0]):

        _y = y[s][~y[s].isnan()]
        _y = _y[_y > 0.0]

        if not _y.any():
            continue

        print(s, len(_y.unique()))

        f, a = plt.subplots(len(models), 1, dpi = 300, figsize = (10, 5*len(models))) 

        for i, model in enumerate(models):
           
            p = model['predictions'][s].flatten()
            p = p[p > 0]
           
            a[i].grid()
            a[i].set_ylim(0.0, 1.0)

            h = a[i].hist(_y, bins = len(_y.unique())//10, density = True)
            a[i].hist(model['predictions'].flatten(), bins = h[1], color = model['color'], alpha = 0.8, density = True)

        f.savefig(join(path, f'dist_{s}.png'), bbox_inches = 'tight')
        plt.close(f)

def plot_roc(models: Dict, config: Dict):

    f, a = plt.subplots(1, dpi = 300, figsize = (8, 8))

    for model in models:
        a.plot(model['roc'][1], model['roc'][0], label = model['label'], color = model['color'])

    a.set_xlabel('FPR')
    a.set_ylabel('TPR')
    a.legend()
    a.grid()

    f.savefig(join(config['outputPath'], 'roc.png'))
    plt.close(f)

def plot_score(models: Dict, config: Dict):

    assert 'xlabel' in config
    assert 'ylabel' in config

    assert 'outputPath' in config
    assert 'outputName' in config

    assert 'metric' in config

    key: str = config['metric']

    f, a = plt.subplots(1, dpi = 300)
    markersize: float = 8.0

    if 'c' in config:
        an = a.twinx()
        an.set_zorder(2)

    for model in models:
       
        x: torch.Tensor = model[f"{config['metric']}_x"]

        N: int = len(model[key])

        skill_score: torch.Tensor = torch.stack([c for c in model[key]], dim = 0)
        
        if N > 1:
            skill_score_sd: torch.Tensor = skill_score.std(dim = 0)
        skill_score = skill_score.mean(dim = 0) 

        a.set_xlabel(config['xlabel'])

        a.plot(x, skill_score, color = model['color'], alpha = 0.8, zorder = 2)

        if 'markevery' in config:

            e: torch.Tensor = torch.linspace(x.min(), x.max(), config['markevery'])
            i: torch.Tensor = (e[None] - x[:, None]).abs().argmin(dim = 0)

            a.scatter(x[i], skill_score[i], label = model['label'], marker = model['marker'], color = model['color'], alpha = 0.8, zorder = 3, s = markersize**2, edgecolors = 'none')
            if N > 1:
                a.errorbar(x[i], skill_score[i], yerr = skill_score_sd[i], color = model['color'], linestyle = '', barsabove = True, capsize = 3, alpha = 0.8)
        else:
            a.scatter(x, skill_score, label = model['label'], marker = model['marker'], color = model['color'], alpha = 0.8, zorder = 3, s = markersize**2, edgecolors = 'none')
            if N > 1:
                a.errorbar(x, skill_score, yerr = skill_score_sd, color = model['color'], linestyle = '', barsabove = True, capsize = 3, alpha = 0.8)

        print(skill_score.mean())

    if 'c' in config:

        x_: torch.Tensor = x[::len(x)//10]
        dx: torch.Tensor = (x_[1:] - x_[:-1]).min()*0.8

        an.bar(x_, model[f"{config['metric']}_c"][::len(x)//10], width = dx, alpha = 0.7, align = 'edge', color = 'gainsboro', zorder = 1)
        an.set_ylabel('Fraction of observations above threshold')
        an.set_ylim(a.get_ylim()[0]/a.get_ylim()[1]*1.1, 1.1)

        a.set_zorder(an.get_zorder() + 1)
        a.patch.set_visible(False)

        an.set_yticks(torch.linspace(0.0, 1.0, steps = 5))
        an.grid(axis = 'y', linestyle = 'dashed', alpha = 0.2)

    a.grid()
    a.legend(loc = 'best', fancybox = True, framealpha = 0.5, bbox_to_anchor = [0.5, 0.0, 0.5, 1.0])
    a.set_xlabel(config['xlabel'])
    a.set_ylabel(config['ylabel'])

    if 'c' in config:
        a.yaxis.tick_right()
        a.yaxis.set_label_position('right')
        an.yaxis.tick_left()
        an.yaxis.set_label_position('left')

    f.savefig(join(config['outputPath'], config['outputName']), bbox_inches = 'tight')
    plt.close(f)

def plot_skill_score(models: Dict, config: Dict, legend: bool = True):

    assert 'xlabel' in config
    assert 'ylabel' in config

    assert 'outputPath' in config
    assert 'outputName' in config

    assert 'metric' in config
    assert 'base' in config

    key: str = config['metric']

    base: torch.Tensor = [model for model in models if model['label'] == config['base']][0][key][0]

    print(config['metric'])

    f, a = plt.subplots(1, dpi = 300)
    markersize: float = 15.0
    linewidth = 2

    if 'title' in config:
        a.set_title(config['title'], fontsize = 20)

    if 'c' in config:
        an = a.twinx()
        an.set_zorder(2)

    for model in models:
       
        if 'x' not in config:
            x: torch.Tensor = torch.linspace(1, 20, steps = 20)
            a.set_xticks([1] + [l for l in torch.linspace(5, 20, steps = 4)], labels = ['1', '5', '10', '15', '20'])
        else:
            x: torch.Tensor = config['x']

        N: int = len(model[key])

        skill_score: torch.Tensor = torch.stack([(1 - c/base)*100 for c in model[key]], dim = 0)
        
        if N > 1:
            skill_score_sd: torch.Tensor = skill_score.std(dim = 0)
        skill_score = skill_score.mean(dim = 0) 

        a.set_xlabel(config['xlabel'])

        if model['label'] == config['base']:
            a.plot(x, skill_score, label = model['label'], color = model['color'], linestyle = 'dashed', zorder = 2, linewidth = linewidth)
        else:
            a.plot(x, skill_score, color = model['color'], alpha = 0.8, zorder = 2, linewidth = linewidth)

        if not (model['label'] == config['base']):
            if 'markevery' in config:

                e: torch.Tensor = torch.linspace(x.min(), x.max(), config['markevery'])
                i: torch.Tensor = (e[None] - x[:, None]).abs().argmin(dim = 0)

                a.scatter(x[i], skill_score[i], label = model['label'], marker = model['marker'], color = model['color'], alpha = 0.8, zorder = 3, s = markersize**2, edgecolors = 'none')
                if N > 1:
                    a.errorbar(x[i], skill_score[i], yerr = skill_score_sd[i], color = model['color'], linestyle = '', barsabove = True, capsize = 3, alpha = 0.8)
            else:
                a.scatter(x, skill_score, label = model['label'], marker = model['marker'], color = model['color'], alpha = 0.8, zorder = 3, s = markersize**2, edgecolors = 'none')
                if N > 1:
                    a.errorbar(x, skill_score, yerr = skill_score_sd, color = model['color'], linestyle = '', barsabove = True, capsize = 3, alpha = 0.8)

        print(skill_score.mean())

    if 'c' in config:

        x_: torch.Tensor = x[::len(x)//10]
        dx: torch.Tensor = (x_[1:] - x_[:-1]).min()*0.8

        an.bar(x_, config['c'][::len(x)//10], width = dx, alpha = 0.7, align = 'edge', color = 'gainsboro', zorder = 1)
        an.set_ylabel('Fraction of observations\nabove threshold', fontsize = 15)
        an.set_ylim(a.get_ylim()[0]/a.get_ylim()[1]*1.1, 1.1)

        a.set_zorder(an.get_zorder() + 1)
        a.patch.set_visible(False)

        an.set_yticks(torch.linspace(0.0, 1.0, steps = 5))
        an.grid(axis = 'y', linestyle = 'dashed', alpha = 0.2)
    
        an.tick_params(axis = 'both', which = 'minor', labelsize = 15)
        an.tick_params(axis = 'both', which = 'major', labelsize = 15)

    if 'q999' in config:
        an.vlines(config['q999'], 0, 1, color = 'black', linestyle = 'dashed', alpha = 0.8)

    a.grid()
    
    if legend:
        if 'c' in config:
            a.legend(loc = 'best', fancybox = True, framealpha = 0.5, bbox_to_anchor = [0.5, 0.0, 0.5, 1.0], fontsize = 16)
            #a.legend(loc = 'lower right', fancybox = True, framealpha = 0.5, fontsize = 16)
        else:
            a.legend(loc = 'best', fancybox = True, framealpha = 0.5, fontsize = 16)

    a.set_xlabel(config['xlabel'], fontsize = 18)
    a.set_ylabel(config['ylabel'], fontsize = 18)

    a.tick_params(axis = 'both', which = 'major', labelsize = 15)
    a.tick_params(axis = 'both', which = 'minor', labelsize = 15)

    if 'c' in config:
        a.yaxis.tick_right()
        a.yaxis.set_label_position('right')
        an.yaxis.tick_left()
        an.yaxis.set_label_position('left')

    f.tight_layout()
    f.savefig(join(config['outputPath'], config['outputName']), bbox_inches = 'tight')
    plt.close(f)

def plot_crps(models: Dict, config: Dict):

    assert 'title' in config
    assert 'ylabel' in config
    assert 'outputPath' in config
    assert 'prefix' in config
    assert 'datasetType' in config

    f, a = plt.subplots(1, dpi = 300, figsize = (8, 6))

    x: torch.Tensor = torch.arange(6, 21*6, step = 6)

    for model in models:
        a.plot(x, model['crps'].nanmean(dim = (0, 1)), label = model['label'], color = model['color'], marker = model['marker'])
        
    a.legend()
    a.grid()

    a.set_title(f"{config['title']}")
    a.set_ylabel(config['ylabel'])
    a.set_xlabel('Lead time [6 hour steps]')
    a.set_xticks(torch.arange(6, 21*6, step = 6), labels = [f'{int(k)}' for k in torch.arange(6, 21*6, step = 6)])

    f.tight_layout()
    
    f.savefig(join(config['outputPath'], f"crps_{config['prefix']}_{config['datasetType']}.png"), bbox_inches = 'tight')
    plt.close(f)

def plot_energy_per_y(models: Dict, config: Dict):

    assert 'base' in config

    model_base: Dict = [model for model in models if model['label'] == config['base']][0]

    f, a = plt.subplots(1, dpi = 300)

    for model in models:
        
        if model['label'] == config['base']:
            a.plot((1 - model['energy_thr'][1]/model_base['energy_thr'][1])*100, label = model['label'], color = model['color'], linestyle = 'dashed')
        else:
            a.plot((1 - model['energy_thr'][1]/model_base['energy_thr'][1])*100, label = model['label'], color = model['color'], alpha = 0.8, marker = model['marker'])

    a.legend(loc = 'best')
    a.set_xlabel(config['xlabel'])
    a.set_ylabel('Energy Skill Score')
    f.savefig(join(config['outputPath'], 'energy_per_y.png'), bbox_inches = 'tight')
    plt.close(f)

def plot_vario_per_y(models: Dict, config: Dict):

    assert 'base' in config

    model_base: Dict = [model for model in models if model['label'] == config['base']][0]

    f, a = plt.subplots(1, dpi = 300)

    for model in models:
        
        if model['label'] == config['base']:
            a.plot((1 - model['vario_thr'][1]/model_base['vario_thr'][1])*100, label = model['label'], color = model['color'], linestyle = 'dashed')
        else:
            a.plot((1 - model['vario_thr'][1]/model_base['vario_thr'][1])*100, label = model['label'], color = model['color'], alpha = 0.8)

    a.legend(loc = 'best')
    a.set_xlabel(config['xlabel'])
    a.set_ylabel('VSS')
    f.savefig(join(config['outputPath'], 'vario_per_y.png'), bbox_inches = 'tight')
    plt.close(f)

def plot_rank(models: Dict, config: Dict):

    assert 'outputPath' in config
    assert 'prefix' in config
    assert 'datasetType' in config

    for model in models:

        rnk: torch.Tensor = model['rank']

        if 'rank_sd' in model:
            rnk_sd: torch.Tensor = model['rank_sd']
        else:
            rnk_sd = None

        f, a = plt.subplots(1, dpi = 300, figsize = (8, 8))

        abs_dist: float = (rnk - 1/len(rnk)).abs().mean()
        a.bar(torch.arange(len(rnk)), rnk, width = 1.0, color = model['color'], align = 'edge', alpha = 0.8, label = f'{100*len(rnk)*abs_dist:.3f} %')
        if not (rnk_sd is None):
            a.errorbar(torch.arange(len(rnk)) + 0.5, rnk, yerr = rnk_sd, ecolor = model['color'], linestyle = '', barsabove = True, capsize = 1)
        a.hlines(1/len(rnk), 0, len(rnk), color = 'black', linestyle = 'dashed')

        if config['prefix'] == 'temperature':
            a.set_title(model['label'], fontsize = 26)

        if config['prefix'] == 'wind':
            a.set_xlabel('Ranks', fontsize = 24)

        if model['label'] == 'ECMWF':
            a.set_ylabel('Density', fontsize = 24)

        a.tick_params(axis = 'both', which = 'major', labelsize = 20)
        a.tick_params(axis = 'both', which = 'minor', labelsize = 20)
        a.legend(loc = 'best', fancybox = True, framealpha = 0.5, fontsize = 22)

        if model['label'] != 'ECMWF':
            nom: float = 1.5/len(rnk)
            a.set_ylim(0.0, nom)

        f.tight_layout()
        f.savefig(join(config['outputPath'], f"rank_{config['prefix']}_{config['datasetType']}_{model['label']}.png"), bbox_inches = 'tight')
        plt.close(f)


def plot_rank_per_station(models: Dict, config: Dict, station_names: Dict, y: torch.Tensor):

    assert 'outputPath' in config
    assert 'prefix' in config
    assert 'datasetType' in config

    for model in models:

        path: str = join(config['outputPath'], f"rank_per_station_{config['prefix']}_{config['datasetType']}", model['label'])
        makedirs(path)

        for i in tqdm(range(model['predictions'].shape[0])):

            rnk: torch.Tensor = rank_histogram(model['predictions'][i], y[i])

            f, a = plt.subplots(1, dpi = 300, figsize = (8, 8))

            a.bar(torch.arange(len(rnk)), rnk, width = 1.0, label = model['label'], color = model['color'], align = 'edge')
            a.hlines(1/len(rnk), 0, len(rnk), color = 'black', linestyle = 'dashed')

            a.set_ylim(0.0, 0.04)
            a.set_title(station_names[i])

            f.tight_layout()
            f.savefig(join(path, f"{i + 1}_{station_names[i].replace('/', '_')}.png"), bbox_inches = 'tight')
            plt.close(f)

def plot_qss(models: Dict, config: Dict):
    
    assert 'title' in config
    assert 'ylabel' in config
    assert 'outputPath' in config
    assert 'prefix' in config
    assert 'datasetType' in config
    assert 'base' in config

    q_base: torch.Tensor = [model for model in models if model['label'] == config['base']][0]['qloss'].nanmean(dim = (0, 1, 2))
    n: int = q_base.shape[0]
    x: torch.Tensor = torch.arange(1/(n + 1), 1, step = 1/(n + 1))

    f, a = plt.subplots(1, dpi = 300)

    for model in models:
       
        q: torch.Tensor = model['qloss'].nanmean(dim = (0, 1, 2))

        qss: torch.Tensor = (1 - q/q_base)*100

        if model['label'] == config['base']:
            a.plot(x, qss, label = model['label'], color = model['color'], linestyle = 'dashed')
        else:
            a.plot(x, qss, label = model['label'], color = model['color'], alpha = 0.8)

        print(qss.mean()) 

    a.legend(loc = 'best')
    a.set_xlabel('Quantiles')
    a.set_ylabel('QSS')
    f.savefig(join(config['outputPath'], 'crpss'), bbox_inches = 'tight')
    plt.close(f)

def plot_case(models: Dict, 
              config: Dict, 
              y: torch.Tensor, 
              station_names: Dict,
              station_metadata: Dict,
              #main_title: str = r"Joint $\overset{\mathrm{ECC}}{\rightarrow}$ Marginal forecast",
              main_title: str = 'Joint forecast',
              station_index: int = 30):

    assert 'outputPath' in config
    assert 'prefix' in config
    assert 'datasetType' in config
    assert 'ylabel' in config
    assert 'ylim' in config

    ymax: torch.Tensor = y.nan_to_num().flatten(start_dim = -2).max(dim = -1)[0].argmax()
    station_index = ymax.item()
   
    station_alt: float = [station_metadata[s]['alt'] for s in station_metadata if station_metadata[s]['index'] == station_index][0]

    prefix: str = config['prefix']
    #prc: bool = config['prefix'] == 'precipitation'
    prc: bool = False

    for model in models:

        path: str = join(config['outputPath'], f"case_{config['prefix']}_{config['datasetType']}_{station_index}_{model['label']}")
        makedirs(path)

        for i in tqdm(range(model['predictions'].shape[1])):

            f, a = plt.subplots(1, dpi = 300)
            
            x: torch.Tensor = torch.linspace(1, 20, steps = 20)

            if prc:
                for j in range(p.shape[-1]):
                    
                    if j == 0:
                        a.fill_between(torch.arange(p.shape[0]), p[:, j], facecolor = model['color'], alpha = 0.9 - j*0.9/(p.shape[1] + 1), linewidth = None)
                    else:
                        a.fill_between(torch.arange(p.shape[0]), p[:, j - 1], p[:, j], facecolor = model['color'], alpha = 0.9 - j*0.9/(p.shape[1] + 1), linewidth = None)
                    
                a.plot(model['predictions'][station_index, i, ..., model['predictions'].shape[-1]//2], color = model['color'], linestyle = 'dashed')
            else:
                a.plot(x, model['predictions'][station_index, i], color = model['color'], alpha = 0.4)
                a.plot(x, model['predictions'][station_index, i].quantile(0.5, dim = -1, keepdim = True).swapaxes(0, -1)[0], color = model['color'], label = model['label'] + f": {model['predictions'].shape[-1]}", linestyle = 'dashed', linewidth = 3)
            
            a.plot(x, y[station_index, i], label = 'Observations', color = 'lightskyblue', linewidth = 2)
            a.set_xticks([1] + [l for l in torch.linspace(5, 20, steps = 4)], labels = ['1', '5', '10', '15', '20'])
            a.set_ylim(config['ylim'])

            if prefix == 'wind': 
                a.set_xlabel('Lead time [6 hour steps]', fontsize = 15)

            #a.set_ylabel(f'{station_names[station_index]} @ {station_alt} $m$' + f"\n{config['ylabel']}", fontsize = 15)
            #a.legend(loc = 'best', fancybox = True, framealpha = 0.5, fontsize = 13)

            if prefix == 'temperature':
                a.set_title(main_title, fontsize = 18)

            a.tick_params(axis = 'both', which = 'minor', labelsize = 15)
            a.tick_params(axis = 'both', which = 'major', labelsize = 15)
            a.grid()
            f.tight_layout()
            f.savefig(join(path, f"{i + 1}_{station_names[station_index].replace('/', '_')}.png"), bbox_inches = 'tight')
            plt.close(f)

