# Code ECG

import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.fftpack import fft, fftfreq
from scipy.stats import zscore
import numpy as np
import pandas as pd

from os import getlogin
from os.path import expanduser
import os
import itertools

from hrvanalysis import remove_outliers, remove_ectopic_beats
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, get_csi_cvi_features
from hrvanalysis import plot_psd, plot_distrib, plot_poincare

from scipy import stats
#import statsmodels.stats.api as sms

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
# for non-interactive figure
matplotlib.use('Agg')

# patient par patient



#sessions = ['PRE']
sessions = ['S2']

conf_subj = {'Herve':{'S1':{'ectopic': 0.18, 'deb':[0/60, 180/60], 'low_rri': 300, 'high_rri': 2000, 'freq':500},
                      'S2':{'ectopic': 0.18, 'deb':[180/60, 360/60], 'low_rri': 300, 'high_rri': 2000, 'freq':500},
                      'S3':{'ectopic': 0.18, 'deb':[360/60, 510/60], 'low_rri': 300, 'high_rri': 2000, 'freq':500},
                      'ST':{'ectopic': 0.18, 'deb':[0/60, 510/60], 'low_rri': 300, 'high_rri': 2000, 'freq':500}}}


subjects = ['Herve']



def interpolate_nan(rr):
    """Overcome hrv_function, as we need to specify direction 
    """
    series = pd.Series(rr)
    interp = series.interpolate(method="linear",
                                limit=None,
                                limit_area=None,
                                limit_direction="both")
    return interp.values.tolist()


def compute_period(nn, peaks,sfreq,period, interp='cubic', method='direct'):
    
    period_nn = np.array(nn)[np.logical_and(peaks > period[0], peaks < period[1])[:-1, 0]]
    td_feat = get_time_domain_features(period_nn)
    fd_feat = freq_domain_feat(period_nn, sfreq, interp=interp,
                               method=method)
    csi_feat = get_csi_cvi_features(period_nn)
    return td_feat, fd_feat, csi_feat, period_nn


def plot_multi_distrib(period_nn, title):
    max_nn = max([int(max(p)) for p in period_nn])
    min_nn = min([int(min(p)) for p in period_nn])
    bin_length = 20
    style.use("seaborn-white")
    fig = plt.figure(figsize=(10, 5))
    plt.title(title, fontsize=20)
    plt.xlabel("Time (ms)", fontsize=15)
    plt.ylabel("Number of RR Interval per bin", fontsize=15)
    plt.hist(period_nn, bins=range(min_nn - 10, max_nn + 10, bin_length),
             histtype='bar', label=["début", "milieu", "fin"])
    plt.legend()
    plt.show()
    return fig


def get_psd(nn, method="direct", interp="linear"):
    freq_data=conf_subj[subj][sess]['freq']
    ts = np.cumsum(np.array(nn)/sfreq) - nn[0]/sfreq
    # Interpolation
    interpf = interp1d(x=ts, y=nn, kind=interp)
    nni = interpf(np.arange(0, ts[-1], 1./float(sfreq)))
    if method == 'direct':
        L = len(nni)
        nfft = np.int(2**np.ceil(np.log2(np.abs(L))))
        y = fft(zscore(nni), nfft)/L
        f = fftfreq(nfft, 1/sfreq) 
        psd = 2*np.abs(y[f > 0])
        f = f[f > 0]
    elif method == 'welch':
        nfft=2**18
        f, psd = welch(x=zscore(nni), fs=sfreq, window='hann', nperseg=nfft,
                       noverlap=nfft//2, nfft=nfft, return_onesided=True)
    else:
        raise AttributeError("unknow method {}".format(method))
    return f, psd
    # ts = np.cumsum(np.array(nn)/1000) - nn[0]/1000
    # # Interpolation
    # interpf = interp1d(x=ts, y=nn, kind=interp)
    # nninterp = interpf(np.arange(0, ts[-1], 1./float(sfreq)))
    # # Remove DC
    # nn = nninterp - np.mean(nninterp)
    # # Estimate PSD
    # freq, psd = welch(x=nn, fs=sfreq, window='hann', nfft=4096)
    # return freq, psd


def plot_multi_psd(period_nn, title,method="welch"):
    vlf, lf, hf = (0.003, 0.04), (0.04, 0.15), (0.15, 0.40)
    freq_data=conf_subj[subj][sess]['freq']
    # Plot parameters
    style.use("seaborn-white")
    fig = plt.figure(figsize=(6,12))
    for nn, splt in zip(period_nn, [1, 2, 3]):
        freq, psd = get_psd(nn, method=method, interp="cubic") 
        vlf_idx = np.logical_and(freq >= vlf[0], freq < vlf[1])
        lf_idx = np.logical_and(freq >= lf[0], freq < lf[1])
        hf_idx = np.logical_and(freq >= hf[0], freq < hf[1])
        frequencies = [vlf_idx, lf_idx, hf_idx]
        label_list = ["VLF component", "LF component", "HF component"]

        plt.subplot(3,1, splt)
        if splt == 1: plt.title(title, fontsize=15)
        if splt == 2: plt.ylabel(r"PSD ($\frac{s^2}{Hz}$)", fontsize=10)
        if splt == 3: plt.xlabel("Frequency (Hz)", fontsize=10)
        for band_index, label in zip(frequencies, label_list):
            plt.fill_between(freq[band_index], 0,
                             psd[band_index] / (1000 * len(psd[band_index])),
                             label=label)
        plt.legend(loc="best")
        plt.xlim(0, hf[1])
    plt.tight_layout()
    plt.show()
    return fig


def freq_domain_feat(nn, sfreq, interp='cubic', method='direct'):
    """"Get frequency domain information

    interp_type is 'cubic' or 'linear'. Otherwise see scipty interp1d doc.
    method is 'direct' for fft computation or 'welch'.
    """
    f, psd = get_psd(nn, method=method, interp=interp)
    fd = {}
    fd['vlf'] = np.sum(psd[f <= 0.04])
    fd['lf'] = np.sum(psd[f <= 0.15]) - fd['vlf']
    fd['hf'] = np.sum(psd[f <= 0.4]) - fd['vlf'] - fd['lf']
    tp = np.sum(psd[f <= 0.4])
    fd['plf'] = fd['lf'] / (tp - fd['vlf']) * 100
    fd['phf'] = fd['hf'] / (tp - fd['vlf']) * 100
    fd['lf_hf_ratio'] = fd['lf'] / fd['hf']
    return fd


# Main program
hrv = []
for subj in subjects: #subj parcourt "subjects"
    # if getlogin() == 'Lapin':
    #     path  = expanduser('~') + '/Documents/snv/HDJ SNV/Annotation files RR/'
    # # elif getlogin() == 'marinezagdoun':
    # #     path = '/Volumes/NO NAME/SNV_STIM1H/'
    # # elif getlogin() == 'sylchev':
    # #     path = expanduser('~') + '/src/datasets/STIM_1H/RRann/'
    # else:
        path = 'C:/Users/justi/Documents/EPF/5A/Semestre 1/Analyse donnees neuro/'
        
        for sess in sessions: #sess parcourt "sessions"
            if subj=='99ZZ' and sess=='S41':
                fname = path + subj + '/' + subj +  '_' + sess + '_waveform_ann.mat'
            # elif subj == "130GM": # and sess==sess:
            #     fname = path + subj + '/' + 'HDJ_SNV_' + subj + '_' + sess + '_ann.mat'
            else:
                fname = path + subj + '_' + sess + '_ann.mat'
          
            print("\nProcessing {} - {}\n=================".format(subj, sess))
        
            if fname.split("/")[6] in os.listdir(path): #récup_re le nom fichier.ext dans le dossier 10 (fin) du chemin
                os.chdir('C:/Users/justi/Documents/EPF/5A/Semestre 1/Analyse donnees neuro/') #change de dossier
        
            
            #if ~exist(fname)
            hrvtools_ann = sio.loadmat(fname, struct_as_record=True)
            sfreq = conf_subj[subj][sess]['freq']
            peaks = hrvtools_ann['Ann'] /sfreq * 1000
            rr = np.diff(peaks.reshape(len(peaks)))
            rr = rr.reshape(len(rr)).tolist()
            # plot_distrib(np.rint(rr).astype(int))
            
            ### Remove outliers
            rr_wo_outliers = remove_outliers(rr_intervals=rr,  
                                 low_rri=conf_subj[subj][sess]['low_rri'],
                                 high_rri=conf_subj[subj][sess]['high_rri'])
            print ("Removed {} outliers for {} RR: {:.2f}% of the data".format(
                    np.isnan(rr_wo_outliers).sum(), len(rr),
                    np.isnan(rr_wo_outliers).sum()/len(rr)))
            
            # Replace outliers nan values with linear interpolation
            rr_interp = interpolate_nan(rr=rr_wo_outliers)
            # interpolation methods describes in Pandas:
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
            # plot_distrib(np.rint(rr_interp).astype(int))
            
            ### Remove ectopic beats from signal
            # (peaks that differs more than 20-25% from the previous peaks)
            # methods could be malik, kamath, karlsson, acar or custom.
            nn = remove_ectopic_beats(rr_intervals=rr_interp, method="karlsson",
                                      custom_removing_rule=conf_subj[subj][sess]['ectopic'])
            
            # Replace ectopic beats nan values with linear interpolation
            nn_interp = interpolate_nan(rr=nn)
            # plot_distrib(np.rint(nn_interp).astype(int))
            plot_poincare(nn_interp)
            plt.savefig("poincare-"+subj+"-"+sess+".png")
                
            # td_feat = get_time_domain_features(nn_interp)
            # for f in td_feat.keys():
            #     print ('{}: {:.2f}'.format(f, td_feat[f]))
            # fd_feat = get_frequency_domain_features(nn_interp, "welch")
            # for f in fd_feat.keys():
            #     print ('{}: {:.2f}'.format(f, fd_feat[f]))
            # plot_psd(nn_interp, method="welch")
            # plot_psd(nn_interp, method="lomb")
            
            m = 60000
            deb = [conf_subj[subj][sess]['deb'][0]*m,
                   conf_subj[subj][sess]['deb'][1]*m]
#            mil = [conf_subj[subj][sess]['mil'][0]*m,
#                   conf_subj[subj][sess]['mil'][1]*m]
#            fin = [conf_subj[subj][sess]['fin'][0]*m,
#                   conf_subj[subj][sess]['fin'][1]*m]
            
            period_nn = []
            for period, pname in zip([deb], ["debut"]):
                freq_data = conf_subj[subj][sess]['freq']
                td, fd, csi, pnn = compute_period(nn_interp, peaks,freq_data, period)
                hrv.append({'subject':subj, 'session':sess, "period":pname,
                                   **td, **fd, **csi})
                period_nn.append(pnn)
            fig = plot_multi_distrib(period_nn, "RR intervals - "+subj+" - "+sess)
            fig.savefig("RRintervals-"+subj+"-"+sess+".png")
           ################ fig = plot_multi_psd(period_nn, "PSD - "+subj+" - "+sess)
            fig.savefig("PSD-"+subj+"-"+sess+".png")
            plt.close('all')
        
        else:
            td ='NaN' 
            fd ='NaN'
            csi ='NaN'
            pnn ='NaN'
            for period, pname in zip([deb], ["debut"]):
                
                hrv.append({'subject':subj, 'session':sess, "period":pname})


hrv = pd.DataFrame(hrv)
hrv.to_excel("hrv.xlsx")
# hrv = pd.read_excel("hrv.xlsx")

# hrv.loc[(hrv['subj'] == 'AF') & (hrv['sess'] == 'S04') &
#         (hrv['period'] == "deb"), ['sdnn']]

# ['subject', 'session', 'period', 'mean_nni', 'sdnn', 'sdsd', 'nni_50',
#     'pnni_50', 'nni_20', 'pnni_20', 'rmssd', 'median_nni', 'range_nni',
#     'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr', 'lf', 'hf',
#     'lf_hf_ratio', 'lfnu', 'hfnu', 'total_power', 'vlf', 'csi', 'cvi',
#     'Modified_csi']
# sns.catplot(hrv, x="subj", y="sdnn", hue="session", kind='bar')
# ####

# features = ['sdnn', 'lf_hf_ratio', 'csi', 'cvi', 'mean_nni',
#             'nni_50', 'pnni_50', 'sdsd', 'mean_hr']
# palettes = ['viridis', 'Reds', 'Blues', 'Purples', 'Oranges',
#             'plasma', 'inferno', 'Greys', 'Greens']



# # HRV figures
# for f, p in zip(features, palettes):
#     sns.catplot(data=hrv, x="subject", y=f, hue="session", row='period', 
#                 kind='bar', height=5, aspect=2, palette=p)
#     plt.savefig(f+"-all.png")
#     sns.catplot(data=hrv, x="period", y=f, col="session", kind='box',
#                 height=5, aspect=2, palette=p)
#     plt.savefig(f+"-avgsubj.png")
#     sns.catplot(data=hrv, x="period", y=f, kind='box',
#                 height=5, aspect=2, palette=p)
#     plt.title(f+" across subjets and sessions")
#     plt.savefig(f+"-avgsubjsess.png")
#     sns.catplot(data=hrv, x="period", y=f, hue="subject", kind='point',
#                 join=True, dodge=True, errwidth=0.8, capsize=0.05,
#                 height=5, aspect=2, palette=p)
#     plt.title(f+" per subject across sessions")
#     plt.savefig(f+"-avgsess.png")
#     sns.catplot(data=hrv, x="session", y=f, hue="subject", kind='point',
#                 join=True, dodge=True, errwidth=0.8, capsize=0.05,
#                 height=5, aspect=2, palette=p)
#     plt.title(f+" per subject across periods")
#     plt.savefig(f+"-avgperiod.png")
#     sns.catplot(data=hrv, x="session", y=f, kind='box',
#                 height=5, aspect=2, palette=p)
#     plt.title(f+" across subjects and periods")
#     plt.savefig(f+"-avgsubjperiod.png")
#     plt.close("all")

# rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2


# COMPTE RENDU SJSR - Figure pour chaque patient

os.chdir('/Users/justi/Documents/EPF/5A/Semestre 1/Analyse donnees neuro/') 

features = [ 'lf_hf_ratio', 'csi', 'cvi', 'mean_hr']
palettes = ['Reds', 'Blues', 'Purples', 'Greens']


# HRV figures
for f, p in zip(features, palettes):
    sns.catplot(data=hrv, x="subject", y=f, hue="session", row='period', 
                kind='bar', height=5, aspect=2, palette=p)
    plt.savefig(subj + "_" + f +"-all.png")
    
    
    # sns.catplot(data=hrv, x="period", y=f, col="session", kind='box',
    #             height=5, aspect=2, palette=p)
    # plt.savefig(subj + "_" + f+"-avgsubj.png")
    # sns.catplot(data=hrv, x="period", y=f, kind='box',
    #             height=5, aspect=2, palette=p)
    # plt.title(f+" across subjects and sessions")
    # plt.savefig(subj + "_" + f+"-avgsubjsess.png")
    # sns.catplot(data=hrv, x="period", y=f, hue="subject", kind='point',
    #             join=True, dodge=True, errwidth=0.8, capsize=0.05,
    #             height=5, aspect=2, palette=p)
    # plt.title(f+" per subject across sessions")
    # plt.savefig(subj + "_" + f+"-avgsess.png")
    # sns.catplot(data=hrv, x="session", y=f, hue="subject", kind='point',
    #             join=True, dodge=True, errwidth=0.8, capsize=0.05,
    #             height=5, aspect=2, palette=p)
    # plt.title(f+" per subject across periods")
    # plt.savefig(subj + "_" +f+"-avgperiod.png")
    # sns.catplot(data=hrv, x="session", y=f, kind='box',
    #             height=5, aspect=2, palette=p)
    # plt.title(f+" across subjects and periods")
    # plt.savefig(subj + "_" +f+"-avgsubjperiod.png")
    # plt.close("all")
# Correlation between HRV indices
#for f1, f2, in itertools.combinations(features, 2):
#    print (f1, f2)
#    g = sns.JointGrid(x=f1, y=f2, data=hrv)
#    g = g.plot(sns.regplot, sns.distplot)
#    g = g.annotate(rsquare, template="{stat}: {val:.2f}", stat="$R^2$")
#    plt.savefig("correlation-{}-{}-avgperiod.png".format(f1, f2))
#    plt.close()
#
#
## Correlation HRV with questionnaires
#quest = pd.read_excel("snv_sjsr_questionnaire_score.xlsx")
#
#period = ['debut', 'milieu', 'fin']
#q_score = ['IRLS', 'LEEDS', 'RLS_QOL', 'HAD A', 'HAD D']
#
## For each period (debut, milieu, fin)
#for p in period:
#    hrv_p = hrv.groupby("period").get_group(p)
#    df = pd.merge(hrv_p, quest, how='inner')
#    for q in q_score:
#        for f in features:
#            print (f, q, p)
#            g = sns.JointGrid(x=q, y=f, data=df)
#            g = g.plot(sns.regplot, sns.distplot)
#            g = g.annotate(rsquare, template="{stat}: {val:.2f}", stat="$R^2$")
#            plt.savefig("correlation-{}-{}-{}.png".format(f, q, p))
#            plt.close()
#
## for the average of each period
#for q in q_score:
#    for f in features:
#        hrv_p = hrv.groupby(["subject",
#                             "session"])[f].mean().to_frame().reset_index()
#        df = pd.merge(hrv_p, quest, how='inner')
#        print (f, q)
#        g = sns.JointGrid(x=q, y=f, data=df)
#        g = g.plot(sns.regplot, sns.distplot)
#        g = g.annotate(rsquare, template="{stat}: {val:.2f}", stat="$R^2$")
#        plt.savefig("correlation-{}-{}-avgperiod.png".format(f, q))
#        plt.close()
#
## Correlation between questionnaires
#for q1, q2, in itertools.combinations(q_score, 2):
#    print (q1, q2)
#    g = sns.JointGrid(x=q1, y=q2, data=quest)
#    g = g.plot(sns.regplot, sns.distplot)
#    g = g.annotate(rsquare, template="{stat}: {val:.2f}", stat="$R^2$")
#    plt.savefig("correlation-{}-{}-avgperiod.png".format(q1, q2))
#    plt.close()
#
## Questionnaires figures
#palettes_quest = ['YlOrBr', 'PuRd', 'GnBu', 'YlGnBu']
#for q, p, in zip(q_score, palettes_quest):
#    sns.catplot(data=quest, x="session", y=q, hue="subject", kind='point',
#                join=True, dodge=True, errwidth=0.8, capsize=0.05,
#                height=5, aspect=2, palette=p)
#    plt.title(q+" per subject across periods")
#    plt.savefig(q+"-avgperiod.png")
#    sns.catplot(data=quest, x="session", y=q, kind='box',
#                height=5, aspect=2, palette=p)
#    plt.title(q+" across subjects and periods")
#    plt.savefig(q+"-avgsubjperiod.png")
#    plt.close("all")
#    
