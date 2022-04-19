#Ferme/ouvert/ferme/ouvert (3min chacun)

import mne
from mne.epochs import concatenate_epochs
from os import chdir
from os import listdir
from os.path import exists
from mne.io import read_raw_edf
import numpy as np
import matplotlib.pyplot as plt


def get_PPI_event(raw, condition, subject_id, tstart, tstop, hrstart, hrstop):

    cond = {'S1':1, 'S2':2, 'S3':3, 'S4':4,'ST':5}
    subj = {'Herve':2}
            
    s = 3 + 10 * cond [condition] + 100 * subj[subject_id]
    c = 10 * cond[condition] + 100 * subj[subject_id]
    i, e = 1 + c, 2 + c
  
    
      
    C = raw.copy().pick_channels(['ECG'])
    e2, _, _ = mne.preprocessing.find_ecg_events(C, event_id=c,
                           ch_name='ECG',tstart=hrstart)
    if hrstop:
      e2s = np.array(e2[e2[:,0] < hrstop * raw.info['sfreq']])
    
    ev = np.vstack([ e2s])
    event_ids ={'cardio/'+condition+'/'+subject_id : c}
    return ev, event_ids



conf_subj = {'Herve':{'S1':{'start':0, 'stop':180,'ch_drop':(),'ICA_OUT':[0,5,4,3],'reject':40e-6,'HR_Start':180,'HR_Stop':360,'n_comp':0.95},
                      'S2':{'start':180, 'stop':360,'ch_drop':(),'ICA_OUT':[0,5,4,3],'reject':40e-6,'HR_Start':180,'HR_Stop':360,'n_comp':0.95},
                      'S3':{'start':360, 'stop':540,'ch_drop':(),'ICA_OUT':[0,5,4,3],'reject':40e-6,'HR_Start':360,'HR_Stop':510,'n_comp':0.95},
                      'S4':{'start':540, 'stop':720,'ch_drop':(),'ICA_OUT':[0,5,4,3],'reject':40e-6,'HR_Start':540,'HR_Stop':720,'n_comp':0.95},
                      'ST':{'start':0, 'stop':720,'ch_drop':(),'ICA_OUT':[0,5,4,3],'reject':40e-6,'HR_Start':0,'HR_Stop':510,'n_comp':0.95}}}
          

   
    
all_conf_DYSP_REA = {'exclude':['Oro-nasal Pressu','EMG RAT','EMG LAT', 'EOG Left',
                                    'EOG right','EMG Chin','Sound PHONO','POS','PCPAP','LIGHT','Resp Abdominal','Resp Thoracal',
                                    'Resp oro-nasal','UARS','PTL','SaO2','PR','PULSE'],
              'rename':{'EEG Fz-A1':'Fz','EEG Cz-A1':'Cz', 'EEG Pz-A1':'Pz',
                 'EEG C4-A1':'C4', 'EEG T3-A1':'T3', 'EEG C3-A1':'C3','EEG T4-A1':'T4',
                 'EEG O1-A1':'O1','ECG V1':'ECG'},
              'types':{'ECG':'ecg'}}

l_freq = 1.
h_freq = 40.
epoch_min = -0.45
epoch_max = 0.45
baseline_min = -0.2
baseline_max = -0.1
baseline = (baseline_min, baseline_max)
EPOCHS_TMP = []
         
#          
subject = ['Herve']

session = ['S1','S2','S3','S4']


for ii in subject:
    for yy in session:

        path ='C:/Users/herve/OneDrive/Bureau/Hervé/Casque_Donnees_neurophysiologique/' 
        raw_fname, fname = path+ ii +'_'+yy+'.EDF', ii+'_'+yy+'.EDF'
#        condition = fname.split('_')[3].split('.')[0]
        conditions = yy
        raw = read_raw_edf(raw_fname, preload=True, stim_channel=None)
        
        raw.drop_channels(all_conf_DYSP_REA['exclude'])
        raw.rename_channels(all_conf_DYSP_REA['rename'])
        raw.set_channel_types(all_conf_DYSP_REA['types'])
            
            
        l_freq, h_freq = 3., 40.       
        picks = mne.pick_types(raw.info, eeg=True, resp=True)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        raw.set_channel_types({'ECG':'eeg'})
        raw.notch_filter(50.).filter(l_freq, h_freq, method='iir')
          
        ev, event_ids = get_PPI_event(raw, yy, ii,
                        conf_subj[ii][yy]['start'],
                        conf_subj[ii][yy]['stop'],
                        conf_subj[ii][yy]['HR_Start'],
                        conf_subj[ii][yy]['HR_Stop'])
                    
            
        raw.set_channel_types({'ECG':'ecg'})
            
        for ei in event_ids:
                  print("*********", ": il y a",
                     len(ev[ev[:,2] == event_ids[ei]]), 'events',
                     ei.split('/')[0], '**********')
                  
                  
                	
            
        sig = raw.copy()
        sig.set_channel_types({'ECG':'eeg'})
        sig.notch_filter(50.).filter(l_freq, h_freq, method='iir')
        sig.set_channel_types({'ECG':'ecg'})
                         
            #sig.drop_channels(conf_subj[ii][yy]['ch_drop'])
        picks = mne.pick_types(sig.info, eeg=True, resp=True)
            
            
            
            
        #_ = sig.plot()
                
        for ei in event_ids:   
            # PLOT Pression events
            _ = sig.plot(events=ev[ev[:,2] == event_ids[ei]], scalings='auto', n_channels=13,event_color='green',
                       color={'eeg':'steelblue'}, start=500,title="EVENT DETECTION "+ei)
              
                 
            #_ = sig.plot(events=ev[ev[:,2] == event_ids[ei]], scalings='auto', n_channels=13,event_color='green',
            #               color={'eeg':'steelblue', 'ecg':'darkblue', 'exci':'b'}, start=500,title="EVENT DETECTION "+ei)
        
        from mne.preprocessing import ICA
        chdir('C:/Users/herve/OneDrive/Bureau/Hervé/Casque_Donnees_neurophysiologique')
        ica = ICA(n_components=conf_subj[ii][yy]['n_comp'], method='fastica', random_state=0, max_iter=100)
        ica.fit(sig, start = conf_subj[ii][yy]['start'], stop = conf_subj[ii][yy]['stop'])
        ica.plot_components()
        plt.savefig('ICA '+ii+' '+yy+'.png')
         
        sig_ica = ica.apply(sig, exclude=conf_subj[ii][yy]['ICA_OUT'])
        sig_ica.plot(events = ev[ev[:,2] == event_ids[ei]], scalings='auto', n_channels=13, event_color='indianred',
               color={'eeg':'steelblue'}, start=500)
        
        ep = mne.Epochs(sig_ica, ev, event_ids, tmin=epoch_min, tmax=epoch_max,
                        picks=picks, baseline=None, preload=True,
                        event_repeated='drop',
                        reject={'eeg':conf_subj[ii][yy]['reject']})
        ep.plot_drop_log()
        plt.suptitle('Patient'+ii+'-'+yy)
        plt.savefig("droplog-"+ii+"-"+yy+".png")
        
        
        EPOCHS = mne.Epochs(sig_ica, ev, event_ids, tmin=epoch_min, tmax=epoch_max,
                             picks=picks, baseline=None, preload=True, event_repeated='drop' )
       
        EPOCHS['cardio/'+yy+'/'+ii].average().plot(titles="ERP Cardio All "+ii+" "+yy)
        plt.savefig('ERP Cardio All '+ii+' '+yy+'.png')
        
        
        elec = EPOCHS.ch_names
        for e in elec:
            xda = [EPOCHS['cardio/'+yy+'/'+ii].copy().pick_channels([e])]
#        plot_ERP_one_channel([i.average() for i in xda], conditions, [len(i) for i in xda],
#                            ev_tot, subject+'-xDAWN-PERfilter', e, vmin=-4e-6, vmax=6e-6)
    
            mne.write_evokeds('./'+ii+'-'+yy+'-'+e+'-cardio-ave.fif', [i.average() for i in xda])
        
        mne.write_evokeds('./'+ii+'-'+yy+'-cardio-ave.fif', [EPOCHS['cardio/'+yy+'/'+ii].average()])
        
