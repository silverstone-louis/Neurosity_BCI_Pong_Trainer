# Neurosity_BCI_Pong_Trainer
This is a project designed to use 22 channel clinical eeg motor imagery data from the Physio Net EEG Motor Movement/Imagery Dataset to train a Neurosity Crown BCI to play pong. 

The original data can be found here: https://physionet.org/content/eegmmidb/1.0.0/

In order to make use of it with a neurosity crown, you will need to select only those electrodes in the 22 channel dataset that have analagous electrodes in the Neurosity Crown's 8 channel (plus two reference electrodes) setup. 

Feel free to not do that and instead use the script that I wrote and included in the project files, or just use the pretrained model and fine tune it over time by playing pong with the headset and periodically retraining the model on 'suceess data' (neurosity features) output as jsonl by the pong trainer and/or on the lsl data streams (raw brainwaves, psd, Power by Band, events) that the script outputs. In order to record lsl, you'll need to use labrecorder (free - https://github.com/labstreaminglayer/App-LabRecorder) or some other program. I use labrecorder because its simple and open source. 

I'm sure you will find that many of these scripts are non-functional without modification because many contain absolute file paths to locations on the local machine I used for development. 

I'm going to fix it, but I'm only 1 person and this is a hobby project. [I'll delete this note when the references are corrected].

Citations:

https://neurosity.co/crown

Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
