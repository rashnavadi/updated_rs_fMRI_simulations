this is an updated version of simulated rs-fMRI where only pink noise is added and the mean of the x, y and z timeseries are rescaled by 10000, i.e., 10000 was added to the timeseries. 
scale_tc = 25.0; is also used, and timeseries are high pass filtered using highpass(xp, .01, .5).
