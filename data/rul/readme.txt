Battery Remaining Useful Life (RUL)
Predict the RUL of batteries by features based on voltage and current.

About Dataset
The Hawaii Natural Energy Institute examined 14 NMC-LCO 18650 batteries with a nominal capacity of 2.8 Ah, which were cycled over 1000 times at 25°C with a CC-CV charge rate of C/2 rate and discharge rate of 1.5C.

From that source dataset, I created features that showcase the voltage and current behaviour over each cycle. Those features can be used to predict the remaining useful life (RUL) of the batteries. The dataset contains the summary of the 14 batteries.

Variables:

Cycle Index: number of cycle
F1: Discharge Time (s)
F2: Time at 4.15V (s)
F3: Time Constant Current (s)
F4: Decrement 3.6-3.4V (s)
F5: Max. Voltage Discharge (V)
F6: Min. Voltage Charge (V)
F7: Charging Time (s)
Total time (s)
RUL: target
You may check on GitHub how the dataset was built: https://github.com/ignavinuales/Battery_RUL_Prediction